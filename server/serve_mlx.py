"""Anthropic Messages API server for MLX models on Apple Silicon.

Supports both standard MLX models and JANG quantized models.
Implements tool calling for Claude Code compatibility.
"""

import io
import json
import re
import signal
import sys
import threading
import time
import uuid
from pathlib import Path

# Force unbuffered output so logs appear immediately in files
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = sys.stdout
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.generate import generate_step

MODEL_ALIAS = None  # Set from CLI args
MODEL_PATH = None
_model = None
ANALYTICS_FILE = "/tmp/mlx-analytics.json"
_analytics_lock = threading.Lock()


def record_request(n_in, n_out, elapsed_s, model, ttft_s=None):
    tok_s = round(n_out / elapsed_s, 1) if elapsed_s > 0 else 0
    gen_s = elapsed_s - ttft_s if ttft_s is not None else elapsed_s
    gen_tok_s = round(n_out / gen_s, 1) if gen_s > 0 and n_out > 1 else tok_s
    entry = {
        "ts": time.time(),
        "model": model,
        "input_tokens": n_in,
        "output_tokens": n_out,
        "elapsed_s": round(elapsed_s, 2),
        "tok_s": tok_s,
        "ttft_s": round(ttft_s, 2) if ttft_s is not None else None,
        "gen_tok_s": gen_tok_s,
    }
    with _analytics_lock:
        try:
            data = json.loads(Path(ANALYTICS_FILE).read_text()) if Path(ANALYTICS_FILE).exists() else []
        except Exception:
            data = []
        data.append(entry)
        # Keep last 500 requests
        Path(ANALYTICS_FILE).write_text(json.dumps(data[-500:]))
_tokenizer = None
_gpu_lock = threading.Lock()
_loading = False
_load_error = None
_model_size_gb = None  # Expected size, detected from config


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def detect_model_size_gb(model_path):
    """Estimate model size from config.json (num params * bytes per param)."""
    import pathlib
    config_path = pathlib.Path(model_path) / "config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        hidden = cfg.get("hidden_size", 0)
        layers = cfg.get("num_hidden_layers", 0)
        ffn = cfg.get("intermediate_size", hidden * 4)
        heads = cfg.get("num_attention_heads", 1)
        head_dim = hidden // heads if heads else 64
        kv_heads = cfg.get("num_key_value_heads", heads)
        # Rough param count
        params = layers * (4 * hidden * hidden + 3 * ffn * hidden + 2 * hidden * kv_heads * head_dim)
        # Assume ~1 byte/param for 8-bit, 0.5 for 4-bit (detect from quantization config)
        quant_path = pathlib.Path(model_path) / "config.json"
        bits = 8 if "8bit" in str(model_path).lower() else 4
        return params * (bits / 8) / 1e9
    except Exception:
        return None


def load_model_async(model_path):
    """Load model in background thread."""
    global _model, _tokenizer, _loading, _load_error, _model_size_gb
    _loading = True
    _load_error = None
    _model_size_gb = detect_model_size_gb(model_path)
    print(f"[mlx] Loading model from {model_path}... (expected ~{_model_size_gb:.1f} GB)" if _model_size_gb else f"[mlx] Loading model from {model_path}...")
    t0 = time.perf_counter()

    try:
        try:
            import jang_tools
            if jang_tools.is_jang_model(model_path):
                print("[mlx] Detected JANG format")
                _model, _tokenizer = jang_tools.load_for_inference(model_path)
                print(f"[mlx] JANG model loaded in {time.perf_counter() - t0:.1f}s")
                _loading = False
                return
        except ImportError:
            pass

        from mlx_lm import load
        _model, _tokenizer = load(model_path)
        print(f"[mlx] Model loaded in {time.perf_counter() - t0:.1f}s")
    except Exception as e:
        _load_error = str(e)
        print(f"[mlx] Load error: {e}")
    finally:
        _loading = False


def get_eos_ids():
    eos_ids = set()
    eid = getattr(_tokenizer, "eos_token_id", None)
    if eid is not None:
        if isinstance(eid, list):
            eos_ids.update(eid)
        else:
            eos_ids.add(eid)
    return eos_ids


def build_prompt(messages, tools=None):
    """Build prompt using the tokenizer's chat template, with optional tools."""
    chat_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    text_parts.append(json.dumps(block))
                elif block.get("type") == "tool_result":
                    text_parts.append(
                        f"Tool result for {block.get('tool_use_id', 'unknown')}:\n"
                        + (block.get("content", "") if isinstance(block.get("content"), str)
                           else "\n".join(b.get("text", "") for b in block.get("content", []) if b.get("type") == "text"))
                    )
            content = "\n".join(text_parts)

        chat_messages.append({"role": role, "content": content})

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        # Convert Anthropic tool format to OpenAI-style for chat template
        openai_tools = []
        for t in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })
        kwargs["tools"] = openai_tools

    if hasattr(_tokenizer, "apply_chat_template"):
        return _tokenizer.apply_chat_template(chat_messages, **kwargs)
    return "\n".join(m["content"] for m in chat_messages)


def parse_messages(body):
    """Extract flat message list from Anthropic request body."""
    messages = []
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = " ".join(b["text"] for b in system if b.get("type") == "text")
            if text:
                messages.append({"role": "system", "content": text})
    messages.extend(body.get("messages", []))
    return messages


def parse_tool_calls(text):
    """Parse tool calls from model output.

    Supports multiple formats:
    - Qwen/OpenAI style: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    - JSON blocks with function_call or tool_calls keys
    - Direct JSON tool call objects
    """
    tool_calls = []

    # Qwen-style <tool_call> or <tools> tags
    tc_pattern = re.compile(r'<(?:tool_call|tools)>\s*(.*?)\s*</(?:tool_call|tools)>', re.DOTALL)
    for match in tc_pattern.finditer(text):
        try:
            tc = json.loads(match.group(1))
            name = tc.get("name", tc.get("function", {}).get("name", ""))
            args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))
            if isinstance(args, str):
                args = json.loads(args)
            if name:
                tool_calls.append({"name": name, "input": args})
        except json.JSONDecodeError:
            continue

    # Also try Hermes/generic function call format
    if not tool_calls:
        fc_pattern = re.compile(r'\{"(?:name|function)":\s*"(\w+)".*?"(?:arguments|parameters|input)":\s*(\{[^}]*\})', re.DOTALL)
        for match in fc_pattern.finditer(text):
            try:
                name = match.group(1)
                args = json.loads(match.group(2))
                tool_calls.append({"name": name, "input": args})
            except (json.JSONDecodeError, IndexError):
                continue

    return tool_calls


def split_response(text):
    """Split model response into thinking, text, and tool_call parts."""
    thinking = ""
    content_text = text

    # Extract thinking
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        content_text = parts[1].strip()
    elif text.startswith("<think>"):
        content_text = text.replace("<think>", "").strip()

    # Check for tool calls
    tool_calls = parse_tool_calls(content_text)

    if tool_calls:
        # Remove all tool call markup from visible text
        clean_text = re.sub(r'<(?:tool_call|tools)>.*?</(?:tool_call|tools)>', '', content_text, flags=re.DOTALL)
        clean_text = re.sub(r'<\|tool_call\|>.*?$', '', clean_text, flags=re.DOTALL)
        # Remove bare JSON tool call objects that match what we parsed
        for tc in tool_calls:
            clean_text = re.sub(
                r'\{[^{}]*"name"\s*:\s*"' + re.escape(tc["name"]) + r'"[^{}]*\}',
                '', clean_text
            )
        clean_text = clean_text.strip()
    else:
        clean_text = content_text

    return thinking, clean_text, tool_calls


def generate_text(tokens, sampler, eos_ids, max_tokens):
    """Run generation and return (token_ids, decoded_text)."""
    generated = []
    with _gpu_lock:
        for tok, _ in generate_step(
            prompt=mx.array(tokens), model=_model,
            max_tokens=max_tokens, sampler=sampler,
        ):
            t = tok.item() if hasattr(tok, "item") else int(tok)
            if t in eos_ids:
                break
            generated.append(t)
    return generated, _tokenizer.decode(generated)


def generate_streaming(tokens, sampler, eos_ids, max_tokens):
    """Yield decoded text chunks token by token using the tokenizer's streaming detokenizer."""
    generated = []
    detokenizer = _tokenizer.detokenizer
    detokenizer.reset()
    with _gpu_lock:
        for tok, _ in generate_step(
            prompt=mx.array(tokens), model=_model,
            max_tokens=max_tokens, sampler=sampler,
        ):
            t = tok.item() if hasattr(tok, "item") else int(tok)
            if t in eos_ids:
                break
            generated.append(t)
            detokenizer.add_token(t)
            delta = detokenizer.last_segment
            if delta:
                yield delta, generated
    detokenizer.finalize()
    if detokenizer.last_segment:
        yield detokenizer.last_segment, generated


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {format % args}")

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/v1/models"):
            self._json_response(200, {
                "object": "list",
                "data": [{"id": MODEL_ALIAS, "object": "model", "created": int(time.time())}],
            })
        elif path == "/health":
            self._json_response(200, {"status": "ok"})
        elif path == "/stats":
            try:
                active = mx.get_active_memory() / 1e9
                peak = mx.get_peak_memory() / 1e9
                cache = mx.get_cache_memory() / 1e9
            except Exception:
                active = peak = cache = 0.0
            pct = None
            if _model_size_gb and _model_size_gb > 0:
                pct = min(100, round(active / _model_size_gb * 100))
            if MODEL_PATH:
                p = Path(MODEL_PATH)
                # HF cache: ...models--org--name/snapshots/hash → extract name
                parts = p.parts
                for part in parts:
                    if part.startswith("models--"):
                        model_display = part.split("--")[-1]
                        break
                else:
                    model_display = p.name
            else:
                model_display = MODEL_ALIAS
            self._json_response(200, {
                "model": MODEL_ALIAS,
                "model_display": model_display,
                "loading": _loading,
                "load_error": _load_error,
                "mlx_active_gb": round(active, 2),
                "mlx_peak_gb": round(peak, 2),
                "mlx_cache_gb": round(cache, 2),
                "model_size_gb": round(_model_size_gb, 1) if _model_size_gb else None,
                "load_pct": pct,
            })
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/messages":
            self._handle_messages()
        elif path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._json_response(404, {"error": "not found"})

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _handle_messages(self):
        if _model is None:
            self._json_response(503, {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Model is still loading, please retry in a moment."},
            })
            return
        try:
            body = self._read_body()
            model_name = body.get("model", MODEL_ALIAS)
            is_stream = body.get("stream", False)
            messages = parse_messages(body)
            tools = body.get("tools")
            prompt = build_prompt(messages, tools)
            max_tokens = body.get("max_tokens", 4096)

            tokens = _tokenizer.encode(prompt)
            sampler = make_sampler(
                temp=body.get("temperature", 0.7),
                top_p=body.get("top_p", 1.0),
            )
            eos_ids = get_eos_ids()

            if is_stream:
                self._stream_response(tokens, sampler, eos_ids, max_tokens, model_name, tools)
            else:
                self._sync_response(tokens, sampler, eos_ids, max_tokens, model_name, tools)

        except Exception as e:
            import traceback, sys
            traceback.print_exc()
            sys.stdout.flush()
            try:
                self._json_response(500, {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                })
            except Exception:
                pass

    def _handle_chat_completions(self):
        try:
            body = self._read_body()
            messages = body.get("messages", [])
            prompt = build_prompt(messages)
            max_tokens = body.get("max_tokens", 4096)
            tokens = _tokenizer.encode(prompt)
            sampler = make_sampler(
                temp=body.get("temperature", 0.7),
                top_p=body.get("top_p", 1.0),
            )
            eos_ids = get_eos_ids()
            generated, result = generate_text(tokens, sampler, eos_ids, max_tokens)
            self._json_response(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "model": MODEL_ALIAS,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": result}}],
                "usage": {"prompt_tokens": len(tokens), "completion_tokens": len(generated), "total_tokens": len(tokens) + len(generated)},
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})

    def _sync_response(self, tokens, sampler, eos_ids, max_tokens, model_name, tools=None):
        t0 = time.perf_counter()
        generated, result = generate_text(tokens, sampler, eos_ids, max_tokens)
        thinking, text, tool_calls = split_response(result)

        content = []
        if thinking:
            content.append({"type": "thinking", "thinking": thinking})
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            content.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": tc["name"],
                "input": tc["input"],
            })

        if not content:
            content.append({"type": "text", "text": ""})

        stop_reason = "tool_use" if tool_calls else "end_turn"
        record_request(len(tokens), len(generated), time.perf_counter() - t0, model_name)

        self._json_response(200, {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": model_name,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": len(tokens), "output_tokens": len(generated)},
        })

    def _stream_response(self, tokens, sampler, eos_ids, max_tokens, model_name, tools=None):
        """True per-token streaming. Tool calls are buffered and sent at the end."""
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        input_tokens = len(tokens)
        t0 = time.perf_counter()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        write_lock = threading.Lock()

        def sse(event_type, data):
            with write_lock:
                self.wfile.write(f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()

        def keepalive(stop_event):
            """Send SSE comment pings every 5s so the client doesn't drop the connection."""
            while not stop_event.wait(5):
                try:
                    with write_lock:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                except Exception:
                    break

        stop_ping = threading.Event()
        ping_thread = threading.Thread(target=keepalive, args=(stop_ping,), daemon=True)
        ping_thread.start()

        sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "model": model_name, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })

        block_index = 0
        full_text = ""
        last_generated = []
        in_tool_call = False
        text_block_started = False
        ttft = None

        for delta, last_generated in generate_streaming(tokens, sampler, eos_ids, max_tokens):
            if ttft is None:
                ttft = time.perf_counter() - t0
                stop_ping.set()  # stop keep-alive once tokens start flowing
            full_text += delta

            # Detect start of tool call — stop streaming visible text
            if "<tool_call>" in full_text or "<|tool_call|>" in full_text:
                in_tool_call = True

            if not in_tool_call:
                # Skip <think> tags from visible stream
                visible = delta
                if not text_block_started and visible.strip():
                    # Strip leading think tag if present
                    if "<think>" in full_text and "</think>" not in full_text:
                        continue  # still in thinking block
                    sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    text_block_started = True

                if text_block_started:
                    sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": visible},
                    })

        stop_ping.set()  # ensure ping thread stops if it wasn't stopped by first token

        # Parse final output for tool calls / thinking
        thinking, text, tool_calls = split_response(full_text)

        # If we never opened a text block (e.g. pure tool call response), open one
        if not text_block_started and (text or not tool_calls):
            sse("content_block_start", {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {"type": "text", "text": text},
            })
            text_block_started = True

        if text_block_started:
            sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1

        elapsed = time.perf_counter() - t0
        n_out = len(last_generated)
        gen_s = elapsed - ttft if ttft else elapsed
        gen_tok_s = n_out / gen_s if gen_s > 0 and n_out > 1 else 0
        print(f"[{time.strftime('%H:%M:%S')}] streamed {n_out} tokens | ttft={ttft:.2f}s | gen={gen_tok_s:.1f} tok/s | total={elapsed:.1f}s")
        record_request(input_tokens, n_out, elapsed, model_name, ttft_s=ttft)

        # Send tool_use blocks
        for tc in tool_calls:
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
            sse("content_block_start", {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {"type": "tool_use", "id": tool_id, "name": tc["name"], "input": {}},
            })
            sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(tc["input"])},
            })
            sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1

        stop_reason = "tool_use" if tool_calls else "end_turn"

        sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": n_out},
        })
        sse("message_stop", {"type": "message_stop"})


def main():
    global MODEL_ALIAS
    import argparse
    parser = argparse.ArgumentParser(description="MLX Anthropic Messages API server")
    parser.add_argument("--model", required=True, help="Path to MLX model directory")
    parser.add_argument("--model-alias", default=None, help="Model name exposed via API")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    global MODEL_PATH
    MODEL_ALIAS = args.model_alias or args.model.rstrip("/").split("/")[-1]
    MODEL_PATH = args.model

    # Start HTTP server first so /stats works during loading
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mlx] Serving '{MODEL_ALIAS}' on http://{args.host}:{args.port}")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Load model in background
    t = threading.Thread(target=load_model_async, args=(args.model,), daemon=True)
    t.start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
