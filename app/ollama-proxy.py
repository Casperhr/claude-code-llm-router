#!/usr/bin/env python3
"""
Proxy: Anthropic Messages API → Ollama OpenAI-compatible API.
Handles tool_use/tool_result translation and streaming.
Usage: ollama-anthropic-proxy.py [port] [model]
Defaults: port=4000, model=gemma4:e4b
"""
import json
import http.server
import urllib.request
import urllib.error
import uuid

import sys

OLLAMA_BASE = "http://localhost:11434"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
DEFAULT_MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemma4:e4b"


def anthropic_tools_to_openai(tools):
    """Convert Anthropic tool definitions to OpenAI function calling format."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def anthropic_messages_to_openai(messages):
    """Convert Anthropic messages (with tool_use/tool_result blocks) to OpenAI format."""
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        # List of content blocks
        tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]

        if tool_results:
            # Each tool_result becomes a separate "tool" role message
            for tr in tool_results:
                tr_content = tr.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = " ".join(
                        b.get("text", "") for b in tr_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                result.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_use_id"],
                    "content": tr_content,
                })
        elif tool_uses:
            # Assistant message with tool calls
            text = " ".join(b.get("text", "") for b in text_blocks)
            result.append({
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {
                        "id": tu["id"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu.get("input", {})),
                        },
                    }
                    for tu in tool_uses
                ],
            })
        else:
            # Regular text message
            text = " ".join(b.get("text", "") for b in text_blocks)
            result.append({"role": role, "content": text})

    return result


def openai_response_to_anthropic(oai, model):
    """Convert OpenAI chat completion response to Anthropic Messages API format."""
    choice = oai["choices"][0]
    msg = choice["message"]
    usage = oai.get("usage", {})
    finish_reason = choice.get("finish_reason", "stop")

    content = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})

    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc["function"]
        try:
            inputs = json.loads(fn["arguments"])
        except (json.JSONDecodeError, TypeError):
            inputs = {"raw": fn.get("arguments", "")}
        content.append({
            "type": "tool_use",
            "id": tc["id"] if tc["id"].startswith("toolu_") else f"toolu_{tc['id']}",
            "name": fn["name"],
            "input": inputs,
        })

    stop_reason = "tool_use" if tool_calls else "end_turn"

    return {
        "id": oai.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/model":
            self._json(200, {"model": DEFAULT_MODEL})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        # Always use the model the proxy was started with — ignore what the client sends
        model = DEFAULT_MODEL
        stream = body.get("stream", False)

        # Build OpenAI request
        openai_body = {
            "model": model,
            "messages": anthropic_messages_to_openai(body.get("messages", [])),
            "stream": stream,
        }
        if "max_tokens" in body:
            openai_body["max_tokens"] = body["max_tokens"]
        if "temperature" in body:
            openai_body["temperature"] = body["temperature"]

        # Add system message at front if present
        system = body.get("system")
        if system:
            if isinstance(system, list):
                system = " ".join(b["text"] for b in system if b.get("type") == "text")
            openai_body["messages"].insert(0, {"role": "system", "content": system})

        # Add tools if present
        oai_tools = anthropic_tools_to_openai(body.get("tools"))
        if oai_tools:
            openai_body["tools"] = oai_tools
            openai_body["tool_choice"] = "auto"

        payload = json.dumps(openai_body).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.URLError as e:
            self._json(502, {"error": {"message": str(e), "type": "proxy_error"}})
            return

        if stream:
            self._handle_stream(resp, model)
        else:
            oai = json.loads(resp.read())
            resp.close()
            self._json(200, openai_response_to_anthropic(oai, model))

    def _handle_stream(self, resp, model):
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def sse(event, data):
            line = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            self.wfile.write(line.encode())
            self.wfile.flush()

        sse("message_start", {
            "type": "message_start",
            "message": {"id": msg_id, "type": "message", "role": "assistant",
                        "content": [], "model": model, "stop_reason": None,
                        "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}},
        })

        # Collect full response for tool call detection
        chunks = []
        try:
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str != "[DONE]":
                        try:
                            chunks.append(json.loads(data_str))
                        except json.JSONDecodeError:
                            pass
        finally:
            resp.close()

        # Check if any chunk has tool_calls
        all_tool_calls = []
        text_parts = []
        for chunk in chunks:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                text_parts.append(delta["content"])
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    while len(all_tool_calls) <= idx:
                        all_tool_calls.append({"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        all_tool_calls[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        all_tool_calls[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        all_tool_calls[idx]["arguments"] += fn["arguments"]

        full_text = "".join(text_parts)

        if full_text:
            sse("content_block_start", {"type": "content_block_start", "index": 0,
                                         "content_block": {"type": "text", "text": ""}})
            sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                         "delta": {"type": "text_delta", "text": full_text}})
            sse("content_block_stop", {"type": "content_block_stop", "index": 0})

        for i, tc in enumerate(all_tool_calls):
            block_idx = i + (1 if full_text else 0)
            try:
                inputs = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                inputs = {"raw": tc["arguments"]}
            tool_id = tc["id"] if tc["id"].startswith("toolu_") else f"toolu_{tc['id']}"
            sse("content_block_start", {"type": "content_block_start", "index": block_idx,
                                         "content_block": {"type": "tool_use", "id": tool_id,
                                                           "name": tc["name"], "input": {}}})
            sse("content_block_delta", {"type": "content_block_delta", "index": block_idx,
                                         "delta": {"type": "input_json_delta",
                                                   "partial_json": tc["arguments"]}})
            sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})

        stop_reason = "tool_use" if all_tool_calls else "end_turn"
        sse("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                               "usage": {"output_tokens": len(text_parts)}})
        sse("message_stop", {"type": "message_stop"})

    def _json(self, status, data):
        out = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    import socketserver

    class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = ThreadedServer(("127.0.0.1", PORT), ProxyHandler)
    print(f"Ollama-Anthropic proxy on port {PORT} → {DEFAULT_MODEL}", flush=True)
    server.serve_forever()
