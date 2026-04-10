"""Smart hybrid routing proxy for Claude Code.

Routes requests to the cheapest backend that can handle them:

  Tier 0 — LOCAL:   Tool continuations, simple follow-ups → local Gemma/Qwen (free, fastest)
  Tier 1 — SONNET:  Coding, implementation, bug fixes → Claude Sonnet via OpenRouter
  Tier 2 — OPUS:    Planning, architecture, complex reasoning → Claude Opus via OpenRouter

Classification is pure heuristics — zero AI overhead, microsecond routing decisions.

Typical Claude Code session: ~70% tool continuations → 70% of requests are FREE + LOCAL.

Architecture:
  Claude Code → :5005 (this proxy)
                  ├── local model on :5006 (Gemma 4 / Qwen via llama-server or MLX)
                  ├── Sonnet via OpenRouter
                  └── Opus via OpenRouter

Usage:
  python3 smart-proxy.py <port> <local_port> <openrouter_api_key>
  python3 smart-proxy.py 5005 5006 sk-or-...

Stats:  curl http://localhost:5005/stats
"""
import http.server
import json
import re
import sys
import time
import threading
import urllib.request
import urllib.error
from socketserver import ThreadingMixIn

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5005
LOCAL_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5006
API_KEY = sys.argv[3] if len(sys.argv) > 3 else ""

LOCAL_BASE = f"http://127.0.0.1:{LOCAL_PORT}"
OPENROUTER_BASE = "https://openrouter.ai/api"

# Model config
MODELS = {
    "local":  "local llm",                      # whatever is running on LOCAL_PORT
    "sonnet": "anthropic/claude-sonnet-4",
    "opus":   "anthropic/claude-opus-4",
}

# Rough cost per 1K tokens (blended input+output)
COST_PER_1K = {"local": 0.0, "sonnet": 0.006, "opus": 0.045}

# --- Stats ---
_lock = threading.Lock()
_stats = {"local": 0, "sonnet": 0, "opus": 0, "requests": 0, "start_time": time.time()}
_tokens = {"local": 0, "sonnet": 0, "opus": 0}
_cost = {"local": 0.0, "sonnet": 0.0, "opus": 0.0}
_latency = {"local": [], "sonnet": [], "opus": []}  # last 20 per tier
_local_available = False


def check_local():
    """Check if local model is healthy."""
    global _local_available
    try:
        req = urllib.request.Request(f"{LOCAL_BASE}/health", method="GET")
        resp = urllib.request.urlopen(req, timeout=2)
        _local_available = resp.status == 200
    except Exception:
        _local_available = False


def start_local_health_check():
    def loop():
        while True:
            check_local()
            time.sleep(10)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    check_local()


# --- Classification ---

# Opus trigger patterns (compiled once)
_OPUS_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bplan\s+(the|this|how|a)\b",
        r"\barchitect",
        r"\bdesign\s+(the|a|this|an)\s+\w+",
        r"\brefactor\s+(the|this|all|entire)",
        r"\brewrite\s+(the|this|all|entire)",
        r"\bmigrat\w+\s+(strategy|plan|from|to)",
        r"\btrade.?off",
        r"\breview\s+(the|this|my)\s+(approach|design|architecture|plan)",
        r"\bexplain\s+(how|why)\s+(the\s+)?(overall|entire|system|architecture)",
        r"\bwhat\s+should\s+(the|our)\s+(approach|strategy|architecture)",
        r"\bpropose\s+(a|an|the)",
        r"\bbreak\s+(this|it)\s+down",
    ]
]

# Sonnet trigger patterns — coding tasks that need real reasoning
_SONNET_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(fix|debug|solve|investigate)\s+(this|the|a)",
        r"\b(add|implement|create|build|write)\s+(a|an|the|this)",
        r"\b(update|change|modify)\s+(the|this|a)",
        r"\bwhat('s|\s+is)\s+(wrong|broken|failing)",
        r"\bwhy\s+(is|does|doesn't|isn't)",
        r"\bhow\s+(do|can|should)\s+(I|we)",
    ]
]


def classify(body):
    """Classify request → (tier, reason). Pure heuristics, no AI."""
    messages = body.get("messages", [])
    if not messages:
        return "sonnet", "empty"

    # Find last user and assistant messages
    last_user = None
    last_assistant = None
    for msg in reversed(messages):
        if msg["role"] == "user" and last_user is None:
            last_user = msg
        elif msg["role"] == "assistant" and last_assistant is None:
            last_assistant = msg
        if last_user and last_assistant:
            break

    # --- LOCAL: Tool result continuations ---
    if last_user and _local_available:
        content = last_user.get("content", "")
        if isinstance(content, list):
            tool_results = sum(
                1 for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            )
            text_blocks = [
                b for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text_len = sum(len(b.get("text", "")) for b in text_blocks)

            # Pure tool results (no user text)
            if tool_results > 0 and len(text_blocks) == 0:
                return "local", f"tool_results={tool_results}"

            # Tool results with minimal user text (e.g. "yes", "ok", "continue")
            if tool_results > 0 and text_len < 80:
                return "local", f"tool_results={tool_results},short_text={text_len}"

    # Extract user text for pattern matching
    user_text = ""
    if last_user:
        content = last_user.get("content", "")
        if isinstance(content, str):
            user_text = content
        elif isinstance(content, list):
            user_text = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )

    # --- OPUS: Complex reasoning signals ---
    for pat in _OPUS_PATTERNS:
        m = pat.search(user_text)
        if m:
            return "opus", f"pattern:{m.group()}"

    # System prompt signals
    system = body.get("system", "")
    if isinstance(system, list):
        system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    if isinstance(system, str):
        sys_lower = system.lower()
        if any(kw in sys_lower for kw in ["plan mode", "architect", "design review"]):
            return "opus", "system_prompt"

    # Very long user message with no tool results = initial complex task
    if len(user_text) > 1000 and len(messages) <= 3:
        return "opus", f"long_initial={len(user_text)}"

    # --- SONNET: Coding tasks ---
    for pat in _SONNET_PATTERNS:
        m = pat.search(user_text)
        if m:
            return "sonnet", f"pattern:{m.group()}"

    # Short user text in ongoing conversation (not tool results) → local if available
    if _local_available and len(user_text) < 200 and len(messages) > 4:
        return "local", "short_continuation"

    # Default
    return "sonnet", "default"


def sanitize_messages(body):
    """Strip thinking blocks from message history to avoid cross-model signature errors."""
    messages = body.get("messages", [])
    cleaned = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            filtered = [
                block for block in content
                if not (isinstance(block, dict) and block.get("type") == "thinking")
            ]
            if filtered:
                cleaned.append({**msg, "content": filtered})
            elif msg["role"] == "assistant":
                cleaned.append({**msg, "content": [{"type": "text", "text": ""}]})
            else:
                cleaned.append(msg)
        else:
            cleaned.append(msg)
    body["messages"] = cleaned
    return body


def estimate_tokens(body):
    """Rough token count from message content."""
    return len(json.dumps(body.get("messages", []))) // 4


# --- Server ---

class ThreadedServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._respond(200, {
                "object": "list",
                "data": [{"id": "smart-router", "object": "model", "created": 0}],
            })
        elif self.path == "/stats":
            self._serve_stats()
        else:
            self._respond(404, {"error": "not found"})

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw)
        except Exception:
            body = {}

        tier, reason = classify(body)
        body = sanitize_messages(body)
        t0 = time.time()
        approx_tok = estimate_tokens(body)

        # Track
        with _lock:
            _stats[tier] += 1
            _stats["requests"] += 1
            _tokens[tier] += approx_tok
            _cost[tier] += approx_tok / 1000 * COST_PER_1K[tier]
            req_num = _stats["requests"]

        pct_free = round(_stats["local"] / req_num * 100) if req_num else 0
        total_cost = sum(_cost.values())
        opus_cost = req_num * approx_tok / 1000 * COST_PER_1K["opus"]  # if all were opus

        tier_char = {"local": "L", "sonnet": "S", "opus": "O"}[tier]
        print(
            f"[{tier_char}] #{req_num} {reason:<30s} "
            f"[{_stats['local']}L {_stats['sonnet']}S {_stats['opus']}O] "
            f"{pct_free}% local  ${total_cost:.2f} spent",
            flush=True,
        )

        if tier == "local":
            self._forward_local(body, tier, t0)
        else:
            self._forward_openrouter(body, tier, t0)

    def _forward_local(self, body, tier, t0):
        """Forward to local model on LOCAL_PORT."""
        body["model"] = MODELS["local"]
        encoded = json.dumps(body).encode()
        is_stream = body.get("stream", False)

        target = f"{LOCAL_BASE}/v1/messages"
        req = urllib.request.Request(
            target, data=encoded, method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            resp = urllib.request.urlopen(req, timeout=300)
            self._relay_response(resp, is_stream, tier, t0)
        except Exception as e:
            # Local model failed — fall back to Sonnet
            print(f"[!] Local failed ({e}), falling back to Sonnet", flush=True)
            with _lock:
                _stats["local"] -= 1
                _stats["sonnet"] += 1
            self._forward_openrouter(body, "sonnet", t0)

    def _forward_openrouter(self, body, tier, t0):
        """Forward to OpenRouter."""
        body["model"] = MODELS[tier]
        encoded = json.dumps(body).encode()
        is_stream = body.get("stream", False)

        target = OPENROUTER_BASE + "/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": "https://claudecode.local",
            "X-Title": "Claude Code Smart Router",
        }
        req = urllib.request.Request(target, data=encoded, headers=headers, method="POST")

        try:
            resp = urllib.request.urlopen(req, timeout=300)
            self._relay_response(resp, is_stream, tier, t0)
        except urllib.error.HTTPError as e:
            error_body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_body)))
            self.end_headers()
            self.wfile.write(error_body)
        except Exception as e:
            msg = json.dumps({"error": str(e)}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    def _relay_response(self, resp, is_stream, tier, t0):
        """Relay upstream response back to client."""
        if is_stream:
            self.send_response(resp.status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        else:
            result = resp.read()
            self.send_response(resp.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        # Record latency
        elapsed = time.time() - t0
        with _lock:
            lat = _latency[tier]
            lat.append(elapsed)
            if len(lat) > 20:
                lat.pop(0)

    def _serve_stats(self):
        with _lock:
            total = _stats["requests"] or 1
            uptime_h = (time.time() - _stats["start_time"]) / 3600
            total_cost = sum(_cost.values())
            all_opus_cost = sum(_tokens.values()) / 1000 * COST_PER_1K["opus"]
            savings = all_opus_cost - total_cost

            def avg_lat(tier):
                vals = _latency[tier]
                return round(sum(vals) / len(vals), 2) if vals else None

            data = {
                "uptime_hours": round(uptime_h, 1),
                "total_requests": _stats["requests"],
                "routing": {
                    "local": {"count": _stats["local"], "pct": round(_stats["local"] / total * 100, 1)},
                    "sonnet": {"count": _stats["sonnet"], "pct": round(_stats["sonnet"] / total * 100, 1)},
                    "opus": {"count": _stats["opus"], "pct": round(_stats["opus"] / total * 100, 1)},
                },
                "tokens": {k: v for k, v in _tokens.items()},
                "cost": {
                    "actual": round(total_cost, 4),
                    "if_all_opus": round(all_opus_cost, 4),
                    "saved": round(savings, 4),
                    "saving_pct": round(savings / all_opus_cost * 100, 1) if all_opus_cost > 0 else 0,
                },
                "avg_latency_s": {
                    "local": avg_lat("local"),
                    "sonnet": avg_lat("sonnet"),
                    "opus": avg_lat("opus"),
                },
                "local_model_available": _local_available,
            }
        self._respond(200, data)

    def _respond(self, code, data):
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"╔══════════════════════════════════════╗", flush=True)
    print(f"║       Smart Router v1.0              ║", flush=True)
    print(f"╠══════════════════════════════════════╣", flush=True)
    print(f"║  Port:    {PORT:<26d} ║", flush=True)
    print(f"║  Local:   :{LOCAL_PORT} → {MODELS['local']:<15s} ║", flush=True)
    print(f"║  Sonnet:  OpenRouter                 ║", flush=True)
    print(f"║  Opus:    OpenRouter                 ║", flush=True)
    print(f"║  Stats:   /stats                     ║", flush=True)
    print(f"╚══════════════════════════════════════╝", flush=True)

    start_local_health_check()
    server = ThreadedServer(("127.0.0.1", PORT), Handler)
    server.serve_forever()
