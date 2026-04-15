"""Cloud Smart Router — all models via OpenRouter, zero local dependencies.

Routes requests to the cheapest cloud model that can handle them:

  Tier 0 — QWEN:    Tool continuations, simple follow-ups → Qwen3 32B ($0.16/M)
  Tier 1 — FLASH:   General coding, moderate tasks → Gemini 2.5 Flash ($1.40/M)
  Tier 2 — SONNET:  Complex implementation, debugging → Claude Sonnet ($6/M)
  Tier 3 — OPUS:    Planning, architecture, hard reasoning → Claude Opus ($45/M)

~85% cost reduction vs all-Opus. No local model needed.

Usage:
  python3 cloud-smart-proxy.py <port> <openrouter_api_key>
  python3 cloud-smart-proxy.py 5005 sk-or-...

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
API_KEY = sys.argv[2] if len(sys.argv) > 2 else ""

OPENROUTER_BASE = "https://openrouter.ai/api"
LOCAL_BASE = "http://localhost:4000"  # Ollama proxy (gemma4:e4b)

# Model tiers
MODELS = {
    "local":  "gemma4:e4b",                           # Local Ollama — free
    "flash":  "google/gemini-2.5-flash",              # 1M context
    "sonnet": "anthropic/claude-sonnet-4",             # 250K context
    "opus":   "anthropic/claude-opus-4",               # 1M context
}

TIER_LABELS = {
    "local": "L",
    "flash": "F",
    "sonnet": "S",
    "opus": "O",
}

# Cost per 1M tokens (blended input+output estimate)
COST_PER_M = {"local": 0.0, "flash": 1.40, "sonnet": 6.0, "opus": 45.0}

def _check_local_available():
    try:
        req = urllib.request.Request(f"{LOCAL_BASE}/health")
        urllib.request.urlopen(req, timeout=1)
        return True
    except Exception:
        return False

# --- Stats ---
_lock = threading.Lock()
_stats = {t: 0 for t in MODELS}
_stats["requests"] = 0
_stats["start_time"] = time.time()
_tokens = {t: 0 for t in MODELS}
_cost = {t: 0.0 for t in MODELS}
_latency = {t: [] for t in MODELS}


# --- Classification ---

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

_SONNET_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(fix|debug|solve|investigate)\s+(this|the|a)",
        r"\b(add|implement|create|build|write)\s+(a|an|the|this)\s+\w+",
        r"\b(update|change|modify)\s+(the|this|a)\s+\w+",
        r"\bwhat('s|\s+is)\s+(wrong|broken|failing)",
        r"\bwhy\s+(is|does|doesn't|isn't)",
        r"\bhow\s+(do|can|should)\s+(I|we)",
    ]
]

_FLASH_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(rename|move|copy|delete)\s+(the|this|a)\s+\w+",
        r"\b(format|lint|clean\s*up)\b",
        r"\b(add|remove)\s+(an?\s+)?(import|comment|log)",
        r"\brun\s+(the\s+)?(test|build|lint)",
    ]
]


def classify(body):
    """Classify request → (tier, reason). Pure heuristics, microsecond decisions."""
    messages = body.get("messages", [])
    if not messages:
        return "sonnet", "empty"

    last_user = None
    last_assistant = None
    for msg in reversed(messages):
        if msg["role"] == "user" and last_user is None:
            last_user = msg
        elif msg["role"] == "assistant" and last_assistant is None:
            last_assistant = msg
        if last_user and last_assistant:
            break

    # --- QWEN: Tool result continuations ---
    if last_user:
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

            # Pure tool results — route local if available, else flash
            if tool_results > 0 and len(text_blocks) == 0:
                tier = "local" if _check_local_available() else "flash"
                return tier, f"tool_results={tool_results}"

            # Tool results + short user text ("yes", "ok", "continue")
            if tool_results > 0 and text_len < 80:
                tier = "local" if _check_local_available() else "flash"
                return tier, f"tool_results={tool_results},text={text_len}"

    # Extract user text
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

    # --- OPUS: Complex reasoning ---
    for pat in _OPUS_PATTERNS:
        m = pat.search(user_text)
        if m:
            return "opus", f"pattern:{m.group()}"

    # System prompt signals
    system = body.get("system", "")
    if isinstance(system, list):
        system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    if isinstance(system, str) and any(
        kw in system.lower() for kw in ["plan mode", "architect", "design review"]
    ):
        return "opus", "system_prompt"

    # Long initial task description
    if len(user_text) > 1000 and len(messages) <= 3:
        return "opus", f"long_initial={len(user_text)}"

    # --- SONNET: Coding tasks needing real reasoning ---
    for pat in _SONNET_PATTERNS:
        m = pat.search(user_text)
        if m:
            return "sonnet", f"pattern:{m.group()}"

    # --- FLASH: Simple tasks, short continuations ---
    for pat in _FLASH_PATTERNS:
        m = pat.search(user_text)
        if m:
            return "flash", f"pattern:{m.group()}"

    # Short continuation in ongoing conversation
    if len(user_text) < 200 and len(messages) > 4:
        tier = "local" if _check_local_available() else "flash"
        return tier, f"short_continuation={len(user_text)}"

    # Default
    return "flash", "default"


def sanitize_messages(body):
    """Strip thinking blocks from message history to avoid cross-model signature errors.

    When a non-Anthropic model (Qwen, Gemini) produces thinking blocks, their signatures
    are invalid for Anthropic's API. Remove them before forwarding.
    """
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
                # Don't drop assistant messages entirely — keep with empty text
                cleaned.append({**msg, "content": [{"type": "text", "text": ""}]})
            else:
                cleaned.append(msg)
        else:
            cleaned.append(msg)
    body["messages"] = cleaned
    return body


def estimate_tokens(body):
    return len(json.dumps(body.get("messages", []))) // 4


# --- Server ---

class ThreadedServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/model":
            self._respond(200, {"model": "Cloud Router"})
        elif self.path == "/v1/models":
            self._respond(200, {
                "object": "list",
                "data": [{"id": "cloud-smart-router", "object": "model", "created": 0}],
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
        model = MODELS[tier]
        body = sanitize_messages(body)
        body["model"] = model
        # Strip Claude Code features OpenRouter doesn't support
        for key in ("betas", "anthropic_beta", "context_management", "output_config", "thinking"):
            body.pop(key, None)
        t0 = time.time()
        approx_tok = estimate_tokens(body)

        with _lock:
            _stats[tier] += 1
            _stats["requests"] += 1
            _tokens[tier] += approx_tok
            _cost[tier] += approx_tok / 1_000_000 * COST_PER_M[tier]
            req_num = _stats["requests"]

        total_cost = sum(_cost.values())
        opus_cost = sum(_tokens.values()) / 1_000_000 * COST_PER_M["opus"]
        saved = opus_cost - total_cost
        label = TIER_LABELS[tier]

        print(
            f"[{label}] #{req_num} {reason:<32s} → {model.split('/')[-1]:<20s} "
            f"[{_stats['local']}L {_stats['flash']}F {_stats['sonnet']}S {_stats['opus']}O] "
            f"${total_cost:.3f} spent  ${saved:.3f} saved",
            flush=True,
        )

        self._forward(body, tier, t0)

    def _forward(self, body, tier, t0):
        encoded = json.dumps(body).encode()
        is_stream = body.get("stream", False)

        if tier == "local":
            target = LOCAL_BASE + "/v1/messages"
            headers = {"Content-Type": "application/json"}
        else:
            target = OPENROUTER_BASE + "/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
                "HTTP-Referer": "https://claudecode.local",
                "X-Title": "Claude Code Cloud Router",
            }
        req = urllib.request.Request(target, data=encoded, headers=headers, method="POST")

        try:
            resp = urllib.request.urlopen(req, timeout=300)
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

            with _lock:
                lat = _latency[tier]
                lat.append(time.time() - t0)
                if len(lat) > 20:
                    lat.pop(0)

        except urllib.error.HTTPError as e:
            # If cheap model fails, escalate to next tier
            fallback = {"local": "flash", "flash": "sonnet"}.get(tier)
            if fallback and e.code >= 500:
                print(f"[!] {tier} failed ({e.code}), escalating to {fallback}", flush=True)
                body["model"] = MODELS[fallback]
                with _lock:
                    _stats[tier] -= 1
                    _stats[fallback] += 1
                self._forward(body, fallback, t0)
                return
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

    def _serve_stats(self):
        with _lock:
            total = _stats["requests"] or 1
            uptime_h = (time.time() - _stats["start_time"]) / 3600
            total_cost = sum(_cost.values())
            all_opus_cost = sum(_tokens.values()) / 1_000_000 * COST_PER_M["opus"]
            saved = all_opus_cost - total_cost

            def avg_lat(tier):
                vals = _latency[tier]
                return round(sum(vals) / len(vals), 2) if vals else None

            data = {
                "uptime_hours": round(uptime_h, 1),
                "total_requests": _stats["requests"],
                "routing": {
                    t: {"count": _stats[t], "pct": round(_stats[t] / total * 100, 1)}
                    for t in MODELS
                },
                "tokens": {k: v for k, v in _tokens.items()},
                "cost": {
                    "actual": round(total_cost, 4),
                    "if_all_opus": round(all_opus_cost, 4),
                    "saved": round(saved, 4),
                    "saving_pct": round(saved / all_opus_cost * 100, 1) if all_opus_cost > 0 else 0,
                },
                "avg_latency_s": {t: avg_lat(t) for t in MODELS},
                "models": MODELS,
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
    print(f"╔══════════════════════════════════════════╗", flush=True)
    print(f"║       Cloud Smart Router v1.0            ║", flush=True)
    print(f"╠══════════════════════════════════════════╣", flush=True)
    print(f"║  Tool calls:  gemma4:e4b      FREE      ║", flush=True)
    print(f"║  Fast:        Gemini 2.5 Flash $1.40/M   ║", flush=True)
    print(f"║  Coding:      Claude Sonnet    $6.00/M   ║", flush=True)
    print(f"║  Complex:     Claude Opus     $45.00/M   ║", flush=True)
    print(f"║  Stats:       /stats                     ║", flush=True)
    print(f"╚══════════════════════════════════════════╝", flush=True)

    server = ThreadedServer(("127.0.0.1", PORT), Handler)
    server.serve_forever()
