"""Tiny reverse proxy: listens on local port, forwards to OpenRouter API."""
import http.server
import json
import os
import sys
import urllib.request
import urllib.error

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5005
MODEL = sys.argv[2] if len(sys.argv) > 2 else "deepseek/deepseek-r1"
API_KEY = sys.argv[3] if len(sys.argv) > 3 else ""

OPENROUTER_BASE = "https://openrouter.ai/api"


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._respond(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0}]})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        # Parse and override model
        try:
            data = json.loads(body)
            data["model"] = MODEL
            body = json.dumps(data).encode()
        except Exception:
            pass

        # Forward to OpenRouter
        target = OPENROUTER_BASE + self.path
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": "https://claudecode.local",
            "X-Title": "Claude Code Local",
        }

        # Check if streaming
        is_stream = b'"stream":true' in body or b'"stream": true' in body

        req = urllib.request.Request(target, data=body, headers=headers, method="POST")
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

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[proxy] {args[0]}", flush=True)


if __name__ == "__main__":
    print(f"OpenRouter proxy: port={PORT} model={MODEL}", flush=True)
    server = http.server.HTTPServer(("127.0.0.1", PORT), ProxyHandler)
    server.serve_forever()
