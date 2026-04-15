# Getting Started: Gemma 4 with Ollama

Run Claude Code against Google's Gemma 4 locally — free, private, no API key needed.
Works on any Mac with 16GB+ RAM.

## What you get

- **gemma4:e4b** (8B MoE, 9.6 GB) — full tool calling, vision, fast. Recommended.
- **gemma4:e2b** (2B MoE, 7.2 GB) — lighter, but too small for reliable tool use in Claude Code.

Both run via [Ollama](https://ollama.com). The app starts a proxy that translates Claude Code's
Anthropic API format into Ollama's OpenAI-compatible format, including full tool call support.

---

## 1. Install Ollama

```bash
brew install ollama
brew services start ollama
```

Verify it's running:
```bash
ollama list
```

## 2. Pull Gemma 4

```bash
# Recommended — 8B MoE, good tool calling
ollama pull gemma4:e4b

# Lighter alternative (not recommended for Claude Code)
ollama pull gemma4:e2b
```

Pulling takes a few minutes. Check progress with `ollama list`.

## 3. Build and install LLM Router

```bash
git clone https://github.com/Casperhr/claude-code-llm-router.git ~/llm-router
cd ~/llm-router/app

swift build -c release

mkdir -p "/Applications/LLM Router.app/Contents/MacOS"
cp .build/release/LocalLLM "/Applications/LLM Router.app/Contents/MacOS/LocalLLM"

# Create a minimal Info.plist so macOS treats it as a proper app
cat > "/Applications/LLM Router.app/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>LocalLLM</string>
    <key>CFBundleIdentifier</key>
    <string>com.llm-router.app</string>
    <key>CFBundleName</key>
    <string>LLM Router</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

open "/Applications/LLM Router.app"
```

The sparkle icon appears in your menu bar.

## 4. Select Gemma 4

Click the menu bar icon → **Ollama (local)** → **gemma4:e4b**

The icon turns cyan when the proxy is ready (~2 seconds).

## 5. Set up the `claudel` alias

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
claudel() {
  local model
  model=$(curl -s --max-time 1 http://localhost:5005/model 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['model'])" 2>/dev/null)
  model=${model:-"local"}
  ANTHROPIC_BASE_URL=http://localhost:5005 ANTHROPIC_MODEL="$model" \
    claude --dangerously-skip-permissions "$@"
}
```

Then reload:
```bash
source ~/.zshrc
```

## 6. Use it

```bash
claudel
```

The header will show `gemma4:e4b`. Claude Code will use local Gemma 4 for everything.

---

## How it works

```
claudel
  └─ ANTHROPIC_BASE_URL=http://localhost:5005
       └─ app/ollama-proxy.py (port 5005)
            └─ Ollama /v1/chat/completions (port 11434)
                 └─ gemma4:e4b
```

The proxy (`app/ollama-proxy.py`) handles:
- Translating Anthropic `messages` format → OpenAI format
- Converting `tool_use` / `tool_result` content blocks ↔ OpenAI function calling
- Streaming SSE in Anthropic format
- Stripping unsupported Claude Code beta headers (`context_management`, `output_config`)

---

## Tips

**gemma4:e4b vs e2b**

Use `e4b`. The 2B model doesn't reliably follow Claude Code's system prompt for tool use.

**Slow first response?**

Ollama loads the model into memory on the first request. Subsequent requests are fast.
You can pre-warm it:
```bash
curl -s http://localhost:11434/api/generate -d '{"model":"gemma4:e4b","prompt":"hi","stream":false}' > /dev/null
```

**Switch models without restarting Claude Code**

Click a different model in the menu bar. The proxy on port 5005 switches immediately.
Claude Code picks it up on the next request.

**Monitor what's happening**

```bash
tail -f /tmp/ollama-proxy.log
```

---

## Combine with Cloud Router for best results

Gemma 4 handles tool call continuations well but struggles with complex planning tasks.
For the best cost/quality tradeoff, use **Cloud Router** instead — it routes tool calls
to local Gemma 4 (free) and only pays for Gemini/Sonnet/Opus when the task needs it.

See [Cloud Router docs](../README.md#cloud-router) for setup.
