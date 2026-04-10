# Claude Code LLM Router

A macOS menu bar app that lets you run local and cloud LLMs as a Claude Code backend. Switch models with one click, route requests to the cheapest model that can handle them, and track cost savings in real time.

## What it does

```
Claude Code
    |
    | ANTHROPIC_BASE_URL=http://localhost:5005
    v
Port 5005 -- managed by this app
    |
    |-- Local GGUF models     --> llama.cpp / llama-server
    |-- Local MLX models      --> server/serve_mlx.py
    |-- Smart Router          --> local model + Sonnet + Opus (hybrid)
    |-- Cloud Router          --> Qwen + Gemini Flash + Sonnet + Opus (all cloud)
    '-- OpenRouter            --> any OpenRouter model (single model proxy)
```

Click a model in the menu bar. The app kills the current server, starts the new one on port 5005. Claude Code picks it up on the next request. No restart needed.

## Features

- **One-click model switching** -- TurboQuant GGUF, upstream GGUF, MLX, OpenRouter, Smart Router
- **Smart Router** -- routes 70% of Claude Code requests (tool calls) to free local model, only uses Sonnet/Opus when needed. ~85% cost reduction.
- **Cloud Router** -- same routing logic, all cloud (Qwen + Gemini Flash + Sonnet + Opus). No local GPU needed.
- **Live menu bar stats** -- tok/s, memory, CPU, GPU for local models; request count, cost, savings % for routers
- **Analytics panel** -- TTFT, generation speed, token counts
- **Auto-detects backend** -- llama.cpp vs MLX vs proxy
- **Status icon** -- grey (stopped), cyan (ready), orange (loading), red (crashed)

## Requirements

- macOS 14+ (Apple Silicon)
- Swift 6.0+ (Xcode 16+)
- One or more of:
  - [llama.cpp](https://github.com/ggml-org/llama.cpp) -- for GGUF models
  - [MLX](https://github.com/ml-explore/mlx) + [mlx-lm](https://github.com/ml-explore/mlx-examples) -- for MLX models
  - [OpenRouter API key](https://openrouter.ai/) -- for cloud models and Smart/Cloud Router

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Casperhr/claude-code-llm-router.git ~/llm-router
cd ~/llm-router

# 2. Build the menu bar app
cd app && swift build -c release

# 3. Install
mkdir -p "/Applications/LLM Router.app/Contents/MacOS"
cp .build/release/LocalLLM "/Applications/LLM Router.app/Contents/MacOS/LocalLLM"

# 4. Download a model
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
  gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --local-dir ~/llm-router/gguf

# 5. Build llama.cpp (if using GGUF models)
git clone https://github.com/ggml-org/llama.cpp.git ~/llama-cpp
cd ~/llama-cpp && cmake -B build -DGGML_METAL=ON && cmake --build build --target llama-server -j

# 6. Launch the app
open "/Applications/LLM Router.app"

# 7. Configure Claude Code
export ANTHROPIC_BASE_URL=http://localhost:5005
claude  # or your claude alias
```

## Configuration

The app looks for files relative to `~/llm-router/` by default. Override with:

```bash
export LLM_ROUTER_DIR=/path/to/your/setup
```

### OpenRouter API Key

Create a `.env` file in the base directory:

```bash
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > ~/llm-router/.env
```

### Directory Structure

```
~/llm-router/
  app/                            <-- menu bar app (Swift)
    Sources/
      LocalLLMApp.swift           <-- UI, menu, icon
      ServerManager.swift         <-- server lifecycle, health, metrics
    Package.swift                 <-- Swift package manifest
    smart-proxy.py                <-- hybrid router (local + cloud)
    cloud-smart-proxy.py          <-- cloud-only router
    openrouter-proxy.py           <-- single-model OpenRouter proxy
    icon.png
  server/
    serve_mlx.py                  <-- MLX inference server (Anthropic Messages API)
  gguf/
    *.gguf                        <-- GGUF model files (not committed)
```

## Smart Router

The killer feature. Routes each Claude Code request to the cheapest backend that can handle it:

| Tier | When | Backend | Cost |
|------|------|---------|------|
| Local | Tool call continuations (~70% of requests) | Your local model on :5006 | Free |
| Sonnet | Coding, implementation, bug fixes | Claude Sonnet via OpenRouter | $6/M tokens |
| Opus | Planning, architecture, complex reasoning | Claude Opus via OpenRouter | $45/M tokens |

Classification is pure regex heuristics -- zero AI overhead, microsecond routing.

```bash
# Monitor routing in real time
curl localhost:5005/stats | python3 -m json.tool

# Logs show every routing decision
tail -f /tmp/smart-router.log
# [L] #42 tool_results=3              [30L 8S 4O] 71% local  $0.48 spent
```

## Cloud Router

Same routing logic, but uses cheap cloud models instead of local:

| Tier | Model | Cost/M tokens |
|------|-------|---------------|
| Tool calls | Qwen3 Coder 30B | $0.17 |
| Fast | Gemini 2.5 Flash | $1.40 |
| Coding | Claude Sonnet | $6.00 |
| Complex | Claude Opus | $45.00 |

No GPU required. Auto-escalates if a cheap model fails.

## Adding Models

### GGUF models
```bash
huggingface-cli download unsloth/MODEL-GGUF MODEL.Q4_K_M.gguf --local-dir ~/llm-router/gguf
```
Click "Refresh Models" in the menu bar.

### MLX models
```bash
huggingface-cli download mlx-community/MODEL-NAME --local-dir-use-symlinks False
# Move to the llm-router directory
mv ~/.cache/huggingface/hub/models--mlx-community--MODEL-NAME ~/llm-router/
```

## Recommended Models for Apple Silicon

| Model | Quant | Size | Speed (M4 Max 128GB) | Use case |
|-------|-------|------|---------------------|----------|
| Qwen3.5-35B-A3B | Q8_0 | 34GB | ~73 tok/s | Best all-around with TurboQuant |
| Gemma 4 26B-A4B | Q4_K_M | 17GB | ~130 tok/s | Fast coding, good tool use |
| Qwen3-Coder-30B-A3B | 4bit (MLX) | 16GB | ~40 tok/s | Dedicated coding model |

### Performance Tips

- **Use MoE models** -- they only activate 3-4B params per token. Dense 30B+ models are unusable (~3 tok/s).
- **Speed = bandwidth / active weights.** At 546 GB/s: 3B active = ~73 tok/s, 30B dense = ~3 tok/s.
- **Smaller quants are faster** -- Q4_K_M reads half the bytes per token vs Q8_0. At long context this matters a lot.
- **KV cache compression matters** -- TurboQuant turbo3 gives 4.6x compression, enabling 256K context at full speed. Without it, reduce context or use smaller KV quants.

## Build & Install

```bash
cd ~/llm-router/app
swift build -c release
cp .build/release/LocalLLM "/Applications/LLM Router.app/Contents/MacOS/LocalLLM"
```

The app auto-discovers models in the `gguf/` directory and HuggingFace cache format (`models--*` directories).

## License

MIT
