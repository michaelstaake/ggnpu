# Web Chat (Prototype)

A minimal local web UI for chatting with a GGUF model via ggnpu. No extra Python
packages — only the standard library.

**Prototype limitations:**

- The model is **reloaded on every message** (slow; often tens of seconds).
- Chat history is re-sent as a plain text prompt each time (no KV-cache reuse).
- One inference at a time (concurrent requests get HTTP 503).
- Must run on the NPU host (`/dev/accel/accel0`).

## Prerequisites

1. ggnpu built with NPU backend: `build-npu/ggnpu`
2. A local `.gguf` model under `models/` (or any path)
3. NPU kernels in `~/.cache/ggnpu/xclbin/` (see [host setup](host-setup-guide.md))

## Start the server

```bash
python3 scripts/chat_server.py --model models/llama-3.2-1b-q4_k_m.gguf
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to `.gguf` file |
| `--ggnpu` | `./build-npu/ggnpu` | Path to ggnpu binary |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Port |
| `--max-tokens` | `128` | Default max new tokens |
| `--temp` | `0.7` | Default temperature |
| `--ctx-size` | `0` | Context override (`0` = model default) |
| `--timeout` | `300` | Subprocess timeout (seconds) |

Open **http://127.0.0.1:8080** in a browser.

## How it works

```
Browser  →  POST /api/chat  →  chat_server.py  →  ggnpu --quiet  →  NPU
```

The server spawns `ggnpu` with `--quiet` so generated text goes to stdout only
(status and load progress go to stderr). The UI keeps conversation history in the
browser and formats each request as:

```text
User: …
Assistant: …
User: <new message>
Assistant:
```

## API

`POST /api/chat`

```json
{
  "message": "Hello",
  "history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"}
  ],
  "max_tokens": 128,
  "temp": 0.7
}
```

Response (200):

```json
{
  "response": "…generated text…",
  "elapsed_s": 42.5
}
```

## Future improvements

- Persistent model daemon (`ggnpu serve`) to avoid reload per message
- SSE token streaming to the browser
- Model-specific chat templates from GGUF metadata
- Multi-turn KV cache without re-encoding full history
