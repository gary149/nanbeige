# Nanbeige 3B — In-Browser & Server Inference

## What this is

Local inference demo for the Nanbeige 4.1 3B model using ONNX (q4) via Transformers.js. Two interfaces:

- **Browser demo** (`index.html`): Loads the model client-side via WebGPU/WASM, streams tokens with thinking block support
- **Server** (`server.js`): OpenAI-compatible API at `http://localhost:8741/v1`

## Running

```
npm install
node server.js
```

Opens at `http://localhost:8741/`. The server also serves static files (`.html`, `.png`, `.svg`, `.js`, `.css`).

## Architecture

- **Single-file UI**: `index.html` contains all HTML/CSS/JS (no build step). Imports Transformers.js from CDN.
- **server.js**: Node.js HTTP server. Loads model via `@huggingface/transformers` + `onnxruntime-node`. Serves both the API and static files.
- **Model**: `onnx-community/Nanbeige4.1-3B-ONNX` (q4 quantization, ~1.7 GB)

## Key files

| File | Purpose |
|------|---------|
| `index.html` | Browser demo — all UI, styles, and client-side inference in one file |
| `server.js` | OpenAI-compatible server (`/v1/chat/completions`, `/v1/models`) + static file server |
| `nanbeige.svg` | Logo wordmark |
| `onboarding-bg.png` | Loading screen background image |

## Conventions

- Monochrome UI (black/white/grey only, no color accents)
- The server only uses Node built-ins + `@huggingface/transformers` + `onnxruntime-node`
- Browser demo uses no bundler — raw ES modules from CDN
- Recommended inference params: temperature 0.6, top-p 0.95, repetition_penalty 1.0
- The model outputs `<think>...</think>` blocks before answering — the `OutputParser` class handles streaming these

## Port

Server runs on port **8741**.
