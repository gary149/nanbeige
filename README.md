# nanbeige

OpenAI-compatible API server that runs [Nanbeige4.1-3B](https://huggingface.co/onnx-community/Nanbeige4.1-3B-ONNX) locally via [Transformers.js](https://huggingface.co/docs/transformers.js) and ONNX Runtime.

The model is a 3B-parameter reasoning LLM with native tool calling support. The server translates Nanbeige's `<think>` / `<tool_call>` XML format into the standard OpenAI chat completions wire format, so any OpenAI-compatible client can use it as a drop-in local backend.

<img width="1553" height="1262" alt="image" src="https://github.com/user-attachments/assets/2ce8bbd9-efa9-4e43-a576-01121b404589" />


## Setup

```bash
npm install
```

**Critical:** `@huggingface/transformers` bundles `onnxruntime-node@1.21.0`, which doesn't support the `GatherBlockQuantized` operator used by this model's quantized variants. The fix:

```bash
# Install a version that supports it
npm install onnxruntime-node@1.25.0-dev.20260213-bd8f781f2c

# Remove the nested outdated copy so the top-level one is used
rm -rf node_modules/@huggingface/transformers/node_modules/onnxruntime-node
```

This is already reflected in `package.json`. If `npm install` re-creates the nested copy, re-run the `rm -rf` command above.

## Usage

```bash
node server.js
```

On first run, the model weights (~1.7 GB, q4 quantization) are downloaded from Hugging Face and cached locally. Subsequent starts load from cache in ~1s.

```
Nanbeige OpenAI-compatible API at http://localhost:8741/v1
  POST /v1/chat/completions  (streaming + non-streaming)
  GET  /v1/models
```

## Browser demo

The server also serves a browser-based chat UI at `http://localhost:8741/`. It loads the model directly in the browser via WebGPU/WASM (no server-side inference). The `<think>` reasoning blocks stream live into a collapsible section so you can watch the model's chain-of-thought in real time.

## API

Implements a subset of the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).

### `GET /v1/models`

Lists the loaded model.

### `POST /v1/chat/completions`

Standard chat completions endpoint. Supported request fields:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `messages` | array | required | `system`, `user`, `assistant`, `tool` roles |
| `stream` | boolean | `false` | SSE streaming with `data:` lines and `[DONE]` |
| `tools` | array | - | OpenAI function-calling tool definitions |
| `max_tokens` | number | `4096` | Also accepts `max_completion_tokens` |
| `temperature` | number | `0` | `0` = greedy, `> 0` = sampling |
| `top_p` | number | `1.0` | Nucleus sampling |

**Streaming** returns `text/event-stream` with `chat.completion.chunk` objects. Each chunk has a `delta` with incremental `content` or `tool_calls`. The final chunk includes `finish_reason` and `usage`.

**Non-streaming** returns a single `chat.completion` JSON object.

### Tool calling

The server accepts OpenAI-format `tools` in the request. Nanbeige uses XML-based tool calling internally (`<tool_call>{"name": ..., "arguments": ...}</tool_call>`). The server translates this to/from the OpenAI `tool_calls` format transparently:

- Request: OpenAI `tools` array -> Nanbeige `<tools>` XML in the chat template
- Response: Nanbeige `<tool_call>` XML -> OpenAI `delta.tool_calls` chunks
- `finish_reason` is `"tool_calls"` when the model invokes a tool, `"stop"` otherwise

Tool responses are passed back as `{"role": "tool", "content": "...", "tool_call_id": "..."}` messages, matching the OpenAI format.

## Architecture

```
                    OpenAI-compatible HTTP
curl / any client ──────────────────────────> server.js (port 8741)
                                                  │
                                                  ├─ prepareInput()
                                                  │    messages + tools -> chat template -> tokenize
                                                  │
                                                  ├─ model.generate() with TextStreamer
                                                  │    token-by-token callbacks
                                                  │
                                                  ├─ OutputParser (state machine)
                                                  │    <think>...  -> silently consumed
                                                  │    </think>    -> start emitting content
                                                  │    <tool_call> -> buffer, parse, emit as tool_calls
                                                  │    plain text  -> emit as content deltas
                                                  │
                                                  └─ SSE / JSON response
```

### OutputParser state machine

The model generates raw text that may contain `<think>` reasoning blocks and `<tool_call>` XML. The `OutputParser` class processes this token-by-token through four states:

| State | Behavior |
|-------|----------|
| `init` | Waiting for first tokens. If `<think>` detected, enter `thinking`. Otherwise enter `content`. |
| `thinking` | Discard all tokens until `</think>` is found. Handles tag spanning chunk boundaries. |
| `content` | Emit text as `content` deltas. Watch for `<tool_call>` tag, holding back partial matches. |
| `tool_call` | Buffer tokens until `</tool_call>`. Parse JSON, emit as OpenAI `tool_calls` with `id`, `name`, `arguments`. |

This means `<think>` blocks (which can be 500+ tokens of internal reasoning) are never sent to the client. The client only sees the final answer and/or tool calls.

## Model details

| Property | Value |
|----------|-------|
| Base model | [Nanbeige/Nanbeige4.1-3B](https://huggingface.co/Nanbeige/Nanbeige4.1-3B) |
| ONNX conversion | [onnx-community/Nanbeige4.1-3B-ONNX](https://huggingface.co/onnx-community/Nanbeige4.1-3B-ONNX) |
| Architecture | LlamaForCausalLM, 32 layers, hidden_size 2560 |
| Parameters | 3B |
| Quantization used | q4 (~1.7 GB download) |
| Context window | 262,144 tokens (config), practical limit much lower on CPU |
| Chat template | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) |
| Reasoning | Chain-of-thought via `<think>` blocks |
| Tool calling | XML-based `<tool_call>` / `<tool_response>` |
| License | Apache 2.0 |

### Available quantizations in the ONNX repo

| dtype | Files | Size | Notes |
|-------|-------|------|-------|
| `fp32` | `model.onnx` (8 shards) | ~12 GB | Works with any onnxruntime-node |
| `fp16` | `model_fp16.onnx` (4 shards) | ~6 GB | Needs WebGPU (fp16 KV cache unsupported on CPU) |
| `q8` | `model_quantized.onnx` (3 shards) | ~3.5 GB | Needs onnxruntime-node >= 1.25 |
| `q4` | `model_q4.onnx` (2 shards) | ~1.7 GB | Needs onnxruntime-node >= 1.25 (used by default) |
| `q4f16` | `model_q4f16.onnx` (2 shards) | ~1.7 GB | q4 weights + fp16 KV cache, needs WebGPU |

## Performance

Measured on Apple Silicon (M-series), Node.js, CPU inference:

| Metric | Value |
|--------|-------|
| Model load (cached) | ~1s |
| Model load (first download) | Depends on connection (~1.7 GB) |
| Inference speed | ~7 tok/s (q4, CPU) |
| Reasoning overhead | 300-700 tokens of `<think>` before answer (not sent to client) |

## Test scripts

| File | Purpose |
|------|---------|
| `test.js` | Basic pipeline text generation |
| `test_q4.js` | Pipeline-based generation with `<think>` stripping |
| `test_tools.js` | End-to-end tool calling: tool call -> tool response -> final answer |

## Connecting to OpenAI-compatible clients

Any client that supports a custom base URL works. Examples:

**curl:**
```bash
curl http://localhost:8741/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

**OpenAI Node SDK:**
```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:8741/v1", apiKey: "unused" });
const res = await client.chat.completions.create({
  model: "onnx-community/Nanbeige4.1-3B-ONNX",
  messages: [{ role: "user", content: "Hello" }],
});
```

**pi-mono (openai-completions provider):**
```typescript
const nanbeige: Model<"openai-completions"> = {
  id: "onnx-community/Nanbeige4.1-3B-ONNX",
  name: "Nanbeige 3B (local)",
  api: "openai-completions",
  provider: "nanbeige-local",
  baseUrl: "http://localhost:8741/v1",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 32768,
  maxTokens: 4096,
};
```

