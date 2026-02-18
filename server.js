import http from "node:http";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
} from "@huggingface/transformers";

const __dirname = dirname(fileURLToPath(import.meta.url));

const MODEL_ID = "onnx-community/Nanbeige4.1-3B-ONNX";
const PORT = 8741;

console.log("Loading tokenizer...");
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);

console.log("Loading model (q4)...");
const t0 = Date.now();
const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
  dtype: "q4",
});
console.log(`Model loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

// ---------------------------------------------------------------------------
// OutputParser: state machine that consumes raw model text token-by-token
// and emits structured events (skipping <think> blocks, detecting <tool_call>)
// ---------------------------------------------------------------------------
class OutputParser {
  constructor({ onTextDelta, onToolCall, onFinish }) {
    this.onTextDelta = onTextDelta;
    this.onToolCall = onToolCall;
    this.onFinish = onFinish;
    this.state = "init"; // init | thinking | content | tool_call
    this.buf = "";
    this.toolBuf = "";
    this.hasToolCalls = false;
  }

  feed(text) {
    this.buf += text;
    this._drain();
  }

  end() {
    // Flush any remaining buffered content
    if (this.state === "content" && this.buf.length > 0) {
      this.onTextDelta(this.buf);
      this.buf = "";
    }
    this.onFinish(this.hasToolCalls ? "tool_calls" : "stop");
  }

  _drain() {
    let changed = true;
    while (changed) {
      changed = false;

      if (this.state === "init") {
        // Waiting to see if output starts with <think>
        if (this.buf.length >= 7) {
          if (this.buf.startsWith("<think>")) {
            this.state = "thinking";
            this.buf = this.buf.slice(7);
            changed = true;
          } else {
            // No think block, go straight to content
            this.state = "content";
            changed = true;
          }
        } else if (this.buf.length > 0 && !"<think>".startsWith(this.buf)) {
          // Buffer can't possibly become <think>, go to content
          this.state = "content";
          changed = true;
        }
        // else: buf is a prefix of "<think>", wait for more
      }

      if (this.state === "thinking") {
        const idx = this.buf.indexOf("</think>");
        if (idx >= 0) {
          // Done thinking — discard everything up to and including </think>
          this.buf = this.buf.slice(idx + 8);
          this.state = "content";
          changed = true;
        } else {
          // Keep only last 7 chars (in case </think> straddles chunks)
          if (this.buf.length > 7) {
            this.buf = this.buf.slice(-7);
          }
        }
      }

      if (this.state === "content") {
        // Look for <tool_call> tag
        const tcIdx = this.buf.indexOf("<tool_call>");
        if (tcIdx >= 0) {
          // Emit text before the tag
          if (tcIdx > 0) {
            this.onTextDelta(this.buf.slice(0, tcIdx));
          }
          this.buf = this.buf.slice(tcIdx + 11);
          this.state = "tool_call";
          this.toolBuf = "";
          changed = true;
        } else {
          // Emit text but hold back chars that could be start of <tool_call>
          const safe = this._safeEmitLength(this.buf, "<tool_call>");
          if (safe > 0) {
            this.onTextDelta(this.buf.slice(0, safe));
            this.buf = this.buf.slice(safe);
            changed = true;
          }
        }
      }

      if (this.state === "tool_call") {
        const endIdx = this.buf.indexOf("</tool_call>");
        if (endIdx >= 0) {
          this.toolBuf += this.buf.slice(0, endIdx);
          this.buf = this.buf.slice(endIdx + 12);
          // Parse and emit tool call
          try {
            const parsed = JSON.parse(this.toolBuf.trim());
            this.hasToolCalls = true;
            this.onToolCall({
              id: `call_${Math.random().toString(36).slice(2, 11)}`,
              type: "function",
              function: {
                name: parsed.name,
                arguments: JSON.stringify(parsed.arguments),
              },
            });
          } catch {
            // Couldn't parse — emit as plain text
            this.onTextDelta(`<tool_call>${this.toolBuf}</tool_call>`);
          }
          this.state = "content";
          changed = true;
        } else {
          // Accumulate into tool buffer
          this.toolBuf += this.buf;
          this.buf = "";
        }
      }
    }
  }

  /** Return how many leading bytes of `str` can't possibly be a prefix of `tag`. */
  _safeEmitLength(str, tag) {
    // We can emit everything except the last (tag.length - 1) chars,
    // which could be a partial match for the tag.
    for (let hold = Math.min(tag.length - 1, str.length); hold >= 0; hold--) {
      const tail = str.slice(str.length - hold);
      if (tag.startsWith(tail)) {
        return str.length - hold;
      }
    }
    return str.length;
  }
}

// ---------------------------------------------------------------------------
// Prepare input: convert OpenAI messages+tools → tokenized input
// ---------------------------------------------------------------------------
function prepareInput(body) {
  const messages = body.messages || [];
  const tools = body.tools || undefined;
  const text = tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
    tools: tools?.length ? tools : undefined,
  });
  return tokenizer(text);
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------
function makeChunk(id, delta, finishReason, usage) {
  const chunk = {
    id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: MODEL_ID,
    choices: [{ index: 0, delta, finish_reason: finishReason ?? null }],
  };
  if (usage) chunk.usage = usage;
  return chunk;
}

function sendSSE(res, data) {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — streaming
// ---------------------------------------------------------------------------
async function handleStream(body, res) {
  const maxTokens = body.max_tokens || body.max_completion_tokens || 4096;
  const temperature = body.temperature ?? 0;
  const topP = body.top_p ?? 1.0;

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  const inputs = prepareInput(body);
  const promptLen = inputs.input_ids.dims[1];
  const id = `chatcmpl-${Date.now()}`;
  let sentRole = false;
  let toolCallIndex = 0;
  let aborted = false;

  res.on("close", () => { aborted = true; });

  const parser = new OutputParser({
    onTextDelta: (text) => {
      if (aborted) return;
      const delta = { content: text };
      if (!sentRole) { delta.role = "assistant"; sentRole = true; }
      sendSSE(res, makeChunk(id, delta, null));
    },
    onToolCall: (tc) => {
      if (aborted) return;
      // Chunk 1: id + name
      const startDelta = {
        tool_calls: [{
          index: toolCallIndex,
          id: tc.id,
          type: "function",
          function: { name: tc.function.name, arguments: "" },
        }],
      };
      if (!sentRole) { startDelta.role = "assistant"; sentRole = true; }
      sendSSE(res, makeChunk(id, startDelta, null));

      // Chunk 2: arguments
      sendSSE(res, makeChunk(id, {
        tool_calls: [{
          index: toolCallIndex,
          function: { arguments: tc.function.arguments },
        }],
      }, null));

      toolCallIndex++;
    },
    onFinish: (reason) => {
      // Sent after generation completes (in the finally block below)
    },
  });

  // Custom streamer: feeds decoded text into the parser token-by-token
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      if (!aborted) parser.feed(text);
    },
  });

  try {
    const outputIds = await model.generate({
      ...inputs,
      max_new_tokens: maxTokens,
      do_sample: temperature > 0,
      temperature: temperature > 0 ? temperature : undefined,
      top_p: topP,
      streamer,
    });

    if (aborted) return;

    // Flush remaining buffered content
    parser.end();

    const completionTokens = outputIds[0].dims[0] - promptLen;
    const finishReason = parser.hasToolCalls ? "tool_calls" : "stop";

    // Final chunk with finish_reason + usage
    sendSSE(res, makeChunk(id, {}, finishReason, {
      prompt_tokens: promptLen,
      completion_tokens: completionTokens,
      total_tokens: promptLen + completionTokens,
    }));

    res.write("data: [DONE]\n\n");
  } catch (err) {
    if (!aborted) {
      console.error("Generation error:", err.message);
      res.write(`data: ${JSON.stringify({ error: { message: err.message, type: "server_error" } })}\n\n`);
    }
  } finally {
    res.end();
  }
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions — non-streaming
// ---------------------------------------------------------------------------
async function handleNonStream(body, res) {
  const maxTokens = body.max_tokens || body.max_completion_tokens || 4096;
  const temperature = body.temperature ?? 0;
  const topP = body.top_p ?? 1.0;

  const inputs = prepareInput(body);
  const promptLen = inputs.input_ids.dims[1];
  const id = `chatcmpl-${Date.now()}`;

  const outputIds = await model.generate({
    ...inputs,
    max_new_tokens: maxTokens,
    do_sample: temperature > 0,
    temperature: temperature > 0 ? temperature : undefined,
    top_p: topP,
  });

  const totalLen = outputIds[0].dims[0];
  const newTokenIds = [];
  for (let i = promptLen; i < totalLen; i++) {
    newTokenIds.push(Number(outputIds[0].data[i]));
  }
  const raw = tokenizer.decode(newTokenIds, { skip_special_tokens: true });

  // Parse with the same state machine
  let textParts = [];
  let toolCalls = [];
  const parser = new OutputParser({
    onTextDelta: (t) => textParts.push(t),
    onToolCall: (tc) => toolCalls.push(tc),
    onFinish: () => {},
  });
  parser.feed(raw);
  parser.end();

  const content = textParts.join("").trim() || null;
  const finishReason = toolCalls.length > 0 ? "tool_calls" : "stop";
  const message = { role: "assistant" };
  if (content) message.content = content;
  if (toolCalls.length > 0) message.tool_calls = toolCalls;

  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify({
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: MODEL_ID,
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage: {
      prompt_tokens: promptLen,
      completion_tokens: totalLen - promptLen,
      total_tokens: totalLen,
    },
  }));
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------
const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }

  const url = new URL(req.url, `http://localhost:${PORT}`);

  // Serve index.html at /
  if ((url.pathname === "/" || url.pathname === "/index.html") && req.method === "GET") {
    try {
      const html = readFileSync(join(__dirname, "index.html"), "utf-8");
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(html);
    } catch {
      res.writeHead(404);
      res.end("index.html not found");
    }
    return;
  }

  if (url.pathname === "/v1/models" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      object: "list",
      data: [{
        id: MODEL_ID,
        object: "model",
        created: Math.floor(Date.now() / 1000),
        owned_by: "onnx-community",
      }],
    }));
    return;
  }

  if (url.pathname === "/v1/chat/completions" && req.method === "POST") {
    try {
      const body = await parseBody(req);
      const nTools = body.tools?.length || 0;
      console.log(`[${new Date().toISOString()}] messages=${body.messages?.length} tools=${nTools} stream=${body.stream ?? false}`);

      if (body.stream) {
        await handleStream(body, res);
      } else {
        await handleNonStream(body, res);
      }
    } catch (err) {
      console.error("Request error:", err);
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" });
      }
      res.end(JSON.stringify({ error: { message: err.message, type: "server_error" } }));
    }
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: { message: "Not found" } }));
});

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let d = "";
    req.on("data", (c) => (d += c));
    req.on("end", () => { try { resolve(JSON.parse(d)); } catch (e) { reject(e); } });
    req.on("error", reject);
  });
}

server.listen(PORT, () => {
  console.log(`\nNanbeige OpenAI-compatible API at http://localhost:${PORT}/v1`);
  console.log(`  POST /v1/chat/completions  (streaming + non-streaming)`);
  console.log(`  GET  /v1/models`);
});
