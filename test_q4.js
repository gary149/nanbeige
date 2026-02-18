import { pipeline } from "@huggingface/transformers";

console.log("Loading model (q4)...");
const t0 = Date.now();

const generator = await pipeline(
  "text-generation",
  "onnx-community/Nanbeige4.1-3B-ONNX",
  { dtype: "q4" },
);

console.log(`Model loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

const messages = [
  { role: "user", content: "What is the capital of France? Reply in one sentence." },
];

console.log("Generating...");
const t1 = Date.now();

const output = await generator(messages, { max_new_tokens: 512 });
const elapsed = ((Date.now() - t1) / 1000).toFixed(1);

const reply = output[0].generated_text.at(-1).content;

console.log(`\nGenerated in ${elapsed}s`);
console.log("\n--- Full response (includes <think> reasoning) ---");
console.log(reply);

// Extract just the answer after </think>
const answerMatch = reply.match(/<\/think>\s*([\s\S]*)/);
if (answerMatch) {
  console.log("\n--- Answer only ---");
  console.log(answerMatch[1].trim());
}
