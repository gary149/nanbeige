import { pipeline } from "@huggingface/transformers";

console.log("Loading model (fp32)...");
console.log("Note: this is ~12GB download, will take a while...");
const t0 = Date.now();

const generator = await pipeline(
  "text-generation",
  "onnx-community/Nanbeige4.1-3B-ONNX",
  { dtype: "fp32" },
);

console.log(`Model loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

const messages = [
  { role: "user", content: "What is 2+2? Reply in one sentence." },
];

console.log("Generating...");
const t1 = Date.now();

const output = await generator(messages, { max_new_tokens: 64 });
const reply = output[0].generated_text.at(-1).content;

console.log(`Generated in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
console.log("\n--- Response ---");
console.log(reply);
