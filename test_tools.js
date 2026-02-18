import { AutoModelForCausalLM, AutoTokenizer } from "@huggingface/transformers";

const model_id = "onnx-community/Nanbeige4.1-3B-ONNX";

console.log("Loading model...");
const tokenizer = await AutoTokenizer.from_pretrained(model_id);
const model = await AutoModelForCausalLM.from_pretrained(model_id, { dtype: "q4" });

// Define tools
const tools = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather for a given city",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string", description: "The city name" },
        },
        required: ["city"],
      },
    },
  },
];

// Helper: generate and return only the new tokens
async function generate(messages, maxTokens = 1024) {
  const text = tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
    tools,
  });

  const inputs = tokenizer(text);
  const promptLen = inputs.input_ids.dims[1];
  console.log(`  Prompt length: ${promptLen} tokens`);

  const output_ids = await model.generate({
    ...inputs,
    max_new_tokens: maxTokens,
    do_sample: false,
  });

  const totalLen = output_ids[0].dims[0];
  console.log(`  Generated ${totalLen - promptLen} new tokens`);

  // Build array of just the new token ids
  const newTokenIds = [];
  for (let i = promptLen; i < totalLen; i++) {
    newTokenIds.push(Number(output_ids[0].data[i]));
  }

  return tokenizer.decode(newTokenIds, { skip_special_tokens: false });
}

// === Step 1: Ask the model, expect a tool call ===
console.log("\n=== Step 1: User asks about weather ===");
const messages = [
  { role: "user", content: "What's the weather in Paris?" },
];

const t0 = Date.now();
const response1 = await generate(messages);
console.log(`  Time: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log("\n--- Model output ---");
console.log(response1);

// Parse tool call
const toolCallMatch = response1.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/);
if (!toolCallMatch) {
  console.log("\nNo <tool_call> found in output.");
  process.exit(0);
}

const call = JSON.parse(toolCallMatch[1]);
console.log("\n--- Parsed tool call ---");
console.log("Function:", call.name);
console.log("Arguments:", JSON.stringify(call.arguments, null, 2));

// === Step 2: Feed tool response back ===
console.log("\n\n=== Step 2: Feeding tool response ===");

const assistantContent = response1.replace(/<\|im_end\|>[\s\S]*$/, "").trim();

const messagesWithTool = [
  ...messages,
  {
    role: "assistant",
    tool_calls: [{ function: { name: call.name, arguments: JSON.stringify(call.arguments) } }],
  },
  {
    role: "tool",
    content: JSON.stringify({
      city: "Paris",
      temperature: 18,
      condition: "partly cloudy",
      humidity: 65,
    }),
  },
];

const t1 = Date.now();
const response2 = await generate(messagesWithTool);
console.log(`  Time: ${((Date.now() - t1) / 1000).toFixed(1)}s`);

console.log("\n--- Final answer ---");
const answer = response2
  .replace(/<think>[\s\S]*?<\/think>\s*/g, "")
  .replace(/<\|im_end\|>/g, "")
  .trim();
console.log(answer);
