import { env } from "../env.js";
import { ChatOpenRouter } from "@langchain/openrouter";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

/**
 * ============================================================================
 * LANGCHAIN CHAT MODEL MASTER GUIDE
 * ============================================================================
 *
 * This file demonstrates best practices and all common patterns for using
 * Chat Models in LangChain:
 *
 * 1. Simple String Invocation
 * 2. Role-based Message Objects (OpenAI-style tuples)
 * 3. LangChain Message Classes (SystemMessage, HumanMessage, AIMessage)
 * 4. Multi-turn Conversation & Few-shot Prompting
 * 5. Streaming Responses Chunk by Chunk (`model.stream`)
 * 6. Batch Invocations (`model.batch`)
 * 7. Prompt Templates & LCEL Chaining (`prompt.pipe(model).pipe(parser)`)
 * 8. Inspecting Response Metadata & Token Usage
 * ============================================================================
 */

// Initialize the Chat Model using centralized environment configuration
const model = new ChatOpenRouter({
  apiKey: env.apiKey,
  model: env.model,
  temperature: 0.7,
  maxTokens: 1024,
});

// ============================================================================
// 1. SIMPLE STRING INVOCATION
// The simplest way to invoke a model with a single question or statement.
// ============================================================================
export async function example1_StringInput() {
  console.log("\n--- 1. Simple String Input ---");

  const response = await model.invoke("In one sentence, what is JavaScript?");

  console.log("Response Content:\n", response.content);
  return response;
}

// ============================================================================
// 2. ROLE-BASED OBJECTS (OpenAI-Style Arrays)
// Passing role and content objects directly.
// ============================================================================
export async function example2_RoleObjects() {
  console.log("\n--- 2. Role-Based Message Objects ---");

  const messages = [
    {
      role: "system",
      content: "You are a concise translator. Translate the text to Spanish.",
    },
    {
      role: "user",
      content: "Learning LangChain is exciting and powerful!",
    },
  ];

  const response = await model.invoke(messages);

  console.log("Response Content:\n", response.content);
  return response;
}

// ============================================================================
// 3. LANGCHAIN MESSAGE CLASSES
// SystemMessage: Sets system instructions/behavior.
// HumanMessage: Represents the user prompt.
// AIMessage: Represents previous model responses.
// ============================================================================
export async function example3_MessageClasses() {
  console.log("\n--- 3. LangChain Message Classes ---");

  const messages = [
    new SystemMessage(
      "You are a strict programming tutor. Always explain concepts with a very short code snippet."
    ),
    new HumanMessage("What is an arrow function in JavaScript?"),
  ];

  const response = await model.invoke(messages);

  console.log("Response Content:\n", response.content);
  return response;
}

// ============================================================================
// 4. MULTI-TURN CONVERSATION & CONTEXT
// Simulating an ongoing chat history by chaining Human and AI messages.
// ============================================================================
export async function example4_MultiTurnChat() {
  console.log("\n--- 4. Multi-Turn Conversation (Chat History) ---");

  const conversation = [
    new SystemMessage("You are a friendly personal travel assistant."),
    new HumanMessage("Hi! I'm planning a 3-day trip to Tokyo."),
    new AIMessage(
      "Hello! Tokyo is wonderful. I recommend Shibuya, Shinjuku, and Asakusa. What are your main interests: food, anime, culture, or shopping?"
    ),
    new HumanMessage("I'm mainly interested in authentic Japanese food and historic temples."),
  ];

  const response = await model.invoke(conversation);

  console.log("Response Content:\n", response.content);
  return response;
}

// ============================================================================
// 5. STREAMING RESPONSES CHUNK BY CHUNK
// Real-time output streaming for better user experience.
// ============================================================================
export async function example5_Streaming() {
  console.log("\n--- 5. Streaming Response ---");
  process.stdout.write("Streaming: ");

  const stream = await model.stream("Write a short haiku about coding.");

  let fullResponse = "";
  for await (const chunk of stream) {
    process.stdout.write(chunk.content);
    fullResponse += chunk.content;
  }
  console.log("\n");

  return fullResponse;
}

// ============================================================================
// 6. BATCH INVOCATIONS
// Process multiple inputs in parallel for efficiency.
// ============================================================================
export async function example6_Batching() {
  console.log("\n--- 6. Batch Invocations ---");

  const prompts = [
    "Name 3 colors that symbolize energy.",
    "Name 3 famous landmarks in France.",
    "Name 3 prime numbers above 50.",
  ];

  const responses = await model.batch(prompts);

  responses.forEach((res, index) => {
    console.log(`[Batch ${index + 1}] Prompt: "${prompts[index]}"`);
    console.log(`Response:\n${res.content}\n`);
  });

  return responses;
}

// ============================================================================
// 7. PROMPT TEMPLATES & LCEL PIPELINES
// Using ChatPromptTemplate and LangChain Expression Language (.pipe).
// ============================================================================
export async function example7_PromptTemplateAndChaining() {
  console.log("\n--- 7. ChatPromptTemplate & Chaining (.pipe) ---");

  const promptTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a technical writer. Explain technical terms for a {audience} audience.",
    ],
    ["human", "Explain {concept} in 2 sentences."],
  ]);

  // LCEL: Prompt -> Model -> OutputParser
  const chain = promptTemplate.pipe(model).pipe(new StringOutputParser());

  const result = await chain.invoke({
    audience: "5-year-old child",
    concept: "Internet",
  });

  console.log("Chained Result:\n", result);
  return result;
}

// ============================================================================
// 8. INSPECTING RESPONSE METADATA & TOKEN USAGE
// Accessing finish reason, token usage, and provider details.
// ============================================================================
export async function example8_ResponseMetadata() {
  console.log("\n--- 8. Response Metadata & Token Usage ---");

  const response = await model.invoke("Give a 5-word motivational quote.");

  console.log("Content:", response.content);
  console.log("Response Metadata:", response.response_metadata);
  console.log("Token Usage (if provided):", response.usage_metadata);

  return response;
}

// ============================================================================
// RUNNER
// ============================================================================
export async function runAllExamples() {
  console.log("==================================================");
  console.log("Running LangChain Chat Model Examples");
  console.log("==================================================");

  await example1_StringInput();
  await example2_RoleObjects();
  await example3_MessageClasses();
  await example4_MultiTurnChat();
  await example5_Streaming();
  await example6_Batching();
  await example7_PromptTemplateAndChaining();
  await example8_ResponseMetadata();

  console.log("All Chat Model examples completed!");
}

// When executed directly (e.g. `node src/chatmodel.js`)
if (process.argv[1]?.endsWith("chatmodel.js")) {
  // Runs example 1 & 3 by default:
  await example1_StringInput();
  await example3_MessageClasses();

  // To run all examples, uncomment the line below:
  // await runAllExamples();
}
