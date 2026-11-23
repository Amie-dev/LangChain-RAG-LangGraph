import { env } from "../env.js";
import { ChatOpenRouter } from "@langchain/openrouter";
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  FewShotChatMessagePromptTemplate,
} from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";

/**
 * ============================================================================
 * LANGCHAIN PROMPT TEMPLATES & CHAT PROMPT TEMPLATES MASTER GUIDE
 * ============================================================================
 *
 * Prompts are how you guide and instruct LLMs. Instead of hardcoding static strings,
 * Prompt Templates allow you to create dynamic, reusable, and reproducible prompts
 * with placeholders, chat histories, few-shot examples, and partial variable bindings.
 *
 * Topics Covered in this File:
 * 1. PromptTemplate: String-based templates (single string prompt)
 * 2. ChatPromptTemplate: Message-based templates (for Chat Models)
 * 3. Inspecting PromptValue (`.format()` vs `.invoke()`)
 * 4. MessagesPlaceholder: Injecting dynamic chat history arrays
 * 5. Partial Prompts: Pre-filling variables ahead of time (`.partial()`)
 * 6. FewShotChatMessagePromptTemplate: Multi-turn in-context examples
 * 7. Full LCEL Pipeline: Prompt -> Model -> Structured Output Parser
 * ============================================================================
 */

// Initialize Chat Model
const model = new ChatOpenRouter({
  apiKey: env.apiKey,
  model: env.model,
  temperature: 0.3,
  maxTokens: 512,
});

// ============================================================================
// 1. PROMPT TEMPLATE (String-based)
// Best for generating a single string prompt with {variables}.
// ============================================================================
export async function example1_PromptTemplate() {
  console.log("\n--- 1. PromptTemplate (String Template) ---");

  // Create template using .fromTemplate()
  const template = PromptTemplate.fromTemplate(
    "You are an expert chef. What is the secret to making the perfect {dish} in under {timeMinutes} minutes?"
  );

  // 1. Format to raw string
  const formattedString = await template.format({
    dish: "Italian Carbonara",
    timeMinutes: 20,
  });
  console.log("Formatted String:\n", formattedString);

  // 2. Chaining directly with a Chat Model via LCEL
  const chain = template.pipe(model).pipe(new StringOutputParser());
  const response = await chain.invoke({
    dish: "Italian Carbonara",
    timeMinutes: 20,
  });

  console.log("\nModel Response:\n", response.trim());
  return response;
}

// ============================================================================
// 2. CHAT PROMPT TEMPLATE (Message-based)
// The standard for Chat Models. Composes system, user, and assistant roles.
// ============================================================================
export async function example2_ChatPromptTemplate() {
  console.log("\n--- 2. ChatPromptTemplate (Role-based Messages) ---");

  // Style A: Array of [role, template] tuples (most popular & concise)
  const chatPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a helpful software tutor specializing in {language}. Provide clean, beginner-friendly explanations with short code snippets.",
    ],
    ["human", "How do I use {concept} in {language}?"],
  ]);

  // Invoke prompt to inspect PromptValue
  const promptValue = await chatPrompt.invoke({
    language: "JavaScript",
    concept: "Array.prototype.reduce()",
  });

  console.log("Generated Messages Array:\n", promptValue.toChatMessages());

  // Chain with Model
  const chain = chatPrompt.pipe(model).pipe(new StringOutputParser());
  const response = await chain.invoke({
    language: "JavaScript",
    concept: "Array.prototype.reduce()",
  });

  console.log("\nModel Response:\n", response.trim());
  return response;
}

// ============================================================================
// 3. EXPLICIT MESSAGE PROMPT TEMPLATES
// Using SystemMessagePromptTemplate and HumanMessagePromptTemplate explicitly.
// ============================================================================
export async function example3_ExplicitMessageTemplates() {
  console.log("\n--- 3. Explicit Message Prompt Templates ---");

  const systemTemplate = SystemMessagePromptTemplate.fromTemplate(
    "You are a senior {role}. Maintain a {tone} tone."
  );
  const humanTemplate = HumanMessagePromptTemplate.fromTemplate(
    "Review this code snippet:\n```{language}\n{code}\n```"
  );

  const chatPrompt = ChatPromptTemplate.fromMessages([
    systemTemplate,
    humanTemplate,
  ]);

  const chain = chatPrompt.pipe(model).pipe(new StringOutputParser());

  const response = await chain.invoke({
    role: "Security Engineer",
    tone: "professional and cautious",
    language: "javascript",
    code: "const query = `SELECT * FROM users WHERE id = '${userId}'`;",
  });

  console.log("Security Review Response:\n", response.trim());
  return response;
}

// ============================================================================
// 4. MESSAGES PLACEHOLDER (Dynamic Chat History)
// Dynamically injects an arbitrary array of conversation messages (chat history)
// without hardcoding individual message slots.
// ============================================================================
export async function example4_MessagesPlaceholder() {
  console.log("\n--- 4. MessagesPlaceholder (Chat History) ---");

  const chatPromptWithHistory = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a helpful e-commerce customer support assistant for 'TechGear'.",
    ],
    // Injects previous conversation turns dynamically
    new MessagesPlaceholder("chat_history"),
    ["human", "{user_input}"],
  ]);

  // Simulate existing past conversation messages
  const existingChatHistory = [
    new HumanMessage("Hi, I bought a mechanical keyboard last week (Order #4928)."),
    new AIMessage("Hello! Thank you for purchasing from TechGear. How can I assist with Order #4928?"),
  ];

  const chain = chatPromptWithHistory.pipe(model).pipe(new StringOutputParser());

  const response = await chain.invoke({
    chat_history: existingChatHistory,
    user_input: "The spacebar is sticky. Can I get a replacement?",
  });

  console.log("Context-Aware Support Response:\n", response.trim());
  return response;
}

// ============================================================================
// 5. PARTIAL PROMPT TEMPLATES (Pre-filling Variables)
// Pre-binds some template variables early, allowing other variables to be
// supplied later during execution (e.g. current date, user plan, environment).
// ============================================================================
export async function example5_PartialPrompts() {
  console.log("\n--- 5. Partial Prompt Templates ---");

  const basePrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a customer assistant for users on the {userPlan} plan. System current date: {currentDate}.",
    ],
    ["human", "{query}"],
  ]);

  // Partially bind `userPlan` and dynamic function for `currentDate`
  const proUserPrompt = await basePrompt.partial({
    userPlan: "Enterprise VIP",
    currentDate: () => new Date().toLocaleDateString(),
  });

  // Now you only need to provide `{ query }`!
  const chain = proUserPrompt.pipe(model).pipe(new StringOutputParser());

  const response = await chain.invoke({
    query: "What is my SLA response time guarantee?",
  });

  console.log("Partially Bound Prompt Response:\n", response.trim());
  return response;
}

// ============================================================================
// 6. FEW-SHOT CHAT PROMPT TEMPLATES
// Condition the model by demonstrating input/output examples inside the prompt.
// ============================================================================
export async function example6_FewShotPrompting() {
  console.log("\n--- 6. FewShotChatMessagePromptTemplate ---");

  // 1. Define examples
  const examples = [
    {
      word: "happy",
      antonym: "sad",
    },
    {
      word: "tall",
      antonym: "short",
    },
    {
      word: "energetic",
      antonym: "lethargic",
    },
  ];

  // 2. Define example prompt template
  const examplePrompt = ChatPromptTemplate.fromMessages([
    ["human", "{word}"],
    ["ai", "{antonym}"],
  ]);

  // 3. Assemble few-shot chat prompt
  const fewShotPrompt = new FewShotChatMessagePromptTemplate({
    examplePrompt,
    examples,
    inputVariables: [],
  });

  const finalPrompt = ChatPromptTemplate.fromMessages([
    ["system", "You are an antonym expert. Respond ONLY with the single antonym word."],
    fewShotPrompt,
    ["human", "{word}"],
  ]);

  const chain = finalPrompt.pipe(model).pipe(new StringOutputParser());

  const response = await chain.invoke({ word: "benevolent" });
  console.log("Antonym for 'benevolent':\n", response.trim());
  return response;
}

// ============================================================================
// 7. FULL LCEL PIPELINE: Prompt -> Model -> Structured Output
// Combining ChatPromptTemplate with Zod schema for validated data.
// ============================================================================
export async function example7_PromptWithStructuredOutput() {
  console.log("\n--- 7. ChatPromptTemplate with Structured Output ---");

  const ticketClassificationSchema = z.object({
    category: z.enum(["BILLING", "TECHNICAL", "SALES", "OTHER"]),
    urgency: z.enum(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    suggestedAction: z.string().describe("Recommended next step for the support team"),
    confidenceScore: z.number().min(0).max(1),
  });

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are an automated support ticket triage classifier. Analyze incoming tickets accurately.",
    ],
    ["human", "Ticket Subject: {subject}\nTicket Body: {body}"],
  ]);

  const structuredLlm = model.withStructuredOutput(ticketClassificationSchema);
  const chain = prompt.pipe(structuredLlm);

  const result = await chain.invoke({
    subject: "Production database down after migration",
    body: "Our web app is returning 500 error on all endpoints. Database connection pool exhausted.",
  });

  console.log("Structured Triage Result:\n", result);
  return result;
}

// ============================================================================
// RUNNER
// ============================================================================
export async function runAllExamples() {
  console.log("==================================================");
  console.log("Running LangChain Prompt Templates Examples");
  console.log("==================================================");

  await example1_PromptTemplate();
  await example2_ChatPromptTemplate();
  await example3_ExplicitMessageTemplates();
  await example4_MessagesPlaceholder();
  await example5_PartialPrompts();
  await example6_FewShotPrompting();
  await example7_PromptWithStructuredOutput();

  console.log("\nAll Prompt Template examples completed!");
}

// When executed directly (e.g. `node src/prompt_templates.js`)
if (process.argv[1]?.endsWith("prompt_templates.js")) {
  // Run primary examples (1 & 2) by default:
  await example1_PromptTemplate();
  await example2_ChatPromptTemplate();

  // To run all examples:
  // await runAllExamples();
}

