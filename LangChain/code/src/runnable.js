import { env } from "../env.js";
import { ChatOpenRouter } from "@langchain/openrouter";
import {
  RunnableLambda,
  RunnableSequence,
  RunnableParallel,
  RunnablePassthrough,
  RunnableBranch,
} from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

/**
 * ============================================================================
 * LANGCHAIN RUNNABLES & LCEL MASTER GUIDE
 * ============================================================================
 *
 * The Runnable protocol is the fundamental building block of LangChain Expression
 * Language (LCEL). Every component (ChatModels, Prompts, Parsers, Retrievers)
 * implements this interface.
 *
 * Core Methods available on EVERY Runnable:
 * 1. .invoke(input)   -> Processes a single input synchronously/asynchronously.
 * 2. .batch([inputs]) -> Processes multiple inputs concurrently.
 * 3. .stream(input)   -> Streams chunks of output as they are generated.
 * 4. .pipe(next)      -> Composes this Runnable sequentially with another.
 *
 * Topics Covered in this File:
 * 1. RunnableLambda: Transforming JS functions into Runnables.
 * 2. RunnableSequence & .pipe(): Linear sequential pipelines.
 * 3. RunnableParallel: Running multiple tasks simultaneously on the same input.
 * 4. RunnablePassthrough & .assign(): Preserving and enriching pipeline state.
 * 5. RunnableBranch: Conditional routing (if / else branching).
 * 6. Real-World LLM Pipeline: Parallel multi-angle text analysis.
 * 7. Fallback Handling: Using withFallbacks for resilient systems.
 * ============================================================================
 */

// Initialize model
const model = new ChatOpenRouter({
  apiKey: env.apiKey,
  model: env.model,
  temperature: 0,
  maxTokens: 512,
});

// ============================================================================
// 1. RUNNABLE LAMBDA: Wrapping Pure JavaScript Functions
// Turn any sync or async function into a full Runnable with invoke, batch & pipe.
// ============================================================================
export async function example1_RunnableLambda() {
  console.log("\n--- 1. RunnableLambda Basics ---");

  // Define pure JavaScript functions
  const addPrefix = (text) => `[LOG]: ${text}`;
  const toUpperCase = (text) => text.toUpperCase();

  // Convert to Runnables
  const logRunnable = RunnableLambda.from(addPrefix);
  const upperRunnable = RunnableLambda.from(toUpperCase);

  // 1. Single invoke
  const single = await logRunnable.invoke("system online");
  console.log("invoke():", single);

  // 2. Batch processing (runs across array of items automatically)
  const batchResult = await upperRunnable.batch(["react", "langchain", "node"]);
  console.log("batch():", batchResult);

  return { single, batchResult };
}

// ============================================================================
// 2. RUNNABLE SEQUENCE & .pipe()
// Sequentially chaining operations where output of step A becomes input to B.
// ============================================================================
export async function example2_RunnableSequence() {
  console.log("\n--- 2. RunnableSequence & .pipe() ---");

  const trim = RunnableLambda.from((str) => str.trim());
  const capitalize = RunnableLambda.from((str) => str.charAt(0).toUpperCase() + str.slice(1));
  const addExclamation = RunnableLambda.from((str) => `${str}!`);

  // Chain using .pipe() (syntactic sugar for RunnableSequence)
  const pipeline = trim.pipe(capitalize).pipe(addExclamation);

  const result = await pipeline.invoke("   welcome to langchain runnables   ");
  console.log("Pipeline output:", result);

  // Can also create explicitly via RunnableSequence.from([...])
  const explicitSequence = RunnableSequence.from([trim, capitalize, addExclamation]);
  const explicitResult = await explicitSequence.invoke("   pipeline sequence   ");
  console.log("Explicit sequence output:", explicitResult);

  return result;
}

// ============================================================================
// 3. RUNNABLE PARALLEL: Concurrent Branch Execution
// Takes a single input and distributes it across multiple branches simultaneously.
// Returns an object whose keys match the branch names.
// ============================================================================
export async function example3_RunnableParallel() {
  console.log("\n--- 3. RunnableParallel ---");

  const parallel = RunnableParallel.from({
    wordCount: RunnableLambda.from((text) => text.trim().split(/\s+/).length),
    charCount: RunnableLambda.from((text) => text.length),
    shouting: RunnableLambda.from((text) => text.toUpperCase()),
    reversed: RunnableLambda.from((text) => text.split("").reverse().join("")),
  });

  const input = "LangChain makes AI development fast and enjoyable";
  const result = await parallel.invoke(input);

  console.log("Input:", input);
  console.log("Parallel Analysis Result:\n", result);
  return result;
}

// ============================================================================
// 4. RUNNABLE PASSTHROUGH & .assign()
// Preserving original inputs while adding new computed fields.
// ============================================================================
export async function example4_RunnablePassthrough() {
  console.log("\n--- 4. RunnablePassthrough & .assign() ---");

  // Say our input is an object: { topic: "Quantum Computing" }
  // We want to keep `topic` and add an `estimatedReadingTime` and `uppercaseTopic` field.
  const enrichChain = RunnablePassthrough.assign({
    uppercaseTopic: (input) => input.topic.toUpperCase(),
    topicLength: (input) => input.topic.length,
    timestamp: () => new Date().toISOString(),
  });

  const inputObj = { topic: "Artificial Neural Networks" };
  const enriched = await enrichChain.invoke(inputObj);

  console.log("Original Input:", inputObj);
  console.log("Enriched Object:\n", enriched);
  return enriched;
}

// ============================================================================
// 5. RUNNABLE BRANCH: Conditional Routing (If / Else Logic)
// Routes input to different sub-chains depending on predicate functions.
// ============================================================================
export async function example5_RunnableBranch() {
  console.log("\n--- 5. RunnableBranch (Conditional Logic) ---");

  // Predicates
  const isQuestion = (input) => input.trim().endsWith("?");
  const isExclamation = (input) => input.trim().endsWith("!");

  // Branch branches: [condition, runnable] pairs + default fallback runnable
  const branch = RunnableBranch.from([
    [isQuestion, RunnableLambda.from((text) => `Handling as a QUESTION: "${text}"`)],
    [isExclamation, RunnableLambda.from((text) => `Handling as an URGENT EXCLAMATION: "${text}"`)],
    RunnableLambda.from((text) => `Handling as a STANDARD STATEMENT: "${text}"`), // default
  ]);

  const testInputs = [
    "What is the airspeed velocity of an unladen swallow?",
    "System overload detected!",
    "The weather today is cloudy with light rain.",
  ];

  for (const item of testInputs) {
    const outcome = await branch.invoke(item);
    console.log(outcome);
  }
}

// ============================================================================
// 6. REAL-WORLD LLM PIPELINE: Multi-Angle Parallel Analysis
// Combining ChatPromptTemplate, model, StringOutputParser, and RunnableParallel.
// ============================================================================
export async function example6_RealWorldLLMPipeline() {
  console.log("\n--- 6. Real-World LLM Pipeline (Parallel Analysis) ---");

  const parser = new StringOutputParser();

  // Branch 1: One-sentence summary
  const summaryChain = ChatPromptTemplate.fromMessages([
    ["system", "Summarize the text in exactly one sentence."],
    ["human", "{text}"],
  ]).pipe(model).pipe(parser);

  // Branch 2: Sentiment detection
  const sentimentChain = ChatPromptTemplate.fromMessages([
    ["system", "Classify the sentiment of the text as POSITIVE, NEGATIVE, or NEUTRAL. Answer with one word."],
    ["human", "{text}"],
  ]).pipe(model).pipe(parser);

  // Branch 3: Actionable takeaway
  const takeawayChain = ChatPromptTemplate.fromMessages([
    ["system", "Provide 1 key actionable takeaway from the text."],
    ["human", "{text}"],
  ]).pipe(model).pipe(parser);

  // Combine branches in parallel
  const analysisPipeline = RunnableParallel.from({
    originalText: (input) => input.text,
    summary: summaryChain,
    sentiment: sentimentChain,
    takeaway: takeawayChain,
  });

  const textToAnalyze =
    "We launched our new web application yesterday. While the response time was twice as fast as before, three critical database connection bugs were discovered by users within the first hour.";

  console.log("Analyzing text in parallel with LLM...");
  const analysis = await analysisPipeline.invoke({ text: textToAnalyze });

  console.log("\nResults:");
  console.log("- Summary:", analysis.summary.trim());
  console.log("- Sentiment:", analysis.sentiment.trim());
  console.log("- Takeaway:", analysis.takeaway.trim());
  return analysis;
}

// ============================================================================
// 7. FALLBACK HANDLING (withFallbacks)
// Automatically recover when a primary runnable fails.
// ============================================================================
export async function example7_Fallbacks() {
  console.log("\n--- 7. Resilient Chains with withFallbacks ---");

  // A flaky service that throws an error
  const flakyService = RunnableLambda.from(() => {
    throw new Error("Primary API service temporarily unavailable (503)");
  });

  // A safe fallback service
  const fallbackService = RunnableLambda.from(() => {
    return "Fallback response: cached data served safely.";
  });

  // Attach fallback
  const robustChain = flakyService.withFallbacks({
    fallbacks: [fallbackService],
  });

  const result = await robustChain.invoke("fetch data");
  console.log("Result after fallback:", result);
  return result;
}

// ============================================================================
// RUNNER
// ============================================================================
export async function runAllExamples() {
  console.log("==================================================");
  console.log("Running LangChain Runnables & LCEL Examples");
  console.log("==================================================");

  await example1_RunnableLambda();
  await example2_RunnableSequence();
  await example3_RunnableParallel();
  await example4_RunnablePassthrough();
  await example5_RunnableBranch();
  await example6_RealWorldLLMPipeline();
  await example7_Fallbacks();

  console.log("\nAll Runnable examples completed successfully!");
}

// When executed directly (e.g. `node src/runnable.js`)
if (process.argv[1]?.endsWith("runnable.js")) {
  // Run pure JS runnable examples (1-4) by default:
  await example1_RunnableLambda();
  await example2_RunnableSequence();
  await example3_RunnableParallel();
  await example4_RunnablePassthrough();

  // To run all examples including LLM parallel calls:
  // await runAllExamples();
}
