import { env } from "../env.js";
import { ChatOpenRouter } from "@langchain/openrouter";
import { z } from "zod";

const llm = new ChatOpenRouter({
  apiKey: env.apiKey,
  model: env.model,
  temperature: 0,
  maxTokens: 1024,
});


const explanationSchema = z
  .object({
    topic: z.string().describe("The topic being explained"),
    summary: z.string().describe("A concise explanation of the topic"),
    keyPoints: z
      .array(z.string())
      .describe("The most important facts, as short bullet points"),
    example: z.string().describe("A simple example that illustrates the topic"),
  })
  .describe("A clear, beginner-friendly technical explanation");

const structuredLlm = llm.withStructuredOutput(explanationSchema);

const result = await structuredLlm.invoke(
  "Explain JavaScript closures to a beginner.",
);

console.log(result)

console.dir(result, { depth: null });
