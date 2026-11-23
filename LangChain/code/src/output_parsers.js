import { env } from "../env.js";
import { ChatOpenRouter } from "@langchain/openrouter";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  StringOutputParser,
  CommaSeparatedListOutputParser,
  NumberedListOutputParser,
  MarkdownListOutputParser,
  StructuredOutputParser,
  JsonOutputParser,
  XMLOutputParser,
  BaseOutputParser,
} from "@langchain/core/output_parsers";
import { z } from "zod";

/**
 * ============================================================================
 * LANGCHAIN OUTPUT PARSERS & STRUCTURED OUTPUT MASTER GUIDE
 * ============================================================================
 *
 * This file demonstrates all major methods for obtaining structured output in LangChain:
 *
 * 1. Native Structured Output (`model.withStructuredOutput`) [RECOMMENDED]
 *    1A. Using Zod Schema
 *    1B. Using JSON Schema directly (without Zod)
 *    1C. With `{ includeRaw: true }` (captures raw message + parsed data)
 *
 * 2. StructuredOutputParser (Prompt-based instruction parsing)
 *    2A. From Zod Schema (`StructuredOutputParser.fromZodSchema`)
 *    2B. From Names & Descriptions (`StructuredOutputParser.fromNamesAndDescriptions`)
 *
 * 3. JsonOutputParser (Generic JSON parsing with streaming support)
 *
 * 4. List Output Parsers
 *    4A. CommaSeparatedListOutputParser
 *    4B. NumberedListOutputParser
 *    4C. MarkdownListOutputParser
 *
 * 5. XMLOutputParser (Parses XML tags into JSON objects)
 *
 * 6. StringOutputParser (Extracts plain string content from AIMessage)
 *
 * 7. Custom Output Parser (Extending `BaseOutputParser`)
 * ============================================================================
 */

// Initialize the Chat Model
const model = new ChatOpenRouter({
  apiKey: env.apiKey,
  model: env.model,
  temperature: 0,
  maxTokens: 1024,
});

// ============================================================================
// 1. NATIVE STRUCTURED OUTPUT (model.withStructuredOutput)
// Best practice for models supporting Tool Calling / Function Calling or JSON mode.
// ============================================================================

// 1A. withStructuredOutput using a Zod Schema
export async function example1A_StructuredWithZod() {
  console.log("\n--- 1A: withStructuredOutput (Zod Schema) ---");

  const movieSchema = z
    .object({
      title: z.string().describe("The title of the movie"),
      year: z.number().describe("The release year"),
      director: z.string().describe("The director's name"),
      genres: z.array(z.string()).describe("List of genres"),
      rating: z.number().min(0).max(10).describe("Rating out of 10"),
    })
    .describe("Details about a movie");

  const structuredLlm = model.withStructuredOutput(movieSchema);
  const result = await structuredLlm.invoke("Give details about the movie Inception.");

  console.log("Parsed Object:", result);
  return result;
}

// 1B. withStructuredOutput using pure JSON Schema (no Zod dependency)
export async function example1B_StructuredWithJsonSchema() {
  console.log("\n--- 1B: withStructuredOutput (Pure JSON Schema) ---");

  const countryJsonSchema = {
    name: "country_info",
    description: "Geographical and demographic info about a country",
    parameters: {
      type: "object",
      properties: {
        country: { type: "string", description: "Country name" },
        capital: { type: "string", description: "Capital city" },
        populationInMillions: { type: "number", description: "Estimated population in millions" },
        officialLanguages: {
          type: "array",
          items: { type: "string" },
          description: "Official languages spoken",
        },
      },
      required: ["country", "capital", "populationInMillions", "officialLanguages"],
    },
  };

  const structuredLlm = model.withStructuredOutput(countryJsonSchema);
  const result = await structuredLlm.invoke("Provide details for Japan.");

  console.log("Parsed Object:", result);
  return result;
}

// 1C. withStructuredOutput with { includeRaw: true }
// Returns both the raw AIMessage and the parsed object (or parsing error if it failed)
export async function example1C_StructuredWithRaw() {
  console.log("\n--- 1C: withStructuredOutput ({ includeRaw: true }) ---");

  const userSchema = z.object({
    username: z.string(),
    email: z.string().email(),
    skills: z.array(z.string()),
  });

  const structuredLlm = model.withStructuredOutput(userSchema, {
    name: "user_profile",
    includeRaw: true,
  });

  const result = await structuredLlm.invoke(
    "Generate a fictional developer profile: username 'coder42', email 'coder42@example.com', skills JavaScript and Python."
  );

  console.log("Raw Message:", result.raw);
  console.log("Parsed Output:", result.parsed);
  return result;
}

// ============================================================================
// 2. StructuredOutputParser (Prompt-based structured output)
// Works with ANY model (even those without function/tool calling support) by
// injecting format instructions into the prompt and parsing JSON from the response.
// ============================================================================

// 2A. StructuredOutputParser with Zod
export async function example2A_StructuredOutputParserZod() {
  console.log("\n--- 2A: StructuredOutputParser.fromZodSchema ---");

  const recipeSchema = z.object({
    recipeName: z.string().describe("Name of the recipe"),
    prepTimeMinutes: z.number().describe("Preparation time in minutes"),
    ingredients: z.array(z.string()).describe("List of ingredients with quantities"),
    steps: z.array(z.string()).describe("Step-by-step cooking instructions"),
  });

  const parser = StructuredOutputParser.fromZodSchema(recipeSchema);

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a professional chef. Provide delicious recipes.\n{format_instructions}",
    ],
    ["human", "How do I make {dish}?"],
  ]);

  const chain = prompt.pipe(model).pipe(parser);

  const result = await chain.invoke({
    dish: "Pancakes",
    format_instructions: parser.getFormatInstructions(),
  });

  console.log("Parsed Recipe:", result);
  return result;
}

// 2B. StructuredOutputParser with Names and Descriptions
export async function example2B_StructuredOutputParserNamesAndDescriptions() {
  console.log("\n--- 2B: StructuredOutputParser.fromNamesAndDescriptions ---");

  const parser = StructuredOutputParser.fromNamesAndDescriptions({
    language: "The programming language name",
    yearCreated: "The year the language first appeared",
    creator: "The person or organization who created it",
    primaryUseCases: "A short comma-separated string of primary use cases",
  });

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer questions about programming history.\n{format_instructions}",
    ],
    ["human", "Tell me about {language}."],
  ]);

  const chain = prompt.pipe(model).pipe(parser);

  const result = await chain.invoke({
    language: "Rust",
    format_instructions: parser.getFormatInstructions(),
  });

  console.log("Parsed Language Info:", result);
  return result;
}

// ============================================================================
// 3. JsonOutputParser (Generic JSON output & Streaming)
// Parses LLM output into a generic JSON structure.
// ============================================================================
export async function example3_JsonOutputParser() {
  console.log("\n--- 3: JsonOutputParser ---");

  const parser = new JsonOutputParser();

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Return a JSON object with 3 book recommendations for the given topic.\n{format_instructions}",
    ],
    ["human", "Topic: {topic}"],
  ]);

  const chain = prompt.pipe(model).pipe(parser);

  const result = await chain.invoke({
    topic: "System Design & Distributed Systems",
    format_instructions: parser.getFormatInstructions(),
  });

  console.log("Parsed JSON:", result);
  return result;
}

// ============================================================================
// 4. LIST OUTPUT PARSERS
// Convenient parsers for turning LLM responses into JavaScript arrays.
// ============================================================================

// 4A. CommaSeparatedListOutputParser
export async function example4A_CommaSeparatedList() {
  console.log("\n--- 4A: CommaSeparatedListOutputParser ---");

  const parser = new CommaSeparatedListOutputParser();

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "List the items requested by the user.\n{format_instructions}",
    ],
    ["human", "List 5 major CSS layout mechanisms or properties."],
  ]);

  const chain = prompt.pipe(model).pipe(parser);

  const result = await chain.invoke({
    format_instructions: parser.getFormatInstructions(),
  });

  console.log("Parsed Array:", result);
  return result;
}

// 4B. NumberedListOutputParser
export async function example4B_NumberedList() {
  console.log("\n--- 4B: NumberedListOutputParser ---");

  const parser = new NumberedListOutputParser();

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Provide a numbered list of steps.\n{format_instructions}",
    ],
    ["human", "What are 3 essential steps to secure a REST API?"],
  ]);

  const chain = prompt.pipe(model).pipe(parser);

  const result = await chain.invoke({
    format_instructions: parser.getFormatInstructions(),
  });

  console.log("Parsed Numbered List:", result);
  return result;
}

// 4C. MarkdownListOutputParser
export async function example4C_MarkdownList() {
  console.log("\n--- 4C: MarkdownListOutputParser ---");

  const parser = new MarkdownListOutputParser();

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Provide a markdown bulleted list.\n{format_instructions}",
    ],
    ["human", "List 4 key features of TypeScript over JavaScript."],
  ]);

  const chain = prompt.pipe(model).pipe(parser);

  const result = await chain.invoke({
    format_instructions: parser.getFormatInstructions(),
  });

  console.log("Parsed Markdown Bullets:", result);
  return result;
}

// ============================================================================
// 5. XML OUTPUT PARSER
// Useful when models prefer XML syntax or when generating hierarchical tags.
// ============================================================================
export async function example5_XMLOutputParser() {
  console.log("\n--- 5: XMLOutputParser ---");

  const parser = new XMLOutputParser({
    tags: ["movie", "title", "director", "year"],
  });

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Generate an XML response adhering to the format instructions.\n{format_instructions}",
    ],
    ["human", "Provide details for the movie 'The Matrix'."],
  ]);

  const chain = prompt.pipe(model).pipe(parser);

  const result = await chain.invoke({
    format_instructions: parser.getFormatInstructions(),
  });

  console.log("Parsed XML to Object:", result);
  return result;
}

// ============================================================================
// 6. STRING OUTPUT PARSER
// Extracts the text content directly from AIMessage / AIMessageChunk.
// ============================================================================
export async function example6_StringOutputParser() {
  console.log("\n--- 6: StringOutputParser ---");

  const parser = new StringOutputParser();

  const prompt = ChatPromptTemplate.fromTemplate(
    "Write a witty one-liner programmer joke about {topic}."
  );

  const chain = prompt.pipe(model).pipe(parser);

  const result = await chain.invoke({ topic: "debugging" });

  console.log("Parsed String Result:", result);
  return result;
}

// ============================================================================
// 7. CUSTOM OUTPUT PARSER (Extending BaseOutputParser)
// Build your own parser with custom transformation or validation logic.
// ============================================================================
class SentimentOutputParser extends BaseOutputParser {
  lc_namespace = ["langchain", "output_parsers"];

  getFormatInstructions() {
    return 'Respond ONLY with one of the following words: POSITIVE, NEGATIVE, or NEUTRAL.';
  }

  async parse(text) {
    const cleaned = text.trim().toUpperCase();
    if (cleaned.includes("POSITIVE")) return { sentiment: "POSITIVE", score: 1.0 };
    if (cleaned.includes("NEGATIVE")) return { sentiment: "NEGATIVE", score: -1.0 };
    if (cleaned.includes("NEUTRAL")) return { sentiment: "NEUTRAL", score: 0.0 };

    throw new Error(`Invalid sentiment value received: "${text}"`);
  }
}

export async function example7_CustomOutputParser() {
  console.log("\n--- 7: Custom Output Parser (BaseOutputParser) ---");

  const customParser = new SentimentOutputParser();

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Analyze the sentiment of the text.\n{format_instructions}",
    ],
    ["human", "Customer review: {review}"],
  ]);

  const chain = prompt.pipe(model).pipe(customParser);

  const result = await chain.invoke({
    review: "The application is blazing fast and the user interface is intuitive!",
    format_instructions: customParser.getFormatInstructions(),
  });

  console.log("Custom Parsed Output:", result);
  return result;
}

// ============================================================================
// MAIN RUNNER
// Uncomment or execute the examples you want to test.
// ============================================================================
export async function runAllExamples() {
  console.log("==================================================");
  console.log("Running LangChain Output Parsers Examples");
  console.log("==================================================");

  // 1. Native Structured Output
  await example1A_StructuredWithZod();
  await example1B_StructuredWithJsonSchema();
  await example1C_StructuredWithRaw();

  // 2. StructuredOutputParser
  await example2A_StructuredOutputParserZod();
  await example2B_StructuredOutputParserNamesAndDescriptions();

  // 3. JsonOutputParser
  await example3_JsonOutputParser();

  // 4. List Parsers
  await example4A_CommaSeparatedList();
  await example4B_NumberedList();
  await example4C_MarkdownList();

  // 5. XML Output Parser
  await example5_XMLOutputParser();

  // 6. String Output Parser
  await example6_StringOutputParser();

  // 7. Custom Output Parser
  await example7_CustomOutputParser();

  console.log("\nAll examples executed successfully!");
}

// When executed directly (e.g. `node src/output_parsers.js`)
if (process.argv[1]?.endsWith("output_parsers.js")) {
  // Runs the primary recommended example (1A) by default:
  await example1A_StructuredWithZod();

  // To run all examples, uncomment the line below:
  // await runAllExamples();
}

