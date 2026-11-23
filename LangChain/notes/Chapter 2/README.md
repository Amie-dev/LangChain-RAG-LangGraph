# Chapter 2: Formatting LLM Response (Output Parsers & Structured Output)

## 1. Why Output Formatting Matters

By default, Large Language Models output freeform natural language strings. In real-world software, applications require **deterministic, typed, and structured data** (e.g. JSON objects, typed arrays, or validated schemas).

LangChain provides two primary ways to achieve structured output:
1. **Model-level Structured Output (`withStructuredOutput`)**: Uses the model's native Tool Calling or JSON Mode (Recommended).
2. **Prompt-based Output Parsers (`OutputParser`)**: Injects format instructions into the prompt and parses the LLM's string output.

---

## 2. Overview of Available Methods

```mermaid
flowchart TD
    Start["Need Structured Output?"] --> Q1{"Does the model support Tool Calling / JSON Mode?"}
    Q1 -->|Yes (Recommended)| M1["model.withStructuredOutput(schema)"]
    Q1 -->|No| Q2{"What format do you need?"}
    Q2 -->|Complex JSON Object| M2["StructuredOutputParser.fromZodSchema()"]
    Q2 -->|Generic JSON / Streaming| M3["JsonOutputParser()"]
    Q2 -->|Arrays / Lists| M4["List Parsers (Comma, Numbered, Markdown)"]
    Q2 -->|XML / Hierarchical| M5["XMLOutputParser()"]
    Q2 -->|Domain-Specific Rule| M6["Custom BaseOutputParser"]
```

---

## 3. Method 1: Native Structured Output (`withStructuredOutput`)

Modern models (OpenAI GPT-4o, Claude 3.5, Gemini, Nemotron) natively understand schema definitions via **Tool Calling / Function Calling**.

### A. Using a Zod Schema
```javascript
import { z } from "zod";

const bookSchema = z
  .object({
    title: z.string().describe("The book's title"),
    author: z.string().describe("The author's full name"),
    publicationYear: z.number().describe("Year published"),
    genres: z.array(z.string()).describe("List of genres"),
    summary: z.string().describe("Brief 2-sentence summary"),
  })
  .describe("Structured information about a book");

// Bind schema directly to the model
const structuredLlm = model.withStructuredOutput(bookSchema);

// Invoke returns a fully parsed, validated JavaScript object!
const result = await structuredLlm.invoke("Give details about the book 'Clean Code'.");
console.log(result.title); // "Clean Code"
```

### B. Using Pure JSON Schema (No Zod Dependency)
If you do not want to use Zod, you can pass standard JSON Schema parameters:
```javascript
const schema = {
  name: "weather_report",
  description: "Weather details for a city",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string" },
      temperatureCelsius: { type: "number" },
      conditions: { type: "string" },
    },
    required: ["city", "temperatureCelsius", "conditions"],
  },
};

const structuredLlm = model.withStructuredOutput(schema);
const weather = await structuredLlm.invoke("What's the weather in Tokyo?");
```

### C. Capturing Raw LLM Response with `{ includeRaw: true }`
When debugging or monitoring token usage alongside structured data:
```javascript
const structuredLlm = model.withStructuredOutput(bookSchema, {
  includeRaw: true,
});

const { raw, parsed } = await structuredLlm.invoke("Tell me about Dune.");
console.log("Raw AIMessage:", raw);
console.log("Parsed Object:", parsed);
```

---

## 4. Method 2: `StructuredOutputParser` (Prompt-Based)

For models that do not support tool calling, `StructuredOutputParser` injects explicit formatting rules into the prompt:

### With Zod Schema
```javascript
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";

const parser = StructuredOutputParser.fromZodSchema(
  z.object({
    answer: z.string().describe("The answer to the user's question"),
    confidence: z.number().min(0).max(1).describe("Confidence score between 0 and 1"),
  })
);

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "Answer the question accurately.\n{format_instructions}"],
  ["human", "{question}"],
]);

// Chain: Prompt -> Model -> Parser
const chain = prompt.pipe(model).pipe(parser);

const result = await chain.invoke({
  question: "What is the boiling point of water in Celsius?",
  format_instructions: parser.getFormatInstructions(),
});

console.log(result.answer); // "100°C"
```

### With Names and Descriptions (No Zod)
```javascript
const parser = StructuredOutputParser.fromNamesAndDescriptions({
  capital: "The capital city",
  currency: "The official currency",
});
```

---

## 5. Method 3: `JsonOutputParser`

A lightweight parser that expects a JSON response and parses it into a JavaScript object. Ideal when you want streaming JSON parsing without a strict schema upfront:

```javascript
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const parser = new JsonOutputParser();

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "Return a JSON object with top 3 programming languages.\n{format_instructions}"],
  ["human", "Query: most popular languages for web development"],
]);

const chain = prompt.pipe(model).pipe(parser);
const result = await chain.invoke({
  format_instructions: parser.getFormatInstructions(),
});
```

---

## 6. Method 4: List Output Parsers

LangChain provides specialized list parsers that turn text into JavaScript arrays:

| Parser | Format Expected | Output |
| :--- | :--- | :--- |
| `CommaSeparatedListOutputParser` | `item1, item2, item3` | `["item1", "item2", "item3"]` |
| `NumberedListOutputParser` | `1. item1\n2. item2` | `["item1", "item2"]` |
| `MarkdownListOutputParser` | `* item1\n* item2` | `["item1", "item2"]` |

### Example with `CommaSeparatedListOutputParser`:
```javascript
import { CommaSeparatedListOutputParser } from "@langchain/core/output_parsers";

const parser = new CommaSeparatedListOutputParser();
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "Provide a comma separated list.\n{format_instructions}"],
  ["human", "List 4 popular JavaScript testing frameworks."],
]);

const chain = prompt.pipe(model).pipe(parser);
const result = await chain.invoke({
  format_instructions: parser.getFormatInstructions(),
});
console.log(result); // ["Jest", "Vitest", "Mocha", "Cypress"]
```

---

## 7. Method 5: `XMLOutputParser`

Useful for models (such as Claude) that excel at generating XML, or when generating hierarchical tag-based data:

```javascript
import { XMLOutputParser } from "@langchain/core/output_parsers";

const parser = new XMLOutputParser({ tags: ["book", "title", "author"] });

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "Generate XML adhering to the schema.\n{format_instructions}"],
  ["human", "Information on the book 1984."],
]);

const chain = prompt.pipe(model).pipe(parser);
const result = await chain.invoke({
  format_instructions: parser.getFormatInstructions(),
});
```

---

## 8. Method 6: `StringOutputParser`

Extracts the plain `.content` string directly from the model's `AIMessage`, removing metadata wrappers:

```javascript
import { StringOutputParser } from "@langchain/core/output_parsers";

const chain = prompt.pipe(model).pipe(new StringOutputParser());
const resultText = await chain.invoke({ ... }); // Type is string
```

---

## 9. Method 7: Custom Output Parsers

To implement custom validation, error handling, or transformation logic, extend `BaseOutputParser`:

```javascript
import { BaseOutputParser } from "@langchain/core/output_parsers";

class BooleanOutputParser extends BaseOutputParser {
  lc_namespace = ["my_app", "parsers"];

  getFormatInstructions() {
    return 'Respond with either "YES" or "NO" only.';
  }

  async parse(text) {
    const cleaned = text.trim().toUpperCase();
    if (cleaned.includes("YES")) return true;
    if (cleaned.includes("NO")) return false;
    throw new Error(`Unable to parse boolean from: ${text}`);
  }
}
```

---

## 10. Summary & Best Practices

| Scenario | Recommended Approach |
| :--- | :--- |
| Modern model with tool calling | **`model.withStructuredOutput(zodSchema)`** |
| Model lacks tool calling | **`StructuredOutputParser.fromZodSchema(zodSchema)`** |
| Array of simple items | **`CommaSeparatedListOutputParser`** |
| Generic JSON with streaming | **`JsonOutputParser`** |
| Plain text pipeline output | **`StringOutputParser`** |
| Custom business validation rules | Subclass **`BaseOutputParser`** |

