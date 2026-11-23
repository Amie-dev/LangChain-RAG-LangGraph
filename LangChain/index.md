# LangChain Course Notes

## [Chapter 0 > What is LangChain](notes/Chapter%200/README.md)
- What is LangChain & why use it
- LangChain ecosystem (`@langchain/core`, partner packages, LangGraph, LangSmith)
- Project setup with Node.js and ES Modules
- Environment configuration & minimal starter code

## [Chapter 1 > ChatModels and MessageTypes](notes/Chapter%201/README.md)
- Chat Models vs legacy LLMs
- Model parameters (`temperature`, `maxTokens`, `timeout`)
- LangChain Message Classes (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`)
- Input styles & multi-turn conversations
- Streaming, batching, and inspecting token metadata

## [Chapter 2 > Formating LLM Response](notes/Chapter%202/README.md)
- Native Structured Output: `model.withStructuredOutput(schema)` (Zod & JSON Schema)
- Prompt-based structured output: `StructuredOutputParser`
- `JsonOutputParser` and streaming partial JSON
- List parsers (`CommaSeparatedList`, `NumberedList`, `MarkdownList`)
- `XMLOutputParser`
- `StringOutputParser` & custom `BaseOutputParser`

## [Chapter 3 > Runnable](notes/Chapter%203/README.md)
- Core Runnable interface (`invoke`, `batch`, `stream`, `pipe`)
- `RunnableLambda`: Wrapping sync/async JS functions
- `RunnableSequence`: Linear chaining with `.pipe()`
- `RunnableParallel`: Concurrent execution and data aggregation
- `RunnablePassthrough` & `.assign()`: State enrichment
- `RunnableBranch`: Conditional routing (if/else)
- `withFallbacks`: Resilient chains and error recovery

## [Chapter 4 > PromtTemplate and chatPromptTemplate](notes/Chapter%204/README.md)
- `PromptTemplate`: String-based templates with variable substitution
- `ChatPromptTemplate`: Multi-role chat messages (`system`, `human`, `ai`)
- `MessagesPlaceholder`: Dynamic conversation history injection
- `.partial()`: Pre-binding static values and dynamic timestamp generators
- `FewShotChatMessagePromptTemplate`: In-context learning with example pairs
- Formatting methods: `.format()` vs `.invoke()` (PromptValue)

## Chapter 5 > Multi-Agent System

            5.1 > What is an Agent
            5.2 > What is a Tool
            5.3 > Conceptual Guide On Multi-Agent Systems
                  - Definition
                  - Single Agent
                  - Benefits of Multi Agent Systems
            5.4 > Common Agent Architecture 
              
## Chapter 6 > Middleware

             Middleware Provides a way to more tightly control what happens inside the agent

            6.1 > Benefits of Middleware

                 1.Tracking agent behavior with loggin , analytics and debugging
                 2.Transforming Prompts , tool selections and output formatting
                 3.Adding retries, fallbacs and early termination logic
                 4.Applying rate limits, guardrails
            6.2 > Custom Middleware 

                 6.2.1. Hooks
                 6.2.2. Node-style Hooks
                 6.2.3. Wrap Style Hooks

## Chapter 7 > Guardrails
               Guardrails is a mechanism that allows you to Build safety AI apps by validating and filtering content
               