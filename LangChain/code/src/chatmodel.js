import { config } from "dotenv";
import { fileURLToPath } from "node:url";
import { ChatOpenRouter } from "@langchain/openrouter";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";



config({ path: fileURLToPath(new URL("../.env", import.meta.url)) });

const apiKey = process.env.OPENROUTER_API_KEY;

if (!apiKey) {
  throw new Error(
    "Missing OPENROUTER_API_KEY. Add it to LangChain/code/.env.",
  );
}

const model = new ChatOpenRouter({
  model: "nvidia/nemotron-3-ultra-550b-a55b:free",
  temperature: 0,
  maxTokens: 1024,
  apiKey,
});

//////////////////////////////////

const aiMsg0= await model.invoke("Hello")

console.log("\n\n",aiMsg0,"\n\n")
console.log(aiMsg0.content);


/////////


const aiMsg01 = await model.invoke([
  {
    role: "system",
    content:
      "You are a helpful assistant that translates English to French. Translate the user sentence.",
  },
  {
    role: "user",
    content: "I love programming.",
  },
]);


console.log("\n\n",aiMsg01,"\n\n")
console.log(aiMsg01.content);


/////////////////////////


let message=[
  {
    role: "system",
    content:
      "You are a helpful assistant that translates English to French. Translate the user sentence.",
  },
  {
    role: "user",
    content: "I love programming.",
  },
]
const aiMsg02 = await model.invoke(message);





console.log("\n\n",aiMsg02,"\n\n")
console.log(aiMsg02.content);


// ===MESSAGE TYPES IN LANGCHAIN======

// System message
// HumanMessage
// AIMessage
// ToolMessage


const aiMsg03=await model.invoke([
   new SystemMessage(`Your are helpful ai assistant respond the user only in English.
          RULES
          - if the user provide or input a message which is not in English language,
             say eg: sorry I dont understand this language
          `),
  new AIMessage('you are an expert tech teacher who now about techonology ,programming langurages'),
  new HumanMessage("what is javaScript")
])

console.log("\n\n",aiMsg03,"\n\n")
console.log(aiMsg03.content);