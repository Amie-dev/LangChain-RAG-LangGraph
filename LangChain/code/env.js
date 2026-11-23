import { config } from "dotenv";
import { fileURLToPath } from "node:url";

config({ path: fileURLToPath(new URL(".env", import.meta.url)) });

const apiKey = process.env.OPENROUTER_API_KEY;

if (!apiKey) {
  throw new Error(
    "Missing OPENROUTER_API_KEY. Add it to LangChain/code/.env.",
  );
}

export const env = Object.freeze({
  apiKey,
  model: "nvidia/nemotron-3-ultra-550b-a55b:free",
});