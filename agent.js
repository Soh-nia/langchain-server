/**
 * agent.js — manual ReAct agent, no @langchain/langgraph dependency
 *
 * Removes the langgraph dependency entirely to avoid the @langchain/core
 * version conflict. Implements the same agent behaviour manually using
 * tool_calling directly on the chat model — which is all langgraph's
 * createReactAgent does internally anyway.
 *
 * HOW THIS WORKS (ReAct loop without langgraph):
 * 1. Bind tools to the LLM so it can call them
 * 2. Loop: invoke model → if tool_call, run the tool → append result → repeat
 * 3. Stop when the model returns a plain text response (no tool calls)
 */

import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { createClient } from "@supabase/supabase-js";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";

// ─── CLIENTS ──────────────────────────────────────────────────────────────────

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY,
);

class GeminiEmbeddings {
  async embedDocuments(texts) {
    const results = [];
    for (const text of texts) {
      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1/models/gemini-embedding-001:embedContent?key=${process.env.GEMINI_API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "models/gemini-embedding-001",
            content: { parts: [{ text }] },
            taskType: "RETRIEVAL_QUERY",
            outputDimensionality: 768,
          }),
        },
      );
      const data = await res.json();
      if (!res.ok)
        throw new Error(data.error?.message || `Embedding error ${res.status}`);
      results.push(data.embedding.values);
    }
    return results;
  }

  async embedQuery(text) {
    const [embedding] = await this.embedDocuments([text]);
    return embedding;
  }
}

const embeddings = new GeminiEmbeddings();

const vectorStore = new SupabaseVectorStore(embeddings, {
  client: supabase,
  tableName: "documents",
  queryName: "match_documents",
});

// ─── TOOLS ────────────────────────────────────────────────────────────────────

const searchKnowledgeBaseTool = tool(
  async ({ query, numResults = 4 }) => {
    try {
      const docs = await vectorStore.similaritySearch(query, numResults);
      if (!docs.length)
        return "No relevant documents found in the knowledge base.";
      return docs
        .map(
          (doc, i) =>
            `[Source: ${doc.metadata?.source || "Unknown"}, Chunk ${i + 1}]\n${doc.pageContent}`,
        )
        .join("\n\n---\n\n");
    } catch (err) {
      return `Search error: ${err.message}`;
    }
  },
  {
    name: "search_knowledge_base",
    description:
      "Search ALL uploaded documents for information relevant to a query. Use this as your primary tool when the user asks about document contents.",
    schema: z.object({
      query: z.string().describe("The search query"),
      numResults: z.number().optional().default(4),
    }),
  },
);

const searchSpecificSourceTool = tool(
  async ({ query, source, numResults = 4 }) => {
    try {
      const docs = await vectorStore.similaritySearch(query, numResults, {
        filter: { source },
      });
      if (!docs.length) return `No relevant content found in "${source}".`;
      return docs
        .map((doc, i) => `[Chunk ${i + 1} from ${source}]\n${doc.pageContent}`)
        .join("\n\n---\n\n");
    } catch (err) {
      return `Search error: ${err.message}`;
    }
  },
  {
    name: "search_specific_source",
    description:
      "Search a SPECIFIC document by filename. Use when the user asks about a particular document by name.",
    schema: z.object({
      query: z.string().describe("The search query"),
      source: z.string().describe("The exact filename to search within"),
      numResults: z.number().optional().default(4),
    }),
  },
);

const listSourcesTool = tool(
  async () => {
    try {
      const { data, error } = await supabase
        .from("documents")
        .select("metadata")
        .not("metadata->source", "is", null);
      if (error) return `Error: ${error.message}`;
      const sources = [
        ...new Set(data.map((r) => r.metadata?.source).filter(Boolean)),
      ];
      if (!sources.length) return "No documents uploaded yet.";
      return `Documents in knowledge base:\n${sources.map((s, i) => `${i + 1}. ${s}`).join("\n")}`;
    } catch (err) {
      return `Error: ${err.message}`;
    }
  },
  {
    name: "list_available_sources",
    description:
      "List all document filenames in the knowledge base. Use when the user asks what documents are available.",
    schema: z.object({}),
  },
);

// Map tool name → tool instance for lookup during the loop
const TOOLS = {
  search_knowledge_base: searchKnowledgeBaseTool,
  search_specific_source: searchSpecificSourceTool,
  list_available_sources: listSourcesTool,
};

const TOOLS_LIST = Object.values(TOOLS);

// ─── BUILD LLM ────────────────────────────────────────────────────────────────

function buildLLM(provider, model) {
  switch (provider) {
    case "anthropic":
      return new ChatAnthropic({
        apiKey: process.env.ANTHROPIC_API_KEY,
        model: model || "claude-sonnet-4-5",
        maxTokens: 2048,
      });
    case "openai":
      return new ChatOpenAI({
        apiKey: process.env.OPENAI_API_KEY,
        model: model || "gpt-4o-mini",
        maxTokens: 2048,
      });
    default:
      return new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: model || "gemini-2.5-flash",
        maxOutputTokens: 2048,
      });
  }
}

// ─── MANUAL REACT LOOP ────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a knowledgeable assistant with access to a document knowledge base.

Guidelines:
- ALWAYS search the knowledge base before answering questions about documents
- If asked about a specific document, use search_specific_source
- If asked what documents are available, use list_available_sources
- Cite which document each piece of information comes from
- If information is not in the knowledge base, say so clearly`;

export async function queryAgent(question, history = [], provider, model) {
  const llm = buildLLM(provider, model);

  // Bind tools to the model — this enables tool_calling
  const llmWithTools = llm.bindTools(TOOLS_LIST);

  // Build initial message list
  const messages = [
    new SystemMessage(SYSTEM_PROMPT),
    ...history.flatMap((turn) => [
      new HumanMessage(turn.human),
      new AIMessage(turn.ai),
    ]),
    new HumanMessage(question),
  ];

  const toolsUsed = [];
  const MAX_ITERATIONS = 5; // prevent infinite loops

  for (let i = 0; i < MAX_ITERATIONS; i++) {
    const response = await llmWithTools.invoke(messages);
    messages.push(response);

    // No tool calls → model is done, return the answer
    if (!response.tool_calls || response.tool_calls.length === 0) {
      const answer =
        typeof response.content === "string"
          ? response.content
          : response.content[0]?.text || "";
      return { answer, toolsUsed };
    }

    // Execute each tool call and append results
    for (const toolCall of response.tool_calls) {
      const toolInstance = TOOLS[toolCall.name];
      if (!toolInstance) {
        messages.push(
          new ToolMessage({
            content: `Unknown tool: ${toolCall.name}`,
            tool_call_id: toolCall.id,
          }),
        );
        continue;
      }

      toolsUsed.push(toolCall.name);
      console.log(`[Agent] Calling tool: ${toolCall.name}`, toolCall.args);

      const result = await toolInstance.invoke(toolCall.args);

      messages.push(
        new ToolMessage({
          content: typeof result === "string" ? result : JSON.stringify(result),
          tool_call_id: toolCall.id,
          name: toolCall.name,
        }),
      );
    }
  }

  // Fallback if we hit the iteration limit
  return {
    answer:
      "I was unable to complete the research in the allowed steps. Please try a more specific question.",
    toolsUsed,
  };
}
