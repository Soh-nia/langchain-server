/**
 * rag.js
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * LANGCHAIN RAG PIPELINE
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * This file rebuilds your manual RAG pipeline using LangChain abstractions.
 * Compare each section to what you wrote in RagPanel.jsx to see what
 * LangChain handles for you.
 *
 * WHAT LANGCHAIN REPLACES VS YOUR MANUAL CODE
 * ─────────────────────────────────────────────
 *
 * Manual code (RagPanel.jsx)         LangChain equivalent (this file)
 * ──────────────────────────────     ──────────────────────────────────────
 * chunkText() — manual loop     →    RecursiveCharacterTextSplitter
 * embedTexts() — fetch loop     →    GoogleGenerativeAIEmbeddings
 * supabase.from().insert()      →    SupabaseVectorStore.addDocuments()
 * supabase.rpc('match_docs')    →    vectorStore.asRetriever()
 * ragGenerate() — fetch + build →    createRetrievalChain()
 * No memory support             →    RunnableWithMessageHistory
 * Single document only          →    Multiple documents, filtered by source
 *
 * KEY LANGCHAIN CONCEPTS USED HERE
 * ──────────────────────────────────
 * 1. Document       — { pageContent: string, metadata: {} }
 *                     LangChain's universal unit of text + context
 *
 * 2. TextSplitter   — Splits Documents into chunks. RecursiveCharacterTextSplitter
 *                     tries paragraph → sentence → word → character splits,
 *                     preserving semantic coherence better than fixed char windows.
 *
 * 3. Embeddings     — GoogleGenerativeAIEmbeddings wraps the Gemini embedding API.
 *                     Implements .embedDocuments(texts[]) and .embedQuery(text).
 *                     Both are called automatically by SupabaseVectorStore.
 *
 * 4. VectorStore    — SupabaseVectorStore stores and retrieves Document vectors.
 *                     .addDocuments() embeds + inserts in one call.
 *                     .asRetriever() returns a Retriever that wraps similarity search.
 *
 * 5. Retriever      — A Runnable that takes a string query and returns Document[].
 *                     Plugs directly into chains.
 *
 * 6. Chain (LCEL)   — prompt.pipe(model).pipe(parser) — the | operator of LangChain.
 *                     createRetrievalChain() assembles the full RAG chain.
 *
 * 7. Memory         — RunnableWithMessageHistory wraps any chain to add conversation
 *                     history. The chain sees previous Q&A turns automatically.
 *
 * 8. PromptTemplate — ChatPromptTemplate.fromMessages() builds prompts with slots
 *                     ({context}, {input}, {chat_history}) filled at runtime.
 */

import { createClient } from "@supabase/supabase-js";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";

// ─── INITIALISE SHARED SERVICES ───────────────────────────────────────────────
//
// These are initialised once and reused across all requests.
// In production you'd inject these via dependency injection,
// but for a learning server module-level singletons are fine.

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY, // note: SERVICE_ROLE key for server-side
);

// GoogleGenerativeAIEmbeddings handles:
//  - Calling the Gemini embedding API
//  - Batching texts for you (one call per text internally)
//  - taskType.RETRIEVAL_DOCUMENT for storage
//  - Passing outputDimensionality to match your table

class GeminiEmbeddings {
  constructor(taskType = "RETRIEVAL_DOCUMENT") {
    this.taskType = taskType;
  }

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
            taskType: this.taskType,
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
    return data.embedding.values;
  }
}

const embeddings = new GeminiEmbeddings("RETRIEVAL_DOCUMENT");

// SupabaseVectorStore wraps your documents table.
// It knows how to call match_documents via the queryName parameter.
const vectorStore = new SupabaseVectorStore(embeddings, {
  client: supabase,
  tableName: "documents", // your Supabase table
  queryName: "match_documents", // your RPC function name
});

// In-memory session histories: { sessionId: ChatMessageHistory }
// In production, persist these in Supabase or Redis
const sessionHistories = new Map();

function getSessionHistory(sessionId) {
  if (!sessionHistories.has(sessionId)) {
    sessionHistories.set(sessionId, new ChatMessageHistory());
  }
  return sessionHistories.get(sessionId);
}

// Build a chat model from a provider name
// This is the LangChain equivalent of your provider routing in useChat.js
function buildChatModel(provider, model) {
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
    case "gemini":
    default:
      return new ChatGoogleGenerativeAI({
        apiKey: process.env.GEMINI_API_KEY,
        model: model || "gemini-2.5-flash",
        maxOutputTokens: 2048,
      });
  }
}

// ─── 1. INGESTION ─────────────────────────────────────────────────────────────
//
// Replaces your manual chunkText() + embedTexts() + supabase.insert() loop.
//
// RecursiveCharacterTextSplitter tries to split on:
//   '\n\n' (paragraphs) → '\n' (lines) → '. ' (sentences) → ' ' (words) → ''
// This preserves semantic coherence. Your manual splitter split on fixed
// character counts regardless of where sentences ended.
//
// SupabaseVectorStore.addDocuments():
//   1. Calls embeddings.embedDocuments(chunks.map(c => c.pageContent))
//   2. Inserts rows: { content, embedding, metadata } into your table
//   All in one call, batched properly.
//
// The metadata object is crucial for multi-document knowledge bases:
// it lets you filter retrieval to specific sources.

export async function ingestDocument(text, metadata = {}) {
  // Step 1: Split into chunks
  // This is smarter than your manual chunkText() because it respects
  // natural text boundaries (paragraphs, sentences) before falling back
  // to character-level splitting
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 800,
    chunkOverlap: 150,
    separators: ["\n\n", "\n", ". ", " ", ""],
  });

  // createDocuments returns Document[] — each has { pageContent, metadata }
  const docs = await splitter.createDocuments(
    [text],
    [metadata], // metadata is attached to every chunk from this document
  );

  console.log(`[RAG] Split into ${docs.length} chunks`);

  // Step 2: Embed + store — LangChain handles the loop internally
  await vectorStore.addDocuments(docs);

  console.log(`[RAG] Stored ${docs.length} chunks in Supabase`);
  return { chunksStored: docs.length };
}

// ─── 2. RAG CHAIN WITH MEMORY ─────────────────────────────────────────────────
//
// This builds a full conversational RAG chain — something that was not possible
// in your manual RagPanel.jsx without significant extra code.
//
// THE CHAIN STRUCTURE (read top to bottom):
//
// User question
//       ↓
// historyAwareRetriever   ← reformulates question using chat history
//       ↓                   (handles: "what about that clause?" → "what about the penalty clause?")
// Retrieved documents
//       ↓
// combineDocsChain         ← stuffs docs into the prompt context
//       ↓
// Chat model
//       ↓
// Answer
//       ↓
// RunnableWithMessageHistory ← saves Q&A to history for next turn
//
// WHY historyAwareRetriever?
// Without it, if the user asks "what about section 3?" as a follow-up,
// the retriever searches for "what about section 3?" — which won't match anything.
// historyAwareRetriever first asks the LLM to reformulate the question
// using history context, turning it into "what does section 3 say about payment terms?"

export async function buildRagChain(provider, model, sourceFilter = null) {
  const llm = buildChatModel(provider, model);

  // The retriever — wraps vectorStore.similaritySearch()
  // sourceFilter lets you query only documents from a specific uploaded file
  const baseRetriever = vectorStore.asRetriever({
    k: 5, // retrieve top 5 most similar chunks
    filter: sourceFilter ? { source: sourceFilter } : undefined,
  });

  // Prompt to reformulate a follow-up question into a standalone question
  // using chat history — so the retriever can search effectively
  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.`,
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
  ]);

  // Retriever that uses history to reformulate the search query
  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm,
    retriever: baseRetriever,
    rephrasePrompt: historyAwarePrompt,
  });

  // The main QA prompt — context comes from retrieved docs, input is the question
  const qaPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a precise document assistant for a knowledge base chatbot. Answer questions using ONLY the provided context from the uploaded documents.

Rules:
- If the answer is in the context, answer clearly and mention which source it came from
- If the answer is NOT in the context, say: "This information is not in the uploaded documents"
- Never fabricate information not present in the context
- Be concise but complete

Context:
{context}`,
    ],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
  ]);

  // createStuffDocumentsChain: takes retrieved docs and "stuffs" them into
  // the {context} slot of the prompt as a formatted string
  const combineDocsChain = await createStuffDocumentsChain({
    llm,
    prompt: qaPrompt,
    outputParser: new StringOutputParser(),
  });

  // createRetrievalChain: wires retriever → combineDocsChain into one Runnable
  // Input: { input: string, chat_history: Message[] }
  // Output: { answer: string, context: Document[] }
  const ragChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain,
  });

  // Wrap with memory so conversation history is automatically managed
  // The chain now remembers what was said in previous turns
  const chainWithMemory = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory: getSessionHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    outputMessagesKey: "answer",
  });

  return chainWithMemory;
}

// ─── 3. QUERY ─────────────────────────────────────────────────────────────────
//
// Runs one question through the RAG chain and returns the answer + sources.
// The sessionId is used to maintain separate conversation histories per user.
//
// sourceFilter: if provided, only searches documents from that source file.
// Pass null to search across ALL uploaded documents (the knowledge base).

export async function queryRag(
  question,
  sessionId,
  provider,
  model,
  sourceFilter = null,
) {
  const chain = await buildRagChain(provider, model, sourceFilter);

  const result = await chain.invoke(
    { input: question },
    { configurable: { sessionId } },
  );

  // Extract unique source filenames from retrieved documents
  const sources = [
    ...new Set(
      (result.context || []).map(
        (doc) => doc.metadata?.source || doc.metadata?.fileName || "Unknown",
      ),
    ),
  ];

  return {
    answer: result.answer,
    sources,
    chunks: (result.context || []).map((doc) => ({
      content: doc.pageContent.slice(0, 200),
      source: doc.metadata?.source || "Unknown",
      similarity: doc.metadata?.similarity,
    })),
  };
}

// ─── 4. LIST SOURCES ──────────────────────────────────────────────────────────
// Returns a distinct list of all document sources in the knowledge base

export async function listSources() {
  const { data, error } = await supabase
    .from("documents")
    .select("metadata")
    .not("metadata->source", "is", null);

  if (error) throw new Error(`Supabase: ${error.message}`);

  const sources = [
    ...new Set(data.map((row) => row.metadata?.source).filter(Boolean)),
  ];
  return sources;
}

// ─── 5. CLEAR SESSION HISTORY ─────────────────────────────────────────────────
// Call this when the user starts a new chat session

export function clearSession(sessionId) {
  sessionHistories.delete(sessionId);
}
