/**
 * server.js
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * EXPRESS API SERVER — wraps LangChain pipeline for the React frontend
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * WHY A SEPARATE SERVER?
 * ───────────────────────
 * LangChain packages are too large for a browser bundle. More importantly,
 * this is the CORRECT production pattern — API keys live on the server,
 * never in the browser. The React frontend calls this server, the server
 * calls the AI APIs with keys it holds securely.
 *
 * This is exactly the "backend proxy" pattern mentioned at the end of the
 * previous sessions. LangChain gives us the right reason to implement it.
 *
 * ENDPOINTS
 * ──────────
 * POST /api/ingest          ← upload + process a document
 * POST /api/query           ← RAG query with memory
 * POST /api/agent           ← agent query (multi-tool, multi-document)
 * GET  /api/sources         ← list all uploaded document names
 * DELETE /api/session/:id   ← clear conversation history
 * GET  /api/health          ← server health check
 */

import "dotenv/config";
import express from "express";
import cors from "cors";
import { ingestDocument, queryRag, listSources, clearSession } from "./rag.js";
import { queryAgent } from "./agent.js";

const app = express();
const PORT = process.env.PORT || 3001;

// ─── MIDDLEWARE ────────────────────────────────────────────────────────────────
app.use(express.json({ limit: "10mb" })); // large limit for document text
app.use(
  cors({
    origin: process.env.ALLOWED_ORIGIN || "http://localhost:3000",
    methods: ["GET", "POST", "DELETE"],
    allowedHeaders: ["Content-Type"],
  }),
);

// Converts raw API error messages into friendly user-facing messages
function friendlyError(message) {
  if (
    message.includes("429") ||
    message.includes("Too Many Requests") ||
    message.includes("quota")
  ) {
    return "Rate limit reached — Gemini free tier allows 20 requests/day. Wait a few hours or enable billing at aistudio.google.com to continue.";
  }
  if (message.includes("403") || message.includes("API key")) {
    return "Invalid API key. Check your .env file and restart the server.";
  }
  if (
    message.includes("503") ||
    message.includes("overloaded") ||
    message.includes("busy")
  ) {
    return "The AI model is busy right now. Wait a moment and try again.";
  }
  if (message.includes("PGRST") || message.includes("supabase")) {
    return "Database error. Check your Supabase credentials and that the match_documents function exists.";
  }
  return message;
}

// ─── ROUTES ───────────────────────────────────────────────────────────────────

// Health check — the React app calls this on startup to confirm server is running
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    message: "LangChain server running",
    supabase: !!process.env.SUPABASE_URL,
    gemini: !!process.env.GEMINI_API_KEY,
  });
});

// INGEST — chunk, embed, and store a document
// Body: { text: string, source: string (filename) }
app.post("/api/ingest", async (req, res) => {
  try {
    const { text, source } = req.body;
    if (!text?.trim())
      return res.status(400).json({ error: "text is required" });
    if (!source?.trim())
      return res.status(400).json({ error: "source filename is required" });

    console.log(`[Ingest] Processing: ${source} (${text.length} chars)`);

    const result = await ingestDocument(text, {
      source,
      uploadedAt: new Date().toISOString(),
    });

    res.json({
      success: true,
      source,
      chunksStored: result.chunksStored,
      message: `${result.chunksStored} chunks stored from "${source}"`,
    });
  } catch (err) {
    console.error("[Ingest error]", err.message);
    res.status(500).json({ error: err.message });
  }
});

// QUERY — RAG query with conversation memory
// Body: { question, sessionId, provider, model, sourceFilter (optional) }
app.post("/api/query", async (req, res) => {
  try {
    const { question, sessionId, provider, model, sourceFilter } = req.body;
    if (!question?.trim())
      return res.status(400).json({ error: "question is required" });

    const sid = sessionId || "default";
    console.log(`[Query] Session ${sid}: "${question}"`);

    const result = await queryRag(
      question,
      sid,
      provider,
      model,
      sourceFilter || null,
    );

    res.json(result);
  } catch (err) {
    console.error("[Query error]", err.message);
    res.status(500).json({ error: friendlyError(err.message) });
  }
});

// AGENT — multi-tool agent query (can search across documents, compare, summarise)
// Body: { question, history, provider, model }
app.post("/api/agent", async (req, res) => {
  try {
    const { question, history = [], provider, model } = req.body;
    if (!question?.trim())
      return res.status(400).json({ error: "question is required" });

    console.log(`[Agent] "${question}"`);

    const result = await queryAgent(question, history, provider, model);

    res.json(result);
  } catch (err) {
    console.error("[Agent error]", err.message);
    res.status(500).json({ error: friendlyError(err.message) });
  }
});

// SOURCES — list all uploaded document filenames
app.get("/api/sources", async (req, res) => {
  try {
    const sources = await listSources();
    res.json({ sources });
  } catch (err) {
    console.error("[Sources error]", err.message);
    res.status(500).json({ error: err.message });
  }
});

// CLEAR SESSION — delete conversation history for a session
app.delete("/api/session/:sessionId", (req, res) => {
  clearSession(req.params.sessionId);
  res.json({
    success: true,
    message: `Session ${req.params.sessionId} cleared`,
  });
});

// ─── START ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n LangChain server running on http://localhost:${PORT}`);
  console.log(
    ` Supabase: ${process.env.SUPABASE_URL ? "connected" : "MISSING"}`,
  );
  console.log(
    ` Gemini:   ${process.env.GEMINI_API_KEY ? "connected" : "MISSING"}`,
  );
  console.log(`\n Endpoints:`);
  console.log(`   POST /api/ingest    — upload a document`);
  console.log(`   POST /api/query     — RAG query with memory`);
  console.log(`   POST /api/agent     — agent with tools`);
  console.log(`   GET  /api/sources   — list documents\n`);
});
