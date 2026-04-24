/**
 * Baseline vector RAG — classic approach.
 * 1. Split each doc into chunks (~400 chars, overlap 50)
 * 2. Embed all chunks with Ollama nomic-embed-text
 * 3. At query time: embed query, cosine similarity, take top-K chunks
 * 4. Send chunks as context to chat LLM, get answer
 */

import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import type { BenchDoc } from "./corpus.js";

interface Chunk {
  docId: string;
  title: string;
  text: string;
  embedding: number[];
}

function chunkText(text: string, size = 400, overlap = 50): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size - overlap) {
    out.push(text.slice(i, i + size));
    if (i + size >= text.length) break;
  }
  return out;
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += (a[i] ?? 0) * (b[i] ?? 0);
    na += (a[i] ?? 0) ** 2;
    nb += (b[i] ?? 0) ** 2;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

export class BaselineRAG {
  private chunks: Chunk[] = [];
  private readonly embeddings: OllamaEmbeddings;
  private readonly chat: ChatOllama;

  constructor(
    baseUrl = "http://localhost:11434",
    chatModel = "qwen2.5:1.5b",
    embedModel = "nomic-embed-text",
  ) {
    this.embeddings = new OllamaEmbeddings({ baseUrl, model: embedModel });
    this.chat = new ChatOllama({
      baseUrl,
      model: chatModel,
      temperature: 0,
      numGpu: 0,
      numCtx: 2048,
      numPredict: 256,
    });
  }

  async index(corpus: BenchDoc[]): Promise<void> {
    const allTexts: Array<{ docId: string; title: string; text: string }> = [];
    for (const doc of corpus) {
      for (const chunk of chunkText(doc.body)) {
        allTexts.push({ docId: doc.id, title: doc.title, text: chunk });
      }
    }
    const vectors = await this.embeddings.embedDocuments(allTexts.map((x) => x.text));
    this.chunks = allTexts.map((x, i) => ({
      docId: x.docId,
      title: x.title,
      text: x.text,
      embedding: vectors[i] ?? [],
    }));
  }

  /** Returns top-K chunks by cosine similarity. */
  async retrieve(query: string, topK = 4): Promise<Chunk[]> {
    const qvec = await this.embeddings.embedQuery(query);
    return this.chunks
      .map((c) => ({ c, score: cosine(qvec, c.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((x) => x.c);
  }

  async query(question: string): Promise<{
    answer: string;
    retrievedDocIds: string[];
    contextChars: number;
  }> {
    const top = await this.retrieve(question, 4);
    const context = top.map((c) => `### ${c.title}\n${c.text}`).join("\n\n");
    const prompt = `You are a technical assistant. Answer the question based only on the provided context. Be concise.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;
    const res = await this.chat.invoke(prompt);
    const answer = typeof res.content === "string" ? res.content : JSON.stringify(res.content);
    return {
      answer,
      retrievedDocIds: Array.from(new Set(top.map((c) => c.docId))),
      contextChars: context.length,
    };
  }
}
