import type { EmbeddingClient } from "../../rag-eval-v2/openai-embeddings.js";
import { OpenAIEmbeddingClient, cosineSimilarity } from "../../rag-eval-v2/openai-embeddings.js";
import {
  type KnowledgeDocument,
  allDocuments,
  governingDocumentIds,
  publicIssue,
} from "./documents.js";
import { subsystems } from "./generator.js";

/**
 * A fair retrieval control searches the source documents maintainers actually
 * wrote, not the compact normative records extracted for Kontext. One query per
 * visible subsystem prevents a deliberately vague issue from turning the RAG
 * arm into a strawman.
 */
export const perSubsystemRetrievalCount = 3;

export interface LargeScaleRetrieval {
  readonly documents: readonly KnowledgeDocument[];
  readonly governingRetrieved: number;
  readonly governingTotal: number;
}

export async function retrieveLargeScaleContext(
  client: EmbeddingClient = defaultClient(),
  perSubsystem = perSubsystemRetrievalCount,
): Promise<LargeScaleRetrieval> {
  const corpus = allDocuments();
  const documentVectors = await client.embed(
    corpus.map((document) => ({
      id: document.documentId,
      title: document.title,
      text: `${document.title}\n${document.body}`,
    })),
    "RETRIEVAL_DOCUMENT",
  );
  const queries = await client.embed(
    subsystems.map((subsystem) => ({
      id: subsystem.name,
      text: `${publicIssue.title}\nRetry delay policy for the ${subsystem.name} subsystem.\n${publicIssue.body}`,
    })),
    "RETRIEVAL_QUERY",
  );

  const picked = new Map<string, KnowledgeDocument>();
  for (const query of queries) {
    const ranked = corpus
      .map((document, index) => {
        const vector = documentVectors[index]?.values;
        if (!vector) throw new Error(`Missing embedding for ${document.documentId}`);
        return { document, score: cosineSimilarity(query.values, vector) };
      })
      .sort(
        (left, right) =>
          right.score - left.score ||
          left.document.documentId.localeCompare(right.document.documentId),
      );
    for (const entry of ranked.slice(0, perSubsystem)) {
      picked.set(entry.document.documentId, entry.document);
    }
  }

  const selected = [...picked.values()].sort((left, right) =>
    left.documentId.localeCompare(right.documentId),
  );
  const governingIds = new Set(governingDocumentIds());
  return {
    documents: selected,
    governingRetrieved: selected.filter((document) => governingIds.has(document.documentId)).length,
    governingTotal: governingIds.size,
  };
}

export function renderLargeScaleContext(retrieval: LargeScaleRetrieval): string {
  if (retrieval.documents.length === 0) return "No documentation was retrieved.";
  return retrieval.documents
    .map(
      (document, index) =>
        `[${index + 1}] ${document.documentId}: ${document.title}\n${document.body}`,
    )
    .join("\n\n");
}

function defaultClient(): EmbeddingClient {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey?.trim()) {
    throw new Error("The retrieval arm requires OPENAI_API_KEY for corpus embeddings");
  }
  return new OpenAIEmbeddingClient({ apiKey });
}
