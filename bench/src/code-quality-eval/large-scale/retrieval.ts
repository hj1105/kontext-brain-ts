import type { EmbeddingClient } from "../../rag-eval-v2/openai-embeddings.js";
import { OpenAIEmbeddingClient, cosineSimilarity } from "../../rag-eval-v2/openai-embeddings.js";
import { governedSubsystem, subsystems } from "./generator.js";
import { type LargeScaleRule, allRules, retrievalQueryText } from "./rules.js";

/**
 * Retrieval must be a fair control, neither an oracle nor a strawman.
 *
 * A single query for "the current retry policy" returns nothing useful: at 126
 * documents recall@5 was 0/3, because every subsystem's retry decision reads
 * alike and the governing one does not mention the word in the prompt. That
 * would understate retrieval as badly as the 30-document corpus overstated it.
 *
 * A competent integration issues one query per subsystem it can see in the
 * repository and unions the results, which recovers the governing decision and
 * its invariant while also surfacing the sibling decisions. The model then has
 * to decide which applies, which is the real question.
 */
export const perSubsystemRetrievalCount = 3;

export interface LargeScaleRetrieval {
  readonly rules: readonly LargeScaleRule[];
  readonly governingRetrieved: number;
  readonly governingTotal: number;
}

export async function retrieveLargeScaleContext(
  client: EmbeddingClient = defaultClient(),
  perSubsystem = perSubsystemRetrievalCount,
): Promise<LargeScaleRetrieval> {
  const rules = allRules();
  const documents = await client.embed(
    rules.map((rule) => ({ id: rule.recordId, text: `${rule.text}\n${rule.evidenceText}` })),
    "RETRIEVAL_DOCUMENT",
  );
  const queries = await client.embed(
    subsystems.map((subsystem) => ({
      id: subsystem.name,
      text: `Retry delay policy for the ${subsystem.name} subsystem. ${retrievalQueryText}`,
    })),
    "RETRIEVAL_QUERY",
  );

  const picked = new Map<string, LargeScaleRule>();
  for (const query of queries) {
    const ranked = rules
      .map((rule, index) => {
        const vector = documents[index]?.values;
        if (!vector) throw new Error(`Missing embedding for ${rule.recordId}`);
        return { rule, score: cosineSimilarity(query.values, vector) };
      })
      .sort(
        (left, right) =>
          right.score - left.score || left.rule.recordId.localeCompare(right.rule.recordId),
      );
    for (const entry of ranked.slice(0, perSubsystem)) picked.set(entry.rule.recordId, entry.rule);
  }

  const selected = [...picked.values()];
  return {
    rules: selected,
    governingRetrieved: selected.filter((rule) => rule.subsystem === governedSubsystem).length,
    governingTotal: rules.filter((rule) => rule.subsystem === governedSubsystem).length,
  };
}

export function renderLargeScaleContext(retrieval: LargeScaleRetrieval): string {
  if (retrieval.rules.length === 0) return "No documentation was retrieved.";
  return retrieval.rules
    .map((rule, index) => `[${index + 1}] ${rule.recordId}\n${rule.text}\n${rule.evidenceText}`)
    .join("\n\n");
}

function defaultClient(): EmbeddingClient {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey?.trim()) {
    throw new Error("The retrieval arm requires OPENAI_API_KEY for corpus embeddings");
  }
  return new OpenAIEmbeddingClient({ apiKey });
}
