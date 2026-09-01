import type { EmbeddingClient } from "../rag-eval-v2/openai-embeddings.js";
import { OpenAIEmbeddingClient, cosineSimilarity } from "../rag-eval-v2/openai-embeddings.js";
import type { CodeQualityNormativeRule, CodeQualityScenario } from "./contracts.js";

/**
 * The retrieval arm answers a question the two-arm design cannot: whether the
 * treatment's advantage comes from holding the policy at all, which any
 * retrieval stack could supply, or from the governance workflow around it.
 *
 * The corpus therefore holds every scenario's rules, not just the one under
 * test, so retrieval has to find the right policy among distractors exactly as
 * a production RAG stack would. A corpus containing only the correct rules
 * would be an oracle, not a baseline.
 */
export interface RetrievedRule {
  readonly ruleId: string;
  readonly scenarioId: string;
  readonly text: string;
  readonly score: number;
}

export interface RagCorpusDocument {
  readonly ruleId: string;
  readonly scenarioId: string;
  readonly text: string;
}

export const defaultRetrievalCount = 5;

export function buildRagCorpus(
  scenarios: readonly CodeQualityScenario[],
): readonly RagCorpusDocument[] {
  return scenarios.flatMap((scenario) =>
    scenario.rules.map((rule) => ({
      ruleId: rule.recordId,
      scenarioId: scenario.scenarioId,
      text: ruleText(rule),
    })),
  );
}

export function ruleText(rule: CodeQualityNormativeRule): string {
  const body =
    rule.kind === "domain_term"
      ? `Domain Term ${rule.term}: ${rule.definition}`
      : rule.kind === "invariant"
        ? `Invariant: ${rule.statement}`
        : `Decision: ${rule.statement}`;
  return `${body}\nEvidence: ${rule.evidenceText}`;
}

export function retrievalQuery(scenario: CodeQualityScenario): string {
  return `${scenario.intent}\n${scenario.publicPrompt}`;
}

export interface RagRetriever {
  retrieve(scenario: CodeQualityScenario, count?: number): Promise<readonly RetrievedRule[]>;
}

/**
 * Embeds the corpus once and reuses it for every scenario and repetition, which
 * keeps the arm deterministic for a fixed corpus and keeps embedding cost to a
 * single pass.
 */
export class EmbeddingRagRetriever implements RagRetriever {
  private documentVectors?: Promise<readonly (readonly number[])[]>;

  constructor(
    private readonly corpus: readonly RagCorpusDocument[],
    private readonly client: EmbeddingClient = defaultEmbeddingClient(),
  ) {}

  async retrieve(
    scenario: CodeQualityScenario,
    count = defaultRetrievalCount,
  ): Promise<readonly RetrievedRule[]> {
    if (this.corpus.length === 0) return [];
    this.documentVectors ??= this.client
      .embed(
        this.corpus.map((document) => ({ id: document.ruleId, text: document.text })),
        "RETRIEVAL_DOCUMENT",
      )
      .then((outputs) => outputs.map((output) => output.values));
    const documents = await this.documentVectors;
    const query = await this.client.embed(
      [{ id: `query:${scenario.scenarioId}`, text: retrievalQuery(scenario) }],
      "RETRIEVAL_QUERY",
    );
    const queryVector = query[0]?.values;
    if (!queryVector) throw new Error("The retrieval query produced no embedding");
    return this.corpus
      .map((document, index) => {
        const vector = documents[index];
        if (!vector) throw new Error(`Missing embedding for ${document.ruleId}`);
        return {
          ruleId: document.ruleId,
          scenarioId: document.scenarioId,
          text: document.text,
          score: cosineSimilarity(queryVector, vector),
        };
      })
      .sort((left, right) => right.score - left.score || left.ruleId.localeCompare(right.ruleId))
      .slice(0, count);
  }
}

function defaultEmbeddingClient(): EmbeddingClient {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey?.trim()) {
    throw new Error("The retrieval arm requires OPENAI_API_KEY for corpus embeddings");
  }
  return new OpenAIEmbeddingClient({ apiKey });
}

export function renderRetrievedContext(retrieved: readonly RetrievedRule[]): string {
  if (retrieved.length === 0) return "No documentation was retrieved.";
  return retrieved.map((rule, index) => `[${index + 1}] ${rule.ruleId}\n${rule.text}`).join("\n\n");
}
