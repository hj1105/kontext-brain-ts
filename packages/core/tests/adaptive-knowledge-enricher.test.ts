import { describe, expect, it } from "vitest";
import { AdaptiveKnowledgeEnricher, type LLMAdapter, type ResourceSnapshot } from "../src/index.js";

describe("AdaptiveKnowledgeEnricher", () => {
  it("selects narrative capabilities and builds resource-scoped event facts", async () => {
    const llm = new RecordingLlm([
      extraction({
        capabilities: [
          "identity-resolution",
          "event-extraction",
          "temporal-relations",
          "causal-relations",
          "cross-chunk-consolidation",
        ],
        entities: [
          {
            id: "captain-vale",
            name: "Captain Vale",
            type: "person",
            mention_chunk_ids: ["chapter-1", "chapter-2"],
          },
          {
            id: "storm-event",
            name: "The storm",
            type: "event",
            mention_chunk_ids: ["chapter-1"],
          },
          {
            id: "departure-event",
            name: "Vale leaves the harbor",
            type: "event",
            mention_chunk_ids: ["chapter-2"],
          },
        ],
        facts: [
          {
            subject_id: "departure-event",
            predicate: "has_participant",
            object: { kind: "entity", entity_id: "captain-vale" },
            evidence_chunk_ids: ["chapter-2"],
          },
          {
            subject_id: "storm-event",
            predicate: "causes",
            object: { kind: "entity", entity_id: "departure-event" },
            evidence_chunk_ids: ["chapter-1", "chapter-2"],
          },
          {
            subject_id: "storm-event",
            predicate: "before",
            object: { kind: "entity", entity_id: "departure-event" },
            evidence_chunk_ids: ["chapter-1", "chapter-2"],
          },
        ],
      }),
    ]);
    const result = await new AdaptiveKnowledgeEnricher(llm).enrich(
      snapshot("book-1", [
        ["chapter-1", "Captain Vale watched the storm destroy the harbor."],
        ["chapter-2", "Because of the destruction, he left the harbor at dawn."],
      ]),
    );

    expect(result.capabilities).toEqual([
      "causal-relations",
      "cross-chunk-consolidation",
      "event-extraction",
      "identity-resolution",
      "temporal-relations",
    ]);
    expect(result.processedWindows).toBe(1);
    expect(result.snapshot.entities).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          entityId: "captain-vale",
          scope: "resource",
          mentionChunkIds: ["chapter-1", "chapter-2"],
        }),
        expect.objectContaining({ entityId: "departure-event", type: "event" }),
      ]),
    );
    expect(result.snapshot.facts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          subject: { entityId: "storm-event", scope: "resource" },
          predicate: "causes",
          object: {
            kind: "entity",
            entity: { entityId: "departure-event", scope: "resource" },
          },
          evidenceChunkIds: ["chapter-1", "chapter-2"],
        }),
      ]),
    );
    expect(llm.systemPrompts[0]).not.toMatch(/novel|medical/i);
  });

  it("keeps identical names isolated by Resource when generating Fact keys", async () => {
    const response = extraction({
      capabilities: [],
      entities: [
        {
          id: "alex",
          name: "Alex",
          type: "person",
          mention_chunk_ids: ["chunk-0"],
        },
      ],
      facts: [
        {
          subject_id: "alex",
          predicate: "role",
          object: { kind: "literal", value: "owner" },
          evidence_chunk_ids: ["chunk-0"],
          single_value: true,
        },
      ],
    });
    const enricher = new AdaptiveKnowledgeEnricher(new RecordingLlm([response, response]));

    const left = await enricher.enrich(snapshot("resource-a", [["chunk-0", "Alex is owner."]]));
    const right = await enricher.enrich(snapshot("resource-b", [["chunk-0", "Alex is owner."]]));

    expect(left.snapshot.entities?.[0]?.scope).toBe("resource");
    expect(right.snapshot.entities?.[0]?.scope).toBe("resource");
    expect(left.snapshot.facts?.[0]?.factKey).not.toBe(right.snapshot.facts?.[0]?.factKey);
  });

  it("rejects unsupported chunk citations instead of returning a partial graph", async () => {
    const llm = new RecordingLlm([
      extraction({
        capabilities: ["event-extraction"],
        entities: [
          {
            id: "departure",
            name: "Departure",
            type: "event",
            mention_chunk_ids: ["invented-chunk"],
          },
        ],
        facts: [],
      }),
    ]);

    await expect(
      new AdaptiveKnowledgeEnricher(llm).enrich(
        snapshot("resource-a", [["chunk-0", "The train departed."]]),
      ),
    ).rejects.toThrow('Entity "departure" cites unknown chunks: invented-chunk');
  });
});

class RecordingLlm implements LLMAdapter {
  readonly systemPrompts: string[] = [];

  constructor(private readonly responses: readonly string[]) {}

  async complete(systemPrompt: string): Promise<string> {
    this.systemPrompts.push(systemPrompt);
    const response = this.responses[this.systemPrompts.length - 1];
    if (!response) throw new Error("Missing fake LLM response");
    return response;
  }
}

function extraction(value: {
  readonly capabilities: readonly string[];
  readonly entities: readonly unknown[];
  readonly facts: readonly unknown[];
}): string {
  return JSON.stringify(value);
}

function snapshot(
  externalId: string,
  chunks: readonly (readonly [id: string, text: string])[],
): ResourceSnapshot {
  return {
    organizationId: "acme",
    source: { connectorId: "test", externalId, type: "text" },
    title: externalId,
    contentHash: `${externalId}-hash`,
    body: chunks.map((chunk) => chunk[1]).join("\n"),
    acl: { organizationWide: true },
    chunks: chunks.map(([id, text], position) => ({
      id,
      text,
      position,
      contentHash: `${externalId}-${position}`,
    })),
  };
}
