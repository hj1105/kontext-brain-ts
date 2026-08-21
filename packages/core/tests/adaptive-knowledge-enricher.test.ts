import { describe, expect, it } from "vitest";
import { AdaptiveKnowledgeEnricher, type LLMAdapter, type ResourceSnapshot } from "../src/index.js";

describe("AdaptiveKnowledgeEnricher", () => {
  it("selects capabilities from source text and dispatches only the selected extractors", async () => {
    const llm = new RecordingLlm([
      selection([
        "identity-resolution",
        "event-extraction",
        "temporal-relations",
        "causal-relations",
        "cross-chunk-consolidation",
      ]),
      extraction({
        entities: [
          entity("captain-vale", "Captain Vale", "person", [
            ["chapter-1", "Captain Vale"],
            ["chapter-2", "he"],
          ]),
          entity("storm-event", "The storm", "event", [["chapter-1", "the storm"]]),
          entity("departure-event", "Vale leaves the harbor", "event", [
            ["chapter-2", "he left the harbor at dawn"],
          ]),
        ],
        claims: [
          claim(
            "departure-event",
            "has_participant",
            { kind: "entity", entity_id: "captain-vale" },
            [["chapter-2", "he left the harbor at dawn"]],
          ),
          claim("storm-event", "causes", { kind: "entity", entity_id: "departure-event" }, [
            ["chapter-1", "the storm destroy the harbor"],
            ["chapter-2", "Because of the destruction"],
          ]),
          claim("storm-event", "before", { kind: "entity", entity_id: "departure-event" }, [
            ["chapter-1", "the storm destroy the harbor"],
            ["chapter-2", "left the harbor at dawn"],
          ]),
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
    expect(result.hypothesisCount).toBe(0);
    expect(result.validationFailureCount).toBe(0);
    expect(result.snapshot.entities).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "Captain Vale",
          scope: "resource",
          mentionChunkIds: ["chapter-1", "chapter-2"],
        }),
        expect.objectContaining({ name: "Vale leaves the harbor", type: "event" }),
      ]),
    );
    expect(result.snapshot.facts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          predicate: "causes",
          evidenceChunkIds: ["chapter-1", "chapter-2"],
        }),
      ]),
    );
    expect(llm.systemPrompts).toHaveLength(2);
    expect(llm.systemPrompts[0]).toContain("Select extraction capabilities");
    expect(llm.systemPrompts[1]).toContain("causal-relations");
    expect(llm.systemPrompts[0]).not.toMatch(/novel|medical/i);
    expect(llm.systemPrompts[1]).not.toMatch(/novel|medical/i);
  });

  it("rejects output for a capability that the source-driven selector did not enable", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({
        entities: [entity("departure", "Departure", "event", [["chunk-0", "departed"]])],
        claims: [],
      }),
    ]);

    await expect(
      new AdaptiveKnowledgeEnricher(llm, { maxExtractionAttempts: 1 }).enrich(
        snapshot("resource-a", [["chunk-0", "The train departed."]]),
      ),
    ).rejects.toThrow("event-extraction capability");
    expect(llm.systemPrompts[1]).not.toContain("temporal-relations");
    expect(llm.systemPrompts[1]).not.toContain("causal-relations");
  });

  it("reselects capabilities after a structurally invalid extraction", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({
        entities: [entity("departure", "Departure", "event", [["chunk-0", "departed"]])],
        claims: [],
      }),
      selection([]),
      extraction({
        entities: [entity("departure", "Departure", "event", [["chunk-0", "departed"]])],
        claims: [],
      }),
    ]);

    const result = await new AdaptiveKnowledgeEnricher(llm, {
      maxExtractionAttempts: 2,
    }).enrich(snapshot("resource-a", [["chunk-0", "The train departed."]]));

    expect(result.capabilities).toEqual(["event-extraction"]);
    expect(result.snapshot.entities).toHaveLength(1);
    expect(llm.systemPrompts).toHaveLength(4);
    expect(llm.queries[2]).toContain("Previous extraction failed validation");
  });

  it("uses a distinct repair prompt when the same validation error repeats", async () => {
    const invalid = extraction({
      entities: [entity("alex", "Alex", "person", [["chunk-0", "Alex"]])],
      claims: [
        claim("alex", "related_to", { kind: "entity", entity_id: "missing" }, [
          ["chunk-0", "Alex returned"],
        ]),
      ],
    });
    const llm = new RecordingLlm([
      selection([]),
      invalid,
      selection([]),
      invalid,
      selection([]),
      extraction({
        entities: [entity("alex", "Alex", "person", [["chunk-0", "Alex"]])],
        claims: [],
      }),
    ]);

    await new AdaptiveKnowledgeEnricher(llm, { maxExtractionAttempts: 3 }).enrich(
      snapshot("resource-a", [["chunk-0", "Alex returned."]]),
    );

    expect(llm.queries[2]).toContain("Repair attempt 2 of 3");
    expect(llm.queries[4]).toContain("Repair attempt 3 of 3");
    expect(llm.queries[2]).toContain("add the referenced Entity with an exact source Mention");
    expect(llm.queries[4]).toContain("return empty entities and claims arrays");
    expect(llm.queries[2]).not.toBe(llm.queries[4]);
  });

  it("withholds an invalid window only when empty-window policy is explicit", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({
        entities: [entity("alex", "Alex", "person", [["chunk-0", "invented quote"]])],
        claims: [],
      }),
    ]);

    const result = await new AdaptiveKnowledgeEnricher(llm, {
      maxExtractionAttempts: 1,
      validationFailurePolicy: "empty-window",
    }).enrich(snapshot("resource-a", [["chunk-0", "Alex returned."]]));

    expect(result.snapshot.entities).toEqual([]);
    expect(result.snapshot.facts).toEqual([]);
    expect(result.processedWindows).toBe(1);
    expect(result.validationFailureCount).toBe(1);
  });

  it("withholds inferred Claims as Hypotheses instead of activating Facts", async () => {
    const llm = new RecordingLlm([
      selection(["causal-relations"]),
      extraction({
        entities: [
          entity("rain", "Rain", "concept", [["chunk-0", "It rained"]]),
          entity("delay", "Delay", "concept", [["chunk-0", "the meeting started late"]]),
        ],
        claims: [
          claim(
            "rain",
            "causes",
            { kind: "entity", entity_id: "delay" },
            [["chunk-0", "It rained and the meeting started late"]],
            "inferred",
          ),
        ],
      }),
    ]);

    const result = await new AdaptiveKnowledgeEnricher(llm).enrich(
      snapshot("resource-a", [["chunk-0", "It rained and the meeting started late."]]),
    );

    expect(result.snapshot.facts).toEqual([]);
    expect(result.hypothesisCount).toBe(1);
  });

  it("rejects normalized-empty identifiers instead of returning a partial graph", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({
        entities: [entity("---", "Alex", "person", [["chunk-0", "Alex"]])],
        claims: [],
      }),
    ]);

    await expect(
      new AdaptiveKnowledgeEnricher(llm, { maxExtractionAttempts: 1 }).enrich(
        snapshot("resource-a", [["chunk-0", "Alex."]]),
      ),
    ).rejects.toThrow("normalizes to an empty identifier");
  });

  it("rejects citations to chunks omitted by the character budget", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({
        entities: [entity("hidden", "Hidden", "concept", [["chunk-1", "Hidden"]])],
        claims: [],
      }),
    ]);
    const enricher = new AdaptiveKnowledgeEnricher(llm, {
      chunksPerWindow: 2,
      overlapChunks: 0,
      maxWindowCharacters: 75,
      maxExtractionAttempts: 1,
    });

    await expect(
      enricher.enrich(
        snapshot("resource-a", [
          ["chunk-0", "Visible text consumes the available character budget."],
          ["chunk-1", "Hidden text must not be citeable."],
        ]),
      ),
    ).rejects.toThrow("unknown chunks: chunk-1");
    expect(llm.contexts.every((context) => !context.includes("chunk-1"))).toBe(true);
  });

  it("rejects quotes from the unseen suffix of a truncated chunk", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({
        entities: [entity("suffix", "Suffix", "concept", [["chunk-0", "UNSEEN_SUFFIX"]])],
        claims: [],
      }),
    ]);
    const enricher = new AdaptiveKnowledgeEnricher(llm, {
      chunksPerWindow: 1,
      overlapChunks: 0,
      maxWindowCharacters: 70,
      maxExtractionAttempts: 1,
    });

    await expect(
      enricher.enrich(
        snapshot("resource-a", [
          ["chunk-0", "Visible prefix is followed much later by UNSEEN_SUFFIX."],
        ]),
      ),
    ).rejects.toThrow("quote is not present");
    expect(llm.contexts.every((context) => !context.includes("UNSEEN_SUFFIX"))).toBe(true);
  });

  it("requires exact source quotes for every Mention and explicit Claim", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({
        entities: [entity("alex", "Alex", "person", [["chunk-0", "invented quote"]])],
        claims: [],
      }),
    ]);

    await expect(
      new AdaptiveKnowledgeEnricher(llm, { maxExtractionAttempts: 1 }).enrich(
        snapshot("resource-a", [["chunk-0", "Alex."]]),
      ),
    ).rejects.toThrow("quote is not present");
  });

  it("generates stable semantic Fact keys despite changing model-local entity ids", async () => {
    const first = new AdaptiveKnowledgeEnricher(
      new RecordingLlm([
        selection([]),
        extraction({
          entities: [entity("alex-1", "Alex", "person", [["chunk-0", "Alex"]])],
          claims: [
            claim("alex-1", "has_attribute", { kind: "literal", value: "Owner" }, [
              ["chunk-0", "Alex is Owner"],
            ]),
          ],
        }),
      ]),
    );
    const second = new AdaptiveKnowledgeEnricher(
      new RecordingLlm([
        selection([]),
        extraction({
          entities: [entity("person-a", "Alex", "person", [["chunk-0", "Alex"]])],
          claims: [
            claim("person-a", "has_attribute", { kind: "literal", value: "owner" }, [
              ["chunk-0", "Alex is Owner"],
            ]),
          ],
        }),
      ]),
    );
    const input = snapshot("resource-a", [["chunk-0", "Alex is Owner."]]);

    const left = await first.enrich(input);
    const right = await second.enrich(input);

    expect(left.snapshot.entities?.[0]?.entityId).toBe(right.snapshot.entities?.[0]?.entityId);
    expect(left.snapshot.facts?.[0]?.factKey).toBe(right.snapshot.facts?.[0]?.factKey);
  });

  it("keeps identical semantic Facts isolated by Resource", async () => {
    const responses = () => [
      selection([]),
      extraction({
        entities: [entity("alex", "Alex", "person", [["chunk-0", "Alex"]])],
        claims: [
          claim("alex", "has_attribute", { kind: "literal", value: "owner" }, [
            ["chunk-0", "Alex is owner"],
          ]),
        ],
      }),
    ];
    const left = await new AdaptiveKnowledgeEnricher(new RecordingLlm(responses())).enrich(
      snapshot("resource-a", [["chunk-0", "Alex is owner."]]),
    );
    const right = await new AdaptiveKnowledgeEnricher(new RecordingLlm(responses())).enrich(
      snapshot("resource-b", [["chunk-0", "Alex is owner."]]),
    );

    expect(left.snapshot.facts?.[0]?.factKey).not.toBe(right.snapshot.facts?.[0]?.factKey);
  });
});

class RecordingLlm implements LLMAdapter {
  readonly systemPrompts: string[] = [];
  readonly contexts: string[] = [];
  readonly queries: string[] = [];

  constructor(private readonly responses: readonly string[]) {}

  async complete(systemPrompt: string, context: string, query: string): Promise<string> {
    this.systemPrompts.push(systemPrompt);
    this.contexts.push(context);
    this.queries.push(query);
    const response = this.responses[this.systemPrompts.length - 1];
    if (!response) throw new Error("Missing fake LLM response");
    return response;
  }
}

function selection(capabilities: readonly string[]): string {
  return JSON.stringify({ capabilities });
}

function extraction(value: {
  readonly entities: readonly unknown[];
  readonly claims: readonly unknown[];
}): string {
  return JSON.stringify(value);
}

function entity(
  id: string,
  name: string,
  type: string,
  mentions: readonly (readonly [chunkId: string, quote: string])[],
) {
  return {
    id,
    name,
    type,
    mentions: mentions.map(([chunk_id, quote]) => ({ chunk_id, quote })),
  };
}

function claim(
  subject_id: string,
  predicate: string,
  object:
    | { readonly kind: "entity"; readonly entity_id: string }
    | { readonly kind: "literal"; readonly value: string | number | boolean },
  evidence: readonly (readonly [chunkId: string, quote: string])[],
  support: "explicit" | "inferred" = "explicit",
) {
  return {
    subject_id,
    predicate,
    object,
    evidence: evidence.map(([chunk_id, quote]) => ({ chunk_id, quote })),
    support,
    single_value: false,
  };
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
