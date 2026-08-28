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
      verification(3),
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
    expect(llm.systemPrompts).toHaveLength(3);
    expect(llm.systemPrompts[0]).toContain("Select extraction capabilities");
    expect(llm.systemPrompts[1]).toContain("causal-relations");
    expect(llm.systemPrompts[2]).toContain("Independently verify");
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

  it("rejects the whole enrichment when any window stays invalid", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({ entities: [], claims: [] }),
      selection([]),
      extraction({
        entities: [entity("alex", "Alex", "person", [["chunk-1", "invented quote"]])],
        claims: [],
      }),
    ]);

    await expect(
      new AdaptiveKnowledgeEnricher(llm, {
        chunksPerWindow: 1,
        overlapChunks: 0,
        concurrency: 1,
        maxExtractionAttempts: 1,
      }).enrich(
        snapshot("resource-a", [
          ["chunk-0", "A valid empty window."],
          ["chunk-1", "Alex returned."],
        ]),
      ),
    ).rejects.toThrow("failed validation");
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

  it("rejects an explicit Claim that an independent verifier cannot support", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({
        entities: [entity("alex", "Alex", "person", [["chunk-0", "Alex"]])],
        claims: [
          claim("alex", "has_attribute", { kind: "literal", value: "owner" }, [
            ["chunk-0", "The office is closed"],
          ]),
        ],
      }),
      verification(1, "unsupported"),
    ]);

    await expect(
      new AdaptiveKnowledgeEnricher(llm, { maxExtractionAttempts: 1 }).enrich(
        snapshot("resource-a", [["chunk-0", "Alex arrived. The office is closed."]]),
      ),
    ).rejects.toThrow("not independently verified as explicit");
  });

  it("resolves one Entity across overlapping extraction windows at Resource scope", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({
        entities: [entity("vale-a", "Captain Vale", "person", [["chunk-1", "Captain Vale"]])],
        claims: [],
      }),
      selection([]),
      extraction({
        entities: [entity("vale-b", "Vale", "person", [["chunk-1", "Captain Vale"]])],
        claims: [],
      }),
      selection([]),
      extraction({ entities: [], claims: [] }),
    ]);

    const result = await new AdaptiveKnowledgeEnricher(llm, {
      chunksPerWindow: 2,
      overlapChunks: 1,
      concurrency: 1,
    }).enrich(
      snapshot("resource-a", [
        ["chunk-0", "The harbor was quiet."],
        ["chunk-1", "Captain Vale arrived."],
        ["chunk-2", "The crew assembled."],
      ]),
    );

    expect(result.snapshot.entities).toHaveLength(1);
    expect(result.snapshot.entities?.[0]).toMatchObject({
      name: "Captain Vale",
      mentionChunkIds: ["chunk-1"],
    });
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
      maxWindowCharacters: 150,
      concurrency: 1,
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

  it("rejects an oversized chunk instead of silently extracting only its prefix", async () => {
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
    ).rejects.toThrow("exceeds maxWindowCharacters");
    expect(llm.contexts).toEqual([]);
  });

  it("processes every chunk when the character budget fits fewer chunks than chunksPerWindow", async () => {
    const llm = new RecordingLlm([
      selection([]),
      extraction({ entities: [], claims: [] }),
      selection([]),
      extraction({ entities: [], claims: [] }),
      selection([]),
      extraction({ entities: [], claims: [] }),
    ]);

    const result = await new AdaptiveKnowledgeEnricher(llm, {
      chunksPerWindow: 3,
      overlapChunks: 0,
      maxWindowCharacters: 120,
      concurrency: 1,
    }).enrich(
      snapshot("resource-a", [
        ["chunk-0", "First source chunk."],
        ["chunk-1", "Second source chunk."],
        ["chunk-2", "Third source chunk."],
      ]),
    );

    expect(result.processedWindows).toBe(3);
    for (const chunkId of ["chunk-0", "chunk-1", "chunk-2"]) {
      expect(llm.contexts.some((context) => context.includes(chunkId))).toBe(true);
    }
  });

  it("never places chunks from different effective ACLs in one extraction window", async () => {
    const llm = new RecordingLlm([
      selection(["identity-resolution", "cross-chunk-consolidation"]),
      extraction({
        entities: [
          entity("captain", "Captain Vale", "person", [
            ["public", "he"],
            ["private", "Captain Vale"],
          ]),
        ],
        claims: [],
      }),
    ]);

    await expect(
      new AdaptiveKnowledgeEnricher(llm, {
        chunksPerWindow: 2,
        overlapChunks: 0,
        concurrency: 1,
        maxExtractionAttempts: 1,
      }).enrich(
        snapshot("resource-a", [
          ["public", "Yesterday he arrived.", { organizationWide: true }],
          ["private", "Captain Vale arrived.", { subjectIds: ["admin"] }],
        ]),
      ),
    ).rejects.toThrow("unknown chunks: private");
    expect(
      llm.contexts.every((context) => !(context.includes("public") && context.includes("private"))),
    ).toBe(true);
  });

  it("treats reordered and duplicated ACL grants as the same visibility domain", async () => {
    const llm = new RecordingLlm([
      selection(["identity-resolution", "cross-chunk-consolidation"]),
      extraction({
        entities: [
          entity("captain", "Captain Vale", "person", [
            ["chunk-0", "Captain Vale"],
            ["chunk-1", "Vale"],
          ]),
        ],
        claims: [],
      }),
    ]);

    const result = await new AdaptiveKnowledgeEnricher(llm).enrich(
      snapshot("resource-a", [
        ["chunk-0", "Captain Vale arrived.", { subjectIds: ["alice", "bob", "alice"] }],
        ["chunk-1", "Vale departed.", { subjectIds: ["bob", "alice"] }],
      ]),
    );

    expect(result.processedWindows).toBe(1);
    expect(result.snapshot.entities?.[0]?.mentionChunkIds).toEqual(["chunk-0", "chunk-1"]);
  });

  it("rejects an Entity collision that would merge Mention ACL domains", async () => {
    const first = await new AdaptiveKnowledgeEnricher(
      new RecordingLlm([
        selection([]),
        extraction({
          entities: [entity("alex", "Alex", "person", [["public", "Alex"]])],
          claims: [],
        }),
      ]),
    ).enrich(snapshot("resource-a", [["public", "Alex arrived.", { organizationWide: true }]]));
    const existing = first.snapshot.entities?.[0];
    if (!existing) throw new Error("Expected an extracted Entity");

    const input = snapshot("resource-a", [
      ["public", "Alex arrived.", { organizationWide: true }],
      ["private", "", { subjectIds: ["admin"] }],
    ]);
    await expect(
      new AdaptiveKnowledgeEnricher(
        new RecordingLlm([
          selection([]),
          extraction({
            entities: [entity("alex", "Alex", "person", [["public", "Alex"]])],
            claims: [],
          }),
        ]),
      ).enrich({
        ...input,
        entities: [{ ...existing, mentionChunkIds: ["private"] }],
      }),
    ).rejects.toThrow(`Entity "${existing.entityId}" crosses visibility domains`);
  });

  it("rejects a Fact collision that would merge evidence ACL domains", async () => {
    const responses = () => [
      selection([]),
      extraction({
        entities: [entity("alex", "Alex", "person", [["public", "Alex"]])],
        claims: [
          claim("alex", "has_attribute", { kind: "literal", value: "owner" }, [
            ["public", "Alex is owner"],
          ]),
        ],
      }),
      verification(1),
    ];
    const first = await new AdaptiveKnowledgeEnricher(new RecordingLlm(responses())).enrich(
      snapshot("resource-a", [["public", "Alex is owner.", { organizationWide: true }]]),
    );
    const existing = first.snapshot.facts?.[0];
    if (!existing) throw new Error("Expected an extracted Fact");

    const input = snapshot("resource-a", [
      ["public", "Alex is owner.", { organizationWide: true }],
      ["private", "", { subjectIds: ["admin"] }],
    ]);
    await expect(
      new AdaptiveKnowledgeEnricher(new RecordingLlm(responses())).enrich({
        ...input,
        facts: [{ ...existing, evidenceChunkIds: ["private"] }],
      }),
    ).rejects.toThrow(`Fact "${existing.factKey}" crosses visibility domains`);
  });

  it("stops scheduling new windows after the first concurrent extraction fails", async () => {
    const llm = new FailFastLlm();

    await expect(
      new AdaptiveKnowledgeEnricher(llm, {
        chunksPerWindow: 1,
        overlapChunks: 0,
        concurrency: 2,
        maxExtractionAttempts: 1,
      }).enrich(
        snapshot("resource-a", [
          ["chunk-0", "Fail immediately."],
          ["chunk-1", "Already in flight."],
          ["chunk-2", "Must never be scheduled."],
        ]),
      ),
    ).rejects.toThrow("failed validation");
    expect(llm.contexts.some((context) => context.includes("chunk-2"))).toBe(false);
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

  it("generates stable semantic Fact keys despite changing local ids and display names", async () => {
    const first = new AdaptiveKnowledgeEnricher(
      new RecordingLlm([
        selection([]),
        extraction({
          entities: [entity("vale-1", "Captain Vale", "person", [["chunk-0", "Captain Vale"]])],
          claims: [
            claim("vale-1", "has_attribute", { kind: "literal", value: "Owner" }, [
              ["chunk-0", "Captain Vale is Owner"],
            ]),
          ],
        }),
        verification(1),
      ]),
    );
    const input = snapshot("resource-a", [["chunk-0", "Captain Vale is Owner."]]);
    const left = await first.enrich(input);
    const second = new AdaptiveKnowledgeEnricher(
      new RecordingLlm([
        selection([]),
        extraction({
          entities: [entity("person-a", "Vale", "person", [["chunk-0", "Vale"]])],
          claims: [
            claim("person-a", "has_attribute", { kind: "literal", value: "owner" }, [
              ["chunk-0", "Vale is Owner"],
            ]),
          ],
        }),
        verification(1),
      ]),
    );
    const changed = snapshot("resource-a", [["chunk-0", "Vale is Owner."]]);
    const right = await second.enrich(changed, left.snapshot.entities);

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
      verification(1),
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

class FailFastLlm implements LLMAdapter {
  readonly contexts: string[] = [];

  async complete(systemPrompt: string, context: string): Promise<string> {
    this.contexts.push(context);
    if (context.includes("chunk-0")) throw new Error("synthetic extraction failure");
    if (context.includes("chunk-1")) {
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
    return systemPrompt.includes("Select extraction capabilities")
      ? selection([])
      : extraction({ entities: [], claims: [] });
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

function verification(
  count: number,
  support: "explicit" | "inferred" | "unsupported" = "explicit",
): string {
  return JSON.stringify({
    claims: Array.from({ length: count }, (_, index) => ({ index, support })),
  });
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
  chunks: readonly (readonly [id: string, text: string, acl?: ResourceSnapshot["acl"]])[],
): ResourceSnapshot {
  return {
    organizationId: "acme",
    source: { connectorId: "test", externalId, type: "text" },
    title: externalId,
    contentHash: `${externalId}-hash`,
    body: chunks.map((chunk) => chunk[1]).join("\n"),
    acl: { organizationWide: true },
    chunks: chunks.map(([id, text, acl], position) => ({
      id,
      text,
      position,
      contentHash: `${externalId}-${position}`,
      acl,
    })),
  };
}
