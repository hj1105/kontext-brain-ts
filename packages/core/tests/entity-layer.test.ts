import { describe, expect, it } from "vitest";
import {
  AliasEntityExtractor,
  DataSource,
  EntityRetriever,
  InMemoryEntityIndex,
  createEntity,
  createMetaDocument,
} from "../src/index.js";

describe("InMemoryEntityIndex", () => {
  it("finds entities by name and alias", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({ id: "jwt", name: "JWT", type: "concept", aliases: ["JSON Web Token"] }));
    await idx.addEntity(createEntity({ id: "kafka", name: "Apache Kafka", type: "tool" }));

    const matches = await idx.findEntitiesInText("Why use JWT for auth?");
    expect(matches.map((e) => e.id)).toContain("jwt");

    const aliasMatches = await idx.findEntitiesInText("JSON Web Token is stateless");
    expect(aliasMatches.map((e) => e.id)).toContain("jwt");

    const kafkaMatches = await idx.findEntitiesInText("We use Apache Kafka for events");
    expect(kafkaMatches.map((e) => e.id)).toContain("kafka");
  });

  it("maps docs to entities and back", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({ id: "jwt", name: "JWT", type: "concept" }));
    await idx.addMention({ entityId: "jwt", docId: "doc1" });
    await idx.addMention({ entityId: "jwt", docId: "doc2" });

    expect(await idx.docsForEntity("jwt")).toEqual(expect.arrayContaining(["doc1", "doc2"]));
    const forDoc = await idx.entitiesForDoc("doc1");
    expect(forDoc.map((e) => e.id)).toContain("jwt");
  });

  it("walks typed relations via BFS", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({ id: "react", name: "React", type: "framework" }));
    await idx.addEntity(createEntity({ id: "typescript", name: "TypeScript", type: "language" }));
    await idx.addEntity(createEntity({ id: "vite", name: "Vite", type: "tool" }));
    await idx.addRelation({ from: "react", to: "typescript", type: "uses", weight: 0.8 });
    await idx.addRelation({ from: "react", to: "vite", type: "uses", weight: 0.7 });

    const related = await idx.relatedEntities("react", 1);
    const ids = related.map((r) => r.entity.id);
    expect(ids).toEqual(expect.arrayContaining(["typescript", "vite"]));
    expect(related[0]?.depth).toBe(1);
  });

  it("filters relation walks by type", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({ id: "a", name: "A", type: "x" }));
    await idx.addEntity(createEntity({ id: "b", name: "B", type: "x" }));
    await idx.addEntity(createEntity({ id: "c", name: "C", type: "x" }));
    await idx.addRelation({ from: "a", to: "b", type: "uses" });
    await idx.addRelation({ from: "a", to: "c", type: "alternative_to" });

    const onlyUses = await idx.relatedEntities("a", 1, ["uses"]);
    expect(onlyUses.map((r) => r.entity.id)).toEqual(["b"]);
  });
});

describe("AliasEntityExtractor", () => {
  it("returns subset of vocabulary mentioned in text", async () => {
    const vocab = [
      createEntity({ id: "jwt", name: "JWT", type: "concept" }),
      createEntity({ id: "redis", name: "Redis", type: "tool" }),
      createEntity({ id: "kafka", name: "Apache Kafka", type: "tool" }),
    ];
    const ext = new AliasEntityExtractor(vocab);
    const result = await ext.extract("We cache sessions in Redis and sign tokens as JWT.");
    const ids = result.entities.map((e) => e.id).sort();
    expect(ids).toEqual(["jwt", "redis"]);
    expect(result.relations).toHaveLength(0);
  });
});

describe("EntityRetriever", () => {
  it("ranks docs by entity mention overlap with query", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({ id: "jwt", name: "JWT", type: "concept" }));
    await idx.addEntity(createEntity({ id: "redis", name: "Redis", type: "tool" }));

    const docA = createMetaDocument({ id: "a", title: "Auth with JWT and Redis", source: DataSource.CUSTOM, ontologyNodeId: "x" });
    const docB = createMetaDocument({ id: "b", title: "Just Redis", source: DataSource.CUSTOM, ontologyNodeId: "x" });
    const docC = createMetaDocument({ id: "c", title: "Unrelated", source: DataSource.CUSTOM, ontologyNodeId: "x" });

    await idx.addMention({ entityId: "jwt", docId: "a" });
    await idx.addMention({ entityId: "redis", docId: "a" });
    await idx.addMention({ entityId: "redis", docId: "b" });

    const retriever = new EntityRetriever(idx, async () => new Map([
      ["a", docA], ["b", docB], ["c", docC],
    ]));
    const ranked = await retriever.retrieve("JWT auth with Redis sessions", 3);
    expect(ranked[0]?.doc.id).toBe("a"); // mentions both entities
    expect(ranked[1]?.doc.id).toBe("b"); // mentions one
    expect(ranked.map((r) => r.doc.id)).not.toContain("c"); // unrelated
  });

  it("expands via relations", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({ id: "react", name: "React", type: "framework" }));
    await idx.addEntity(createEntity({ id: "jsx", name: "JSX", type: "concept" }));
    await idx.addRelation({ from: "react", to: "jsx", type: "uses" });

    const jsxDoc = createMetaDocument({ id: "jsx-doc", title: "JSX Syntax", source: DataSource.CUSTOM, ontologyNodeId: "x" });
    await idx.addMention({ entityId: "jsx", docId: "jsx-doc" });

    const retriever = new EntityRetriever(idx, async () => new Map([["jsx-doc", jsxDoc]]), 1);
    const ranked = await retriever.retrieve("Tell me about React", 3);
    expect(ranked.map((r) => r.doc.id)).toContain("jsx-doc");
  });
});
