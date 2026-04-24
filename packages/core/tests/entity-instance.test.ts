import { describe, expect, it } from "vitest";
import {
  InMemoryEntityIndex,
  createEntity,
  createNode,
  validateEntityAttributes,
} from "../src/index.js";

describe("Entity as instance-of-node (proper ontological sense)", () => {
  it("attaches an entity to its parent node via nodeId", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({
      id: "postgres",
      name: "PostgreSQL",
      type: "Database",
      nodeId: "database",
      attributes: { version: "15", supports_json: true, license: "PostgreSQL" },
    }));
    const instances = await idx.entitiesForNode("database");
    expect(instances.map((e) => e.id)).toEqual(["postgres"]);
    expect(instances[0]?.attributes?.version).toBe("15");
    expect(instances[0]?.attributes?.supports_json).toBe(true);
  });

  it("queries entities by attribute equality", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({
      id: "postgres", name: "PostgreSQL", type: "Database", nodeId: "database",
      attributes: { supports_json: true, released: 1996 },
    }));
    await idx.addEntity(createEntity({
      id: "mysql", name: "MySQL", type: "Database", nodeId: "database",
      attributes: { supports_json: true, released: 1995 },
    }));
    await idx.addEntity(createEntity({
      id: "sqlite", name: "SQLite", type: "Database", nodeId: "database",
      attributes: { supports_json: false, released: 2000 },
    }));

    const jsonDbs = await idx.findByAttributes({
      nodeId: "database",
      where: { supports_json: { op: "eq", value: true } },
    });
    expect(jsonDbs.map((e) => e.id).sort()).toEqual(["mysql", "postgres"]);
  });

  it("combines multiple predicates with AND", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({
      id: "postgres", name: "PostgreSQL", type: "Database", nodeId: "database",
      attributes: { supports_json: true, released: 1996 },
    }));
    await idx.addEntity(createEntity({
      id: "mongo", name: "MongoDB", type: "Database", nodeId: "database",
      attributes: { supports_json: true, released: 2009 },
    }));

    const modernJsonDbs = await idx.findByAttributes({
      nodeId: "database",
      where: {
        supports_json: { op: "eq", value: true },
        released: { op: "gte", value: 2000 },
      },
    });
    expect(modernJsonDbs.map((e) => e.id)).toEqual(["mongo"]);
  });

  it("supports comparison operators", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({
      id: "a", name: "A", type: "Tool", nodeId: "tool",
      attributes: { stars: 100 },
    }));
    await idx.addEntity(createEntity({
      id: "b", name: "B", type: "Tool", nodeId: "tool",
      attributes: { stars: 500 },
    }));
    await idx.addEntity(createEntity({
      id: "c", name: "C", type: "Tool", nodeId: "tool",
      attributes: { stars: 1000 },
    }));

    const popular = await idx.findByAttributes({
      nodeId: "tool",
      where: { stars: { op: "gt", value: 200 } },
    });
    expect(popular.map((e) => e.id).sort()).toEqual(["b", "c"]);
  });

  it("supports string contains and array has predicates", async () => {
    const idx = new InMemoryEntityIndex();
    await idx.addEntity(createEntity({
      id: "postgres", name: "PostgreSQL", type: "Database", nodeId: "db",
      attributes: { license: "PostgreSQL License", tags: ["relational", "opensource"] },
    }));
    await idx.addEntity(createEntity({
      id: "oracle", name: "Oracle", type: "Database", nodeId: "db",
      attributes: { license: "Commercial", tags: ["relational", "enterprise"] },
    }));

    const openSource = await idx.findByAttributes({
      nodeId: "db",
      where: { tags: { op: "has", value: "opensource" } },
    });
    expect(openSource.map((e) => e.id)).toEqual(["postgres"]);

    const pgLicensed = await idx.findByAttributes({
      nodeId: "db",
      where: { license: { op: "contains", value: "postgresql" } },
    });
    expect(pgLicensed.map((e) => e.id)).toEqual(["postgres"]);
  });

  it("excludes NER-style entities (no nodeId) from attribute queries", async () => {
    const idx = new InMemoryEntityIndex();
    // typed instance
    await idx.addEntity(createEntity({
      id: "postgres", name: "PostgreSQL", type: "Database", nodeId: "db",
      attributes: { supports_json: true },
    }));
    // NER mention, no nodeId
    await idx.addEntity(createEntity({ id: "kafka", name: "Kafka", type: "tool" }));

    const all = await idx.findByAttributes({});
    expect(all.map((e) => e.id)).toEqual(["postgres"]);
  });
});

describe("validateEntityAttributes", () => {
  it("returns empty list when attributes match schema", () => {
    const node = createNode({
      id: "db",
      description: "database class",
      attributeSchema: { version: "string", supports_json: "boolean", released: "number" },
    });
    const issues = validateEntityAttributes(
      { version: "15", supports_json: true, released: 1996 },
      node.attributeSchema,
    );
    expect(issues).toEqual([]);
  });

  it("reports missing and mistyped attributes", () => {
    const schema = { version: "string" as const, stars: "number" as const };
    const issues = validateEntityAttributes({ version: 15 as unknown as string }, schema);
    expect(issues).toContain("missing attribute: stars");
    expect(issues.some((s) => s.includes("version"))).toBe(true);
  });

  it("treats string[] type correctly", () => {
    const issues = validateEntityAttributes(
      { tags: ["a", "b"] },
      { tags: "string[]" },
    );
    expect(issues).toEqual([]);
    const bad = validateEntityAttributes(
      { tags: "not-an-array" as unknown as readonly string[] },
      { tags: "string[]" },
    );
    expect(bad.some((s) => s.includes("tags"))).toBe(true);
  });
});
