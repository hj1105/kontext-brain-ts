/**
 * Entities as instances of ontology nodes (proper ontological sense).
 *
 * An OntologyNode is a CLASS (e.g. "Database"); an Entity is an INSTANCE
 * of that class carrying attribute values defined by the node's schema.
 *
 * Contrast with the NER-style "mention" interpretation also supported by
 * the same Entity type — see examples/basic for that. This example focuses
 * on the structured-knowledge use case.
 *
 * Run: pnpm --filter @kontext-brain/example-entity-instances start
 */

import {
  InMemoryEntityIndex,
  createEntity,
  createNode,
  validateEntityAttributes,
  type Entity,
  type OntologyNode,
} from "@kontext-brain/core";

async function main(): Promise<void> {
  // 1. Define a node (class) with its attribute schema
  const databaseNode: OntologyNode = createNode({
    id: "database",
    description: "Persistent data store",
    attributeSchema: {
      version: "string",
      released: "number",
      supports_json: "boolean",
      license: "string",
      tags: "string[]",
    },
  });

  // 2. Create entities as instances of that node
  const postgres = createEntity({
    id: "postgres",
    name: "PostgreSQL",
    type: "Database",
    nodeId: databaseNode.id,
    aliases: ["Postgres"],
    attributes: {
      version: "15",
      released: 1996,
      supports_json: true,
      license: "PostgreSQL License",
      tags: ["relational", "opensource", "acid"],
    },
  });
  const mysql = createEntity({
    id: "mysql",
    name: "MySQL",
    type: "Database",
    nodeId: databaseNode.id,
    attributes: {
      version: "8.3",
      released: 1995,
      supports_json: true,
      license: "GPL",
      tags: ["relational", "opensource"],
    },
  });
  const sqlite = createEntity({
    id: "sqlite",
    name: "SQLite",
    type: "Database",
    nodeId: databaseNode.id,
    attributes: {
      version: "3.45",
      released: 2000,
      supports_json: false,
      license: "Public Domain",
      tags: ["relational", "embedded"],
    },
  });
  const mongodb = createEntity({
    id: "mongodb",
    name: "MongoDB",
    type: "Database",
    nodeId: databaseNode.id,
    attributes: {
      version: "7.0",
      released: 2009,
      supports_json: true,
      license: "SSPL",
      tags: ["document", "nosql"],
    },
  });

  // 3. Validate the attributes match the node's schema (optional but recommended)
  for (const e of [postgres, mysql, sqlite, mongodb]) {
    const issues = validateEntityAttributes(
      e.attributes as Record<string, any>,
      databaseNode.attributeSchema,
    );
    if (issues.length > 0) {
      console.warn(`${e.name} schema issues:`, issues);
    }
  }

  // 4. Register in an EntityIndex
  const idx = new InMemoryEntityIndex();
  for (const e of [postgres, mysql, sqlite, mongodb]) await idx.addEntity(e);

  // 5. Structured queries on attribute values
  console.log("=== Entity instance-of-node queries ===\n");

  const allDbs = await idx.entitiesForNode("database");
  console.log(`All databases: ${allDbs.map((e) => e.name).join(", ")}\n`);

  const jsonDbs = await idx.findByAttributes({
    nodeId: "database",
    where: { supports_json: { op: "eq", value: true } },
  });
  console.log(
    `Databases supporting JSON: ${jsonDbs.map((e) => e.name).join(", ")}\n`,
  );

  const modernOpenSource = await idx.findByAttributes({
    nodeId: "database",
    where: {
      released: { op: "gte", value: 2000 },
      tags: { op: "has", value: "opensource" },
    },
  });
  console.log(
    `Open-source DBs released 2000+: ${modernOpenSource.map((e) => e.name).join(", ") || "(none)"}\n`,
  );

  const nosql = await idx.findByAttributes({
    nodeId: "database",
    where: { tags: { op: "has", value: "nosql" } },
  });
  console.log(`NoSQL databases: ${nosql.map((e) => e.name).join(", ")}\n`);

  const relational = await idx.findByAttributes({
    nodeId: "database",
    where: { tags: { op: "has", value: "relational" } },
  });
  console.log(
    `Relational databases: ${relational.map((e) => e.name).join(", ")}\n`,
  );

  console.log(
    `Details for ${postgres.name}: version=${postgres.attributes?.version}, license=${postgres.attributes?.license}`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
