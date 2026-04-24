/**
 * Structured-query benchmark: measures where attribute-based entity
 * retrieval beats vector RAG.
 *
 * SQuAD-style benches can't separate the two capabilities — SQuAD asks for
 * span answers in prose, which vector RAG does well. The instance-of-node
 * entity model shines on structured filter queries ("find X where
 * attribute Y > Z"), which prose RAG fundamentally cannot answer reliably
 * because the answer requires structured reasoning over the corpus.
 *
 * The corpus below is a toy dataset of 8 databases + 6 message queues
 * with their specs as both prose (baseline) and attributes (kontext).
 * The queries exercise: boolean filter, range filter, set membership,
 * combined predicates.
 */

import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import {
  DataSource,
  InMemoryEntityIndex,
  createEntity,
  createMetaDocument,
  type Entity,
  type MetaDocument,
} from "@kontext-brain/core";

// ── typed instance corpus ────────────────────────────────────

const DATABASE_ENTITIES: Entity[] = [
  createEntity({
    id: "postgres", name: "PostgreSQL", type: "Database", nodeId: "database",
    attributes: { kind: "relational", released: 1996, supports_json: true, opensource: true, sharded: false, license: "PostgreSQL" },
  }),
  createEntity({
    id: "mysql", name: "MySQL", type: "Database", nodeId: "database",
    attributes: { kind: "relational", released: 1995, supports_json: true, opensource: true, sharded: false, license: "GPL" },
  }),
  createEntity({
    id: "sqlite", name: "SQLite", type: "Database", nodeId: "database",
    attributes: { kind: "relational", released: 2000, supports_json: false, opensource: true, sharded: false, license: "Public Domain" },
  }),
  createEntity({
    id: "oracle", name: "Oracle DB", type: "Database", nodeId: "database",
    attributes: { kind: "relational", released: 1979, supports_json: true, opensource: false, sharded: true, license: "Commercial" },
  }),
  createEntity({
    id: "mongodb", name: "MongoDB", type: "Database", nodeId: "database",
    attributes: { kind: "document", released: 2009, supports_json: true, opensource: true, sharded: true, license: "SSPL" },
  }),
  createEntity({
    id: "redis", name: "Redis", type: "Database", nodeId: "database",
    attributes: { kind: "keyvalue", released: 2009, supports_json: true, opensource: true, sharded: true, license: "BSD" },
  }),
  createEntity({
    id: "cassandra", name: "Cassandra", type: "Database", nodeId: "database",
    attributes: { kind: "wide-column", released: 2008, supports_json: false, opensource: true, sharded: true, license: "Apache-2.0" },
  }),
  createEntity({
    id: "cockroach", name: "CockroachDB", type: "Database", nodeId: "database",
    attributes: { kind: "relational", released: 2015, supports_json: true, opensource: false, sharded: true, license: "BSL" },
  }),
];

const QUEUE_ENTITIES: Entity[] = [
  createEntity({
    id: "rabbitmq", name: "RabbitMQ", type: "MessageQueue", nodeId: "queue",
    attributes: { released: 2007, opensource: true, replayable: false, license: "MPL-2.0" },
  }),
  createEntity({
    id: "kafka", name: "Apache Kafka", type: "MessageQueue", nodeId: "queue",
    attributes: { released: 2011, opensource: true, replayable: true, license: "Apache-2.0" },
  }),
  createEntity({
    id: "sqs", name: "Amazon SQS", type: "MessageQueue", nodeId: "queue",
    attributes: { released: 2006, opensource: false, replayable: false, license: "Commercial" },
  }),
  createEntity({
    id: "pulsar", name: "Apache Pulsar", type: "MessageQueue", nodeId: "queue",
    attributes: { released: 2016, opensource: true, replayable: true, license: "Apache-2.0" },
  }),
  createEntity({
    id: "nats", name: "NATS", type: "MessageQueue", nodeId: "queue",
    attributes: { released: 2010, opensource: true, replayable: true, license: "Apache-2.0" },
  }),
  createEntity({
    id: "redis-streams", name: "Redis Streams", type: "MessageQueue", nodeId: "queue",
    attributes: { released: 2018, opensource: true, replayable: true, license: "BSD" },
  }),
];

const ALL_ENTITIES = [...DATABASE_ENTITIES, ...QUEUE_ENTITIES];

// Prose description per entity — what a vector RAG would index
function proseDocFor(e: Entity): string {
  const a = e.attributes ?? {};
  if (e.nodeId === "database") {
    return `${e.name} is a ${a.kind} database released in ${a.released}. ` +
      `It is ${a.opensource ? "open source" : "a commercial product"} under the ${a.license} license. ` +
      `It ${a.supports_json ? "supports" : "does not support"} JSON natively. ` +
      `It ${a.sharded ? "supports horizontal sharding" : "does not support built-in sharding"}.`;
  }
  return `${e.name} is a message queue released in ${a.released}. ` +
    `It is ${a.opensource ? "open source" : "a commercial product"} under the ${a.license} license. ` +
    `It ${a.replayable ? "supports replayable event streams" : "uses traditional queue semantics without replay"}.`;
}

// Structured test queries: (natural-language question, predicate spec, expected answers)
interface StructQuery {
  id: string;
  question: string;
  queryFn: (idx: InMemoryEntityIndex) => Promise<Entity[]>;
  expected: string[]; // entity ids
}

const QUERIES: StructQuery[] = [
  {
    id: "sq-0",
    question: "Which open-source relational databases support JSON?",
    queryFn: (idx) => idx.findByAttributes({
      nodeId: "database",
      where: {
        kind: { op: "eq", value: "relational" },
        supports_json: { op: "eq", value: true },
        opensource: { op: "eq", value: true },
      },
    }),
    expected: ["postgres", "mysql"],
  },
  {
    id: "sq-1",
    question: "Which databases support sharding AND were released after 2010?",
    queryFn: (idx) => idx.findByAttributes({
      nodeId: "database",
      where: {
        sharded: { op: "eq", value: true },
        released: { op: "gt", value: 2010 },
      },
    }),
    expected: ["cockroach"],
  },
  {
    id: "sq-2",
    question: "Which message queues support replayable streams and are open source?",
    queryFn: (idx) => idx.findByAttributes({
      nodeId: "queue",
      where: {
        replayable: { op: "eq", value: true },
        opensource: { op: "eq", value: true },
      },
    }),
    expected: ["kafka", "pulsar", "nats", "redis-streams"],
  },
  {
    id: "sq-3",
    question: "Which databases have license Apache or BSD-family?",
    queryFn: (idx) => idx.findByAttributes({
      nodeId: "database",
      where: {
        license: { op: "in", values: ["Apache-2.0", "BSD", "BSL"] },
      },
    }),
    expected: ["cassandra", "redis", "cockroach"],
  },
  {
    id: "sq-4",
    question: "Which databases are NOT open source?",
    queryFn: (idx) => idx.findByAttributes({
      nodeId: "database",
      where: { opensource: { op: "eq", value: false } },
    }),
    expected: ["oracle", "cockroach"],
  },
];

// ── vector-RAG baseline over prose ────────────────────────────

class VectorProseBaseline {
  private chunks: Array<{ id: string; vec: number[]; text: string }> = [];
  private readonly embeddings: OllamaEmbeddings;
  private readonly chat: ChatOllama;

  constructor() {
    const baseUrl = "http://localhost:11434";
    this.embeddings = new OllamaEmbeddings({ baseUrl, model: "nomic-embed-text" });
    this.chat = new ChatOllama({
      baseUrl, model: "qwen2.5:1.5b", temperature: 0, numGpu: 0, numCtx: 2048, numPredict: 256,
    });
  }

  async index(entities: Entity[]): Promise<void> {
    const texts = entities.map((e) => ({ id: e.id, text: proseDocFor(e) }));
    const vecs = await this.embeddings.embedDocuments(texts.map((t) => t.text));
    this.chunks = texts.map((t, i) => ({ id: t.id, vec: vecs[i] ?? [], text: t.text }));
  }

  async answer(question: string): Promise<string[]> {
    const qv = await this.embeddings.embedQuery(question);
    const scored = this.chunks.map((c) => ({ c, s: cosine(qv, c.vec) }));
    scored.sort((a, b) => b.s - a.s);
    const topK = scored.slice(0, 8);
    const context = topK.map((x) => `- ${x.c.text}`).join("\n");

    const prompt = `Given the facts below, list all names that match the question. Output ONLY names separated by commas, nothing else.\n\nFacts:\n${context}\n\nQuestion: ${question}\nAnswer:`;
    const res = await this.chat.invoke(prompt);
    const text = typeof res.content === "string" ? res.content : JSON.stringify(res.content);
    // Parse names → ids
    const out: string[] = [];
    for (const e of [...DATABASE_ENTITIES, ...QUEUE_ENTITIES]) {
      const nameLc = e.name.toLowerCase();
      if (text.toLowerCase().includes(nameLc)) out.push(e.id);
    }
    return out;
  }
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += (a[i] ?? 0) * (b[i] ?? 0);
    na += (a[i] ?? 0) ** 2;
    nb += (b[i] ?? 0) ** 2;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── bench runner ──────────────────────────────────────────────

function f1(expected: string[], got: string[]): { precision: number; recall: number; f1: number } {
  const expSet = new Set(expected);
  const gotSet = new Set(got);
  const tp = got.filter((g) => expSet.has(g)).length;
  const precision = gotSet.size === 0 ? 0 : tp / gotSet.size;
  const recall = expSet.size === 0 ? 1 : tp / expSet.size;
  const f = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { precision, recall, f1: f };
}

async function main(): Promise<void> {
  console.log("=== Structured-query bench: attribute retrieval vs vector RAG ===\n");

  const idx = new InMemoryEntityIndex();
  for (const e of ALL_ENTITIES) await idx.addEntity(e);

  const baseline = new VectorProseBaseline();
  console.log("Indexing baseline vector RAG over prose descriptions...");
  const t0 = Date.now();
  await baseline.index(ALL_ENTITIES);
  console.log(`  done in ${Date.now() - t0}ms`);

  const results = {
    baseline: { p: 0, r: 0, f: 0, latency: 0, perQuery: [] as Array<{ id: string; answers: string[]; latency: number; f1: number }> },
    attribute: { p: 0, r: 0, f: 0, latency: 0, perQuery: [] as Array<{ id: string; answers: string[]; latency: number; f1: number }> },
  };

  for (const q of QUERIES) {
    console.log(`\n[${q.id}] ${q.question}`);
    console.log(`  expected: ${q.expected.join(", ")}`);

    // baseline
    const t1 = performance.now();
    const baseAns = await baseline.answer(q.question);
    const baseLat = performance.now() - t1;
    const bm = f1(q.expected, baseAns);
    results.baseline.p += bm.precision;
    results.baseline.r += bm.recall;
    results.baseline.f += bm.f1;
    results.baseline.latency += baseLat;
    results.baseline.perQuery.push({ id: q.id, answers: baseAns, latency: baseLat, f1: bm.f1 });
    console.log(`  baseline  F1=${bm.f1.toFixed(2)} P=${bm.precision.toFixed(2)} R=${bm.recall.toFixed(2)} [${baseLat.toFixed(0)}ms]: ${baseAns.join(", ")}`);

    // attribute
    const t2 = performance.now();
    const attrEntities = await q.queryFn(idx);
    const attrAns = attrEntities.map((e) => e.id);
    const attrLat = performance.now() - t2;
    const am = f1(q.expected, attrAns);
    results.attribute.p += am.precision;
    results.attribute.r += am.recall;
    results.attribute.f += am.f1;
    results.attribute.latency += attrLat;
    results.attribute.perQuery.push({ id: q.id, answers: attrAns, latency: attrLat, f1: am.f1 });
    console.log(`  attribute F1=${am.f1.toFixed(2)} P=${am.precision.toFixed(2)} R=${am.recall.toFixed(2)} [${attrLat.toFixed(3)}ms]: ${attrAns.join(", ")}`);
  }

  const N = QUERIES.length;
  console.log("\n=== Aggregate (structured queries) ===");
  console.log("system      precision  recall    F1     avg_latency");
  console.log(
    `baseline    ${(results.baseline.p / N).toFixed(3)}    ${(results.baseline.r / N).toFixed(3)}   ${(results.baseline.f / N).toFixed(3)}   ${(results.baseline.latency / N).toFixed(0)}ms`,
  );
  console.log(
    `attribute   ${(results.attribute.p / N).toFixed(3)}    ${(results.attribute.r / N).toFixed(3)}   ${(results.attribute.f / N).toFixed(3)}   ${(results.attribute.latency / N).toFixed(3)}ms`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
