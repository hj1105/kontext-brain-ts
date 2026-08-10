import {
  ActivateOntologyUseCase,
  BidirectionalNLayerRetriever,
  InMemoryResourceContentStore,
  type OntologyCandidateValidator,
  type OntologyCompiler,
  type OntologyReindexer,
  type ResourceSnapshot,
  RetrieveFactsUseCase,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { Pool } from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import {
  PostgresAuthorizedKnowledgeGraphReader,
  PostgresExtractionJobQueue,
  PostgresKnowledgeGraphRepository,
  PostgresKnowledgeSearchGraph,
  PostgresOntologyDeploymentRepository,
  PostgresOntologyProposalQueue,
  migratePostgres,
} from "../src/index.js";

const databaseUrl = process.env.KONTEXT_TEST_DATABASE_URL;
const pool = databaseUrl ? new Pool({ connectionString: databaseUrl }) : null;

afterAll(async () => {
  await pool?.end();
});

describe.runIf(pool !== null)("PostgreSQL adapters", () => {
  beforeEach(async () => {
    await requiredPool().query("DROP SCHEMA public CASCADE; CREATE SCHEMA public");
    await migratePostgres(requiredPool());
  });

  it("runs the same Evidence lifecycle atomically in PostgreSQL", async () => {
    const repository = new PostgresKnowledgeGraphRepository(requiredPool());
    const sync = new SyncResourceUseCase(repository, new InMemoryResourceContentStore());

    await sync.execute(snapshot("v1", "paid"));
    await sync.execute(snapshot("v2"));
    await sync.execute(snapshot("v3", "paid"));

    expect((await repository.getFact("acme", "order:42:status:paid"))?.status).toBe("active");
    expect(await repository.listFactEvents("acme", "order:42:status:paid")).toMatchObject([
      { type: "created" },
      { type: "invalidated" },
      { type: "restored" },
    ]);
  });

  it("applies ACL predicates in SQL before returning Evidence metadata", async () => {
    const repository = new PostgresKnowledgeGraphRepository(requiredPool());
    const contents = new InMemoryResourceContentStore();
    await new SyncResourceUseCase(repository, contents).execute(snapshot("v1", "paid"));
    const retrieve = new RetrieveFactsUseCase(
      new PostgresAuthorizedKnowledgeGraphReader(requiredPool()),
      contents,
    );

    const allowed = await retrieve.execute({
      principal: { organizationId: "acme", subjectId: "u1", groupIds: ["finance"] },
    });
    const denied = await retrieve.execute({
      principal: { organizationId: "acme", subjectId: "u2", groupIds: ["engineering"] },
    });

    expect(allowed[0]?.evidence[0]?.text).toBe("Order 42 is paid");
    expect(denied).toEqual([]);
  });

  it("switches ontology deployments only after candidate preparation", async () => {
    type Graph = { nodes: string[] };
    const compiler: OntologyCompiler<Graph> = {
      async compile(yaml) {
        if (yaml === "invalid") throw new Error("invalid");
        return { nodes: [yaml] };
      },
    };
    const validator: OntologyCandidateValidator<Graph> = { async validate() {} };
    const reindexer: OntologyReindexer<Graph> = { async prepare() {} };
    const repository = new PostgresOntologyDeploymentRepository<Graph>(requiredPool());
    const activate = new ActivateOntologyUseCase(repository, compiler, validator, reindexer);

    await activate.execute({ organizationId: "acme", yaml: "- id: order" });
    const active = await repository.getActive("acme");
    await expect(activate.execute({ organizationId: "acme", yaml: "invalid" })).rejects.toThrow();

    expect(await repository.getActive("acme")).toEqual(active);
  });

  it("deduplicates and leases extraction jobs by resource, content, and ontology hashes", async () => {
    const queue = new PostgresExtractionJobQueue(requiredPool());
    const key = {
      organizationId: "acme",
      resourceId: "notion:orders",
      contentHash: "content-v1",
      ontologyHash: "ontology-v1",
    };

    expect(await queue.enqueue(key)).toBe(true);
    expect(await queue.enqueue(key)).toBe(false);
    const claimed = await queue.claim("acme", "worker-1", 10, 30_000);
    expect(claimed).toHaveLength(1);
    expect(claimed[0]).toMatchObject({ attempts: 1, state: "running", lockedBy: "worker-1" });
    await queue.succeed(key, "worker-1");
    expect(await queue.claim("acme", "worker-2", 10, 30_000)).toEqual([]);
  });

  it("persists and deduplicates ontology proposals without mutating the active ontology", async () => {
    const queue = new PostgresOntologyProposalQueue(requiredPool());
    await queue.enqueue("acme", [
      { suggestedNodeId: "refund", description: "Refunds", resourceIds: ["notion:p1"] },
    ]);
    await queue.enqueue("acme", [
      { suggestedNodeId: "refund", description: "Refunds", resourceIds: ["slack:t1"] },
    ]);

    expect(await queue.listOpen("acme")).toMatchObject([
      { occurrences: 2, resourceIds: ["notion:p1", "slack:t1"] },
    ]);
  });

  it("lifts and grounds through the PostgreSQL KG while filtering ACL before each edge", async () => {
    const repository = new PostgresKnowledgeGraphRepository(requiredPool());
    const contents = new InMemoryResourceContentStore();
    await new SyncResourceUseCase(repository, contents).execute(snapshot("v1", "paid"));
    const deployments = new PostgresOntologyDeploymentRepository<{
      nodes: Array<{ id: string; description: string }>;
      edges: Array<{ from: string; to: string; weight: number }>;
    }>(requiredPool());
    await deployments.stage({
      organizationId: "acme",
      contentHash: "ontology-v1",
      yaml: "- id: order",
      graph: {
        nodes: [{ id: "order", description: "Customer order" }],
        edges: [],
      },
      status: "staged",
      createdAt: new Date().toISOString(),
    });
    await deployments.activate("acme", "ontology-v1");
    const retriever = new BidirectionalNLayerRetriever(
      new PostgresKnowledgeSearchGraph(requiredPool(), contents),
    );

    const allowed = await retriever.retrieve({
      question: "Was order 42 paid?",
      principal: { organizationId: "acme", subjectId: "u1", groupIds: ["finance"] },
    });
    const denied = await retriever.retrieve({
      question: "Was order 42 paid?",
      principal: { organizationId: "acme", subjectId: "u2", groupIds: ["engineering"] },
    });

    expect(allowed.evidence[0]).toMatchObject({
      factKey: "order:42:status:paid",
      text: "Order 42 is paid",
    });
    expect(allowed.evidence[0]?.path.map((edge) => edge.operation)).toEqual(["ground", "ground"]);
    expect(denied.evidence).toEqual([]);
  });
});

function requiredPool(): Pool {
  if (!pool) throw new Error("KONTEXT_TEST_DATABASE_URL is required");
  return pool;
}

function snapshot(contentHash: string, status?: "paid"): ResourceSnapshot {
  return {
    organizationId: "acme",
    source: { connectorId: "notion", externalId: "orders", type: "notion" },
    title: "Orders",
    contentHash,
    body: status ? "Order 42 is paid" : "Status removed",
    acl: { groupIds: ["finance"] },
    ontologyNodeIds: ["order", "payment"],
    chunks: [
      {
        id: "block-1",
        contentHash: `${contentHash}:block`,
        text: status ? "Order 42 is paid" : "Status removed",
        position: 0,
        ontologyNodeIds: ["order", "payment"],
      },
    ],
    facts: status
      ? [
          {
            factKey: "order:42:status:paid",
            subject: { entityId: "order:42", scope: "global" },
            predicate: "status",
            object: { kind: "literal", value: "paid" },
            evidenceChunkIds: ["block-1"],
            singleValue: true,
          },
        ]
      : [],
  };
}
