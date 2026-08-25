import {
  ActivateOntologyUseCase,
  BidirectionalNLayerRetriever,
  CalibratedTraversalScorePolicy,
  DEFAULT_CALIBRATED_SCORING_PROFILE,
  InMemoryResourceContentStore,
  type OntologyCandidateValidator,
  type OntologyCompiler,
  type OntologyReindexer,
  type ResourceSnapshot,
  RetrieveFactsUseCase,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { Pool, type PoolClient } from "pg";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import {
  PostgresAuthorizedKnowledgeGraphReader,
  PostgresChunkVectorIndex,
  PostgresExtractionJobQueue,
  PostgresKnowledgeGraphRepository,
  PostgresKnowledgeSearchGraph,
  PostgresOntologyDeploymentRepository,
  PostgresOntologyProposalQueue,
  PostgresScoringProfileRepository,
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

  it("stages and atomically activates versioned scoring profiles", async () => {
    const profiles = new PostgresScoringProfileRepository(requiredPool(), 0);
    const profile = {
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      id: "test-calibrated",
      version: 2,
    };

    const staged = await profiles.stage("acme", profile, {
      split: "validation",
      recallAt10: 1,
      ndcgAt10: 1,
    });
    expect(staged).toMatchObject({ status: "staged", profile });
    const active = await profiles.activate("acme", staged.profileDigest);
    expect(active.status).toBe("active");
    expect(await profiles.getActive("acme")).toMatchObject({
      profileDigest: staged.profileDigest,
      profile,
    });

    const shadow = await profiles.stage(
      "acme",
      {
        ...DEFAULT_CALIBRATED_SCORING_PROFILE,
        id: "test-shadow",
        version: 1,
      },
      { split: "validation", recallAt10: 1, ndcgAt10: 1 },
    );
    await profiles.setShadow("acme", shadow.profileDigest);
    expect((await profiles.getShadow("acme"))?.profileDigest).toBe(shadow.profileDigest);

    const resolved = await profiles.resolve({
      organizationId: "acme",
      subjectId: "u1",
      groupIds: [],
    });
    expect("descriptor" in resolved && resolved.descriptor.profileDigest).toBe(
      staged.profileDigest,
    );
    await profiles.setCanaryPercent("acme", 0);
    expect(
      "descriptor" in
        (await profiles.resolve({
          organizationId: "acme",
          subjectId: "u1",
          groupIds: [],
        })),
    ).toBe(false);
    await profiles.setCanaryPercent("acme", 100);

    await profiles.activate("acme", shadow.profileDigest);
    expect(await profiles.getShadow("acme")).toBeNull();
    expect((await profiles.rollback("acme", staged.profileDigest)).profileDigest).toBe(
      staged.profileDigest,
    );
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

    await queue.markPublished("acme", ["refund"]);
    await queue.enqueue("acme", [
      { suggestedNodeId: "refund", description: "Refunds", resourceIds: ["github:i1"] },
    ]);
    expect(await queue.listOpen("acme")).toEqual([]);
    expect(await queue.listPending("acme")).toMatchObject([
      { occurrences: 3, status: "published" },
    ]);
  });

  it("uses a single pooled connection and transaction for retrieval and vector seeds", async () => {
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
      graph: { nodes: [{ id: "order", description: "Customer order" }], edges: [] },
      status: "staged",
      createdAt: new Date().toISOString(),
    });
    await deployments.activate("acme", "ontology-v1");

    let checkouts = 0;
    let begins = 0;
    class CountingPool extends Pool {
      override async connect(): Promise<never> {
        if (checkouts > 0) throw new Error("Retrieval attempted a second pool checkout");
        const client = await (super.connect() as Promise<PoolClient>);
        checkouts++;
        const query = client.query.bind(client) as PoolClient["query"];
        // biome-ignore lint/suspicious/noExplicitAny: passthrough spy
        (client as any).query = (...args: any[]) => {
          const text = typeof args[0] === "string" ? args[0] : (args[0]?.text ?? "");
          if (String(text).trimStart().toUpperCase().startsWith("BEGIN")) begins++;
          // biome-ignore lint/suspicious/noExplicitAny: passthrough spy
          return (query as any)(...args);
        };
        return client as never;
      }
    }
    const countingPool = new CountingPool({ connectionString: databaseUrl, max: 1 });
    try {
      const vectorIndex = new PostgresChunkVectorIndex(countingPool, {
        async embed() {
          return new Array<number>(1536).fill(0);
        },
      });
      const retriever = new BidirectionalNLayerRetriever(
        new PostgresKnowledgeSearchGraph(countingPool, contents, [vectorIndex]),
        new CalibratedTraversalScorePolicy(DEFAULT_CALIBRATED_SCORING_PROFILE),
      );
      const result = await retriever.retrieve({
        question: "Was order 42 paid?",
        principal: { organizationId: "acme", subjectId: "u1", groupIds: ["finance"] },
      });

      expect(result.trace.visited).toBeGreaterThan(1);
      expect(checkouts).toBe(1);
      expect(begins).toBe(1);
      expect(countingPool.idleCount).toBe(1);
    } finally {
      await countingPool.end();
    }
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
      new CalibratedTraversalScorePolicy(DEFAULT_CALIBRATED_SCORING_PROFILE),
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
