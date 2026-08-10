import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  ActivateOntologyUseCase,
  InMemoryOntologyDeploymentRepository,
  type OntologyCandidateValidator,
  type OntologyCompiler,
  type OntologyReindexer,
} from "../src/index.js";

interface TestGraph {
  readonly nodes: readonly string[];
}

class TestCompiler implements OntologyCompiler<TestGraph> {
  calls = 0;

  async compile(yaml: string): Promise<TestGraph> {
    this.calls++;
    if (yaml.includes("invalid")) throw new Error("invalid ontology");
    return { nodes: yaml.match(/id:/g)?.map((_, index) => `node-${index}`) ?? [] };
  }
}

class TestValidator implements OntologyCandidateValidator<TestGraph> {
  async validate(candidate: { graph: TestGraph }): Promise<void> {
    if (candidate.graph.nodes.length === 0) throw new Error("ontology must contain a node");
  }
}

class TestReindexer implements OntologyReindexer<TestGraph> {
  preparedHashes: string[] = [];

  async prepare(candidate: { contentHash: string }): Promise<void> {
    this.preparedHashes.push(candidate.contentHash);
  }
}

describe("ActivateOntologyUseCase", () => {
  it("hashes YAML, prepares a candidate, and atomically activates it", async () => {
    const repository = new InMemoryOntologyDeploymentRepository<TestGraph>();
    const compiler = new TestCompiler();
    const reindexer = new TestReindexer();
    const activate = new ActivateOntologyUseCase(
      repository,
      compiler,
      new TestValidator(),
      reindexer,
    );
    const yaml = "ontology:\n  - id: order\n";

    const result = await activate.execute({ organizationId: "acme", yaml, gitCommit: "abc123" });

    const expectedHash = createHash("sha256").update(yaml).digest("hex");
    expect(result).toMatchObject({ changed: true, contentHash: expectedHash });
    expect((await repository.getActive("acme"))?.contentHash).toBe(expectedHash);
    expect(reindexer.preparedHashes).toEqual([expectedHash]);
  });

  it("does no work when the exact YAML content is already active", async () => {
    const repository = new InMemoryOntologyDeploymentRepository<TestGraph>();
    const compiler = new TestCompiler();
    const activate = new ActivateOntologyUseCase(
      repository,
      compiler,
      new TestValidator(),
      new TestReindexer(),
    );
    const input = { organizationId: "acme", yaml: "- id: order" };

    await activate.execute(input);
    const second = await activate.execute(input);

    expect(second.changed).toBe(false);
    expect(compiler.calls).toBe(1);
  });

  it("keeps the previous deployment active when candidate compilation or validation fails", async () => {
    const repository = new InMemoryOntologyDeploymentRepository<TestGraph>();
    const activate = new ActivateOntologyUseCase(
      repository,
      new TestCompiler(),
      new TestValidator(),
      new TestReindexer(),
    );
    await activate.execute({ organizationId: "acme", yaml: "- id: order" });
    const activeBefore = await repository.getActive("acme");

    await expect(activate.execute({ organizationId: "acme", yaml: "invalid" })).rejects.toThrow(
      "invalid ontology",
    );

    expect(await repository.getActive("acme")).toEqual(activeBefore);
  });
});
