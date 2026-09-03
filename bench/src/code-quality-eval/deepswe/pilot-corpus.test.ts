import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";
import { sha256, validateCorpus } from "./corpus.js";
import { type DeepSwePilotSpec, buildDeepSwePilotCorpus } from "./pilot-corpus.js";

const execFileAsync = promisify(execFile);
const cleanup = new Set<string>();

afterEach(async () => {
  await Promise.all([...cleanup].map((entry) => rm(entry, { recursive: true, force: true })));
  cleanup.clear();
});

describe("DeepSWE pilot corpus", () => {
  it("builds exact Evidence and governance links through the ontology graph", async () => {
    const fixture = await createFixture();
    const outputPath = path.join(fixture.root, "corpora", `${fixture.spec.taskId}.json`);
    const result = await buildDeepSwePilotCorpus({
      spec: fixture.spec,
      checkoutPath: fixture.checkout,
      dataDirectory: path.join(fixture.root, "sidecar"),
      outputPath,
      runtimeProvider: "openai",
      generatorRevision: "test-revision",
    });

    expect(result).toMatchObject({
      codeResources: 1,
      evidenceResources: 1,
      governingRecordIds: ["decision:service-builder"],
    });
    expect(result.behaviorBearingSymbols).toBeGreaterThan(0);
    expect(result.corpus.evidence).toHaveLength(1);
    expect(result.corpus.evidence[0]).toMatchObject({
      text: "export function buildService() {\n  return { ready: true }\n}",
      sourceSpan: "src/service.ts:1-3",
      source: {
        connectorId: "github-repository",
        externalId: expect.stringContaining(`/blob/${fixture.spec.baseCommit}/src/service.ts`),
        type: "code",
      },
      allowedRuntimeProviders: ["openai"],
    });
    expect(result.corpus.normativeRecords[0]?.symbolSelectors).toEqual([
      { relativePath: "src/service.ts", qualifiedName: "buildService" },
    ]);
    expect(() =>
      validateCorpus(result.corpus, fixture.spec.taskId, "/tmp/deep-swe/tasks/pilot-task"),
    ).not.toThrow();
    expect(JSON.parse(await readFile(outputPath, "utf8"))).toEqual(result.corpus);
    expect((await stat(outputPath)).mode & 0o777).toBe(0o600);
  });

  it("fails before export when pinned public source bytes change", async () => {
    const fixture = await createFixture();
    await expect(
      buildDeepSwePilotCorpus({
        spec: {
          ...fixture.spec,
          sourceIntegrity: [{ relativePath: "src/service.ts", sha256: "0".repeat(64) }],
        },
        checkoutPath: fixture.checkout,
        dataDirectory: path.join(fixture.root, "sidecar"),
        outputPath: path.join(fixture.root, "corpus.json"),
        runtimeProvider: "openai",
        generatorRevision: "test-revision",
      }),
    ).rejects.toThrow(/Source integrity mismatch/);
  });
});

async function createFixture(): Promise<{
  readonly root: string;
  readonly checkout: string;
  readonly spec: DeepSwePilotSpec;
}> {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-deepswe-pilot-"));
  cleanup.add(root);
  const checkout = path.join(root, "checkout");
  await mkdir(path.join(checkout, "src"), { recursive: true });
  const source = "export function buildService() {\n  return { ready: true }\n}\n";
  await writeFile(path.join(checkout, "src", "service.ts"), source, "utf8");
  await execFileAsync("git", ["init", "-q"], { cwd: checkout });
  await execFileAsync("git", ["add", "."], { cwd: checkout });
  await execFileAsync(
    "git",
    [
      "-c",
      "user.name=Kontext Eval",
      "-c",
      "user.email=eval@example.invalid",
      "commit",
      "-qm",
      "fixture",
    ],
    { cwd: checkout },
  );
  const baseCommit = (
    await execFileAsync("git", ["rev-parse", "HEAD"], { cwd: checkout })
  ).stdout.trim();
  const nodeId = "domain:test:service";
  const spec: DeepSwePilotSpec = {
    taskId: "pilot-task",
    organizationId: "organization:test",
    codebaseId: `codebase:test@${baseCommit.slice(0, 8)}`,
    repository: "example/service",
    repositoryUrl: "https://github.com/example/service.git",
    baseCommit,
    observedAt: "2025-01-01T00:00:00.000Z",
    snapshotAt: "2025-01-01T00:00:01.000Z",
    sourceIntegrity: [{ relativePath: "src/service.ts", sha256: sha256(source) }],
    sources: [
      {
        evidenceAlias: "service-builder",
        relativePath: "src/service.ts",
        title: "Service implementation",
        startLine: 1,
        endLine: 3,
        ontologyNodeIds: [nodeId],
      },
    ],
    normativeRecords: [
      {
        kind: "decision",
        recordId: "decision:service-builder",
        revisionId: "decision:service-builder@1",
        statement: "Build services through the existing function seam.",
        evidenceAliases: ["service-builder"],
        ontologyNodeIds: [nodeId],
      },
    ],
    targets: [
      {
        workItemId: "work-item:service",
        plannedSymbolId: "planned-symbol:service",
        relativePath: "src/service.ts",
        qualifiedName: "buildService",
        kind: "function",
        responsibility: "Build the service",
        ontologyNodeIds: [nodeId],
        allowedPaths: ["src/service.ts"],
      },
    ],
    contract: {
      intent: "Extend the service builder.",
      acceptance: [
        {
          criterionId: "acceptance:service",
          statement: "The verifier passes.",
          verifier: { kind: "test", ref: "separate verifier" },
        },
      ],
      nonGoals: [],
      risk: "medium",
    },
  };
  return { root, checkout, spec };
}
