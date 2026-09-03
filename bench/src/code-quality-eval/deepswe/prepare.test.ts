import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";
import { prepareDeepSweEvaluation, stripPierCanary } from "./prepare.js";
import { fixtureCorpus } from "./test-fixtures.js";

const execFileAsync = promisify(execFile);
const cleanup = new Set<string>();

afterEach(async () => {
  await Promise.all([...cleanup].map((entry) => rm(entry, { recursive: true, force: true })));
  cleanup.clear();
});

describe("DeepSWE evaluation preparation", () => {
  it("pins tasks and produces three isolated context projections", async () => {
    const fixture = await createFixture();
    const manifest = await prepareDeepSweEvaluation({
      repositoryRoot: "/repo",
      datasetTasksPath: fixture.tasks,
      corpusRoot: fixture.corpora,
      runDirectory: fixture.run,
      jobsDirectory: path.join(fixture.run, "jobs"),
      pierBinary: "pier",
      model: "openai/test-model",
      reasoningEffort: "medium",
      attempts: 4,
      concurrency: 1,
      sampleSeed: 7,
      arms: ["baseline", "rag", "kontext"],
      environment: "docker",
      miniSweAgentVersion: "2.1.0",
      deepSweRevision: fixture.revision,
      pierRevision: "pier-0.3.1",
      adapterRevision: "adapter-sha",
    });

    expect(manifest.deepSweRevision).toBe(fixture.revision);
    expect(manifest.tasks).toHaveLength(2);
    expect(manifest.arms).toHaveLength(3);
    const indexes = await Promise.all(
      manifest.arms.map(
        async (arm) =>
          JSON.parse(await readFile(arm.contextIndexPath, "utf8")) as {
            arm: string;
            byInstructionSha256: Record<
              string,
              { evidence: unknown[]; normativeRecords: unknown[]; corpusSha256: string }
            >;
          },
      ),
    );
    const bundles = indexes.map((index) => required(Object.values(index.byInstructionSha256)[0]));
    expect(bundles[0]?.evidence).toEqual([]);
    expect(bundles[1]?.evidence).toHaveLength(1);
    expect(bundles[1]?.normativeRecords).toEqual([]);
    expect(bundles[2]?.evidence).toHaveLength(1);
    expect(bundles[2]?.normativeRecords).toHaveLength(1);
    expect(new Set(bundles.map((bundle) => bundle.corpusSha256)).size).toBe(1);

    const configs = await Promise.all(
      manifest.arms.map(async (arm) => JSON.parse(await readFile(arm.jobConfigPath, "utf8"))),
    );
    for (const config of configs) {
      expect(config.n_attempts).toBe(4);
      expect(config.retry).toEqual({ max_retries: 0 });
      expect(config.agents[0].import_path).toBe("kontext_mini_swe_agent:KontextMiniSweAgent");
      expect(config.agents[0].kwargs.version).toBe("2.1.0");
      expect(config.tasks).toHaveLength(2);
      expect(config.tasks.every((task: { source?: string }) => task.source === undefined)).toBe(
        true,
      );
    }
  });

  it("matches Pier's leading canary normalization", () => {
    expect(stripPierCanary("<!-- benchmark canary -->\n# SECOND CANARY\n\nDo the work.\n")).toBe(
      "Do the work.\n",
    );
    expect(stripPierCanary("Do not remove an inline canary reference.")).toBe(
      "Do not remove an inline canary reference.",
    );
  });

  it("refuses a mismatched DeepSWE checkout revision", async () => {
    const fixture = await createFixture();
    await expect(
      prepareDeepSweEvaluation({
        repositoryRoot: "/repo",
        datasetTasksPath: fixture.tasks,
        corpusRoot: fixture.corpora,
        runDirectory: fixture.run,
        jobsDirectory: path.join(fixture.run, "jobs"),
        pierBinary: "pier",
        model: "openai/test-model",
        reasoningEffort: "medium",
        attempts: 1,
        concurrency: 1,
        sampleSeed: 0,
        arms: ["baseline"],
        environment: "docker",
        miniSweAgentVersion: "2.1.0",
        deepSweRevision: "0000000000000000000000000000000000000000",
        pierRevision: "pier-0.3.1",
        adapterRevision: "adapter-sha",
      }),
    ).rejects.toThrow(/revision mismatch/);
  });

  it("refuses a corpus frozen against a different base code revision", async () => {
    const fixture = await createFixture();
    const corpusPath = path.join(fixture.corpora, "alpha-task.json");
    const corpus = JSON.parse(await readFile(corpusPath, "utf8")) as Record<string, unknown>;
    await writeFile(
      corpusPath,
      `${JSON.stringify({ ...corpus, baseCodeRevision: "b".repeat(40) }, null, 2)}\n`,
      "utf8",
    );

    await expect(
      prepareDeepSweEvaluation({
        repositoryRoot: "/repo",
        datasetTasksPath: fixture.tasks,
        corpusRoot: fixture.corpora,
        runDirectory: fixture.run,
        jobsDirectory: path.join(fixture.run, "jobs"),
        pierBinary: "pier",
        model: "openai/test-model",
        reasoningEffort: "medium",
        attempts: 1,
        concurrency: 1,
        sampleSeed: 0,
        arms: ["baseline"],
        taskIds: ["alpha-task"],
        environment: "docker",
        miniSweAgentVersion: "2.1.0",
        deepSweRevision: fixture.revision,
        pierRevision: "pier-0.3.1",
        adapterRevision: "adapter-sha",
      }),
    ).rejects.toThrow(/base code revision mismatch/);
  });

  it("refuses a corpus frozen for a different runtime provider", async () => {
    const fixture = await createFixture();
    const corpusPath = path.join(fixture.corpora, "alpha-task.json");
    const corpus = JSON.parse(await readFile(corpusPath, "utf8")) as Record<string, unknown>;
    const evidence = (corpus.evidence as Record<string, unknown>[]).map((entry) => ({
      ...entry,
      allowedRuntimeProviders: ["openai", "claude"],
    }));
    const normativeRecords = (corpus.normativeRecords as Record<string, unknown>[]).map(
      (record) => {
        const revision = record.revision as Record<string, unknown>;
        return {
          ...record,
          revision: {
            ...revision,
            egress: {
              ...(revision.egress as Record<string, unknown>),
              allowedRuntimeProviders: ["openai", "claude"],
            },
          },
        };
      },
    );
    await writeFile(
      corpusPath,
      `${JSON.stringify({ ...corpus, runtimeProvider: "claude", evidence, normativeRecords }, null, 2)}\n`,
      "utf8",
    );

    await expect(
      prepareDeepSweEvaluation({
        repositoryRoot: "/repo",
        datasetTasksPath: fixture.tasks,
        corpusRoot: fixture.corpora,
        runDirectory: fixture.run,
        jobsDirectory: path.join(fixture.run, "jobs"),
        pierBinary: "pier",
        model: "openai/test-model",
        reasoningEffort: "medium",
        attempts: 1,
        concurrency: 1,
        sampleSeed: 0,
        arms: ["baseline"],
        taskIds: ["alpha-task"],
        environment: "docker",
        miniSweAgentVersion: "2.1.0",
        deepSweRevision: fixture.revision,
        pierRevision: "pier-0.3.1",
        adapterRevision: "adapter-sha",
      }),
    ).rejects.toThrow(/runtime provider mismatch/);
  });
});

async function createFixture(): Promise<{
  root: string;
  tasks: string;
  corpora: string;
  run: string;
  revision: string;
}> {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-deepswe-prepare-"));
  cleanup.add(root);
  const deepSwe = path.join(root, "deep-swe");
  const tasks = path.join(deepSwe, "tasks");
  const corpora = path.join(root, "corpora");
  const run = path.join(root, "run");
  await mkdir(tasks, { recursive: true });
  await mkdir(corpora, { recursive: true });
  for (const taskId of ["alpha-task", "beta-task"]) {
    const taskPath = path.join(tasks, taskId);
    await mkdir(taskPath, { recursive: true });
    await writeFile(path.join(taskPath, "instruction.md"), `Implement ${taskId}.\n`, "utf8");
    await writeFile(
      path.join(taskPath, "task.toml"),
      `[metadata]\ntask_id = "${taskId}"\nlanguage = "python"\nbase_commit_hash = "${"a".repeat(40)}"\n[environment]\ndocker_image = "example/${taskId}:pinned"\n`,
      "utf8",
    );
    await writeFile(
      path.join(corpora, `${taskId}.json`),
      `${JSON.stringify(fixtureCorpus(taskId), null, 2)}\n`,
      "utf8",
    );
  }
  await execFileAsync("git", ["init", "-q"], { cwd: deepSwe });
  await execFileAsync("git", ["add", "."], { cwd: deepSwe });
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
    { cwd: deepSwe },
  );
  const revision = (
    await execFileAsync("git", ["rev-parse", "HEAD"], { cwd: deepSwe })
  ).stdout.trim();
  return { root, tasks, corpora, run, revision };
}

function required<T>(value: T | undefined): T {
  if (value === undefined) throw new Error("Missing test fixture value");
  return value;
}
