import { execFile } from "node:child_process";
import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { publicIssue } from "./documents.js";
import {
  type GeneratedRepository,
  expectedDecoyDelay,
  expectedGovernedDelay,
  generateRepository,
  governedPolicy,
  subsystems,
} from "./generator.js";

const execFileAsync = promisify(execFile);

export interface LargeScaleWorkspace {
  readonly workspacePath: string;
  readonly baseRevision: string;
  readonly repository: GeneratedRepository;
}

/**
 * The public test only covers failureIndex 0, where every factor agrees, so a
 * no-op change still passes it. That keeps the policy underdetermined by the
 * public surface exactly as the single-function scenarios did.
 */
function publicTestSource(repository: GeneratedRepository): string {
  const first = repository.functions.find((item) => item.governed);
  if (!first) throw new Error("The generated repository has no governed function");
  return `import assert from "node:assert/strict";
import test from "node:test";
import { ${first.name} } from "../${first.file}";

test("the first attempt uses the base delay", () => {
  assert.equal(${first.name}(0, 100), 100);
});
`;
}

/**
 * Every decoy keeps a test of its own. Editing one is therefore caught by the
 * repository's own suite, independently of any diff analysis.
 */
function decoyTestSource(repository: GeneratedRepository): string {
  const cases = repository.functions
    .filter((item) => !item.governed)
    .map(
      (item) =>
        `import { ${item.name} } from "../../${item.file}";\ntest("${item.name} keeps its approved backoff", () => {\n  assert.equal(${item.name}(2, 100), ${expectedDecoyDelay(2, 100)});\n  assert.equal(${item.name}(9, 100), ${expectedDecoyDelay(9, 100)});\n});`,
    )
    .join("\n\n");
  return `import assert from "node:assert/strict";
import test from "node:test";

${cases}
`;
}

export async function createLargeScaleWorkspace(): Promise<LargeScaleWorkspace> {
  const repository = generateRepository();
  const workspacePath = await mkdtemp(path.join(tmpdir(), "kontext-large-scale-"));

  for (const [file, contents] of repository.files) {
    await mkdir(path.join(workspacePath, path.dirname(file)), { recursive: true });
    await writeFile(path.join(workspacePath, file), contents);
  }
  await mkdir(path.join(workspacePath, "test"), { recursive: true });
  await writeFile(
    path.join(workspacePath, "package.json"),
    `${JSON.stringify(
      {
        name: "kontext-large-scale-fixture",
        version: "1.0.0",
        private: true,
        type: "module",
        scripts: { test: "node --test" },
      },
      null,
      2,
    )}\n`,
  );
  await writeFile(path.join(workspacePath, "test", "public.test.js"), publicTestSource(repository));
  await mkdir(path.join(workspacePath, "test", "regression"), { recursive: true });
  await writeFile(
    path.join(workspacePath, "test", "regression", "decoys.test.js"),
    decoyTestSource(repository),
  );
  await writeFile(
    path.join(workspacePath, "TASK.md"),
    `# Issue #${publicIssue.number}: ${publicIssue.title}\n\n${publicIssue.body}\nRun \`npm test\` before finishing.\n`,
  );

  await git(workspacePath, ["init", "-q"]);
  await git(workspacePath, ["add", "."]);
  await git(workspacePath, [
    "-c",
    "user.name=Kontext Large Scale",
    "-c",
    "user.email=large-scale@example.invalid",
    "commit",
    "-qm",
    "fixture baseline",
  ]);
  const baseRevision = (await git(workspacePath, ["rev-parse", "HEAD"])).stdout.trim();
  return { workspacePath, baseRevision, repository };
}

export interface LargeScaleGrade {
  readonly targetRecall: number;
  readonly collateralPrecision: number;
  readonly governedChanged: readonly string[];
  readonly decoysChanged: readonly string[];
  readonly hiddenPassed: number;
  readonly hiddenTotal: number;
  readonly regressionFailures: number;
  readonly publicTestsPassed: boolean;
  readonly canonicalTermPresent: boolean;
  /** The ceiling is defined once in a shared module and imported, not copied. */
  readonly sharedConstantHonoured: boolean;
  readonly constantDefinitionCount: number;
  readonly hiddenFailures: readonly string[];
  readonly changedFiles: readonly string[];
  readonly patch: string;
}

/**
 * Grades from the generator's manifest and `git diff`, never from the product's
 * own symbol observer. Scoring the product with the product would make symbol
 * identity both the instrument and the thing measured.
 */
export async function gradeLargeScaleWorkspace(
  workspace: LargeScaleWorkspace,
): Promise<LargeScaleGrade> {
  const changedFiles = new Set(
    (await git(workspace.workspacePath, ["diff", "--name-only", "HEAD", "--"])).stdout
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean),
  );
  const untracked = (
    await git(workspace.workspacePath, ["ls-files", "--others", "--exclude-standard"])
  ).stdout
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  for (const file of untracked) changedFiles.add(file);

  const governedChanged = workspace.repository.functions
    .filter((item) => item.governed && changedFiles.has(item.file))
    .map((item) => item.name);
  const decoysChanged = workspace.repository.functions
    .filter((item) => !item.governed && changedFiles.has(item.file))
    .map((item) => item.name);

  const governedTotal = workspace.repository.governedNames.length;
  const changedTotal = governedChanged.length + decoysChanged.length;

  const hidden = await evaluateHiddenBehaviour(workspace);
  const publicTests = await run(workspace.workspacePath, "npm", [
    "test",
    "--",
    "--test-reporter=dot",
  ]);
  const structure = await inspectConstantStructure(workspace);
  const changedFileList = [...changedFiles].sort();
  const trackedPatch = (await git(workspace.workspacePath, ["diff", "--binary", "HEAD", "--"]))
    .stdout;
  const untrackedPatch = await renderUntrackedFiles(workspace.workspacePath, untracked);

  return {
    targetRecall: governedTotal === 0 ? 0 : governedChanged.length / governedTotal,
    // An arm that rewrites everything reaches full recall, so precision is what
    // stops that from scoring well.
    collateralPrecision: changedTotal === 0 ? 1 : governedChanged.length / changedTotal,
    governedChanged,
    decoysChanged,
    hiddenPassed: hidden.passed,
    hiddenTotal: hidden.total,
    regressionFailures: hidden.regressionFailures,
    publicTestsPassed: publicTests.exitCode === 0,
    canonicalTermPresent: structure.definitions > 0 || structure.referencingFiles > 0,
    sharedConstantHonoured: structure.definitions === 1 && structure.referencingFiles >= 2,
    constantDefinitionCount: structure.definitions,
    hiddenFailures: hidden.failures,
    changedFiles: changedFileList,
    patch: [trackedPatch.trimEnd(), untrackedPatch.trimEnd()].filter(Boolean).join("\n\n"),
  };
}

async function renderUntrackedFiles(
  workspacePath: string,
  files: readonly string[],
): Promise<string> {
  const { readFile } = await import("node:fs/promises");
  const rendered: string[] = [];
  for (const file of files) {
    const contents = await readFile(path.join(workspacePath, file), "utf8").catch(() => "");
    rendered.push(
      `diff --git a/${file} b/${file}\nnew file mode 100644\n--- /dev/null\n+++ b/${file}\n${contents}`,
    );
  }
  return rendered.join("\n\n");
}

/**
 * Counts where the ceiling constant is declared and how many files reference it.
 * One declaration with several referencing files means the shared module was
 * created; eight declarations means the constant was copied instead.
 */
async function inspectConstantStructure(
  workspace: LargeScaleWorkspace,
): Promise<{ definitions: number; referencingFiles: number }> {
  const { readFile } = await import("node:fs/promises");
  const tracked = (
    await git(workspace.workspacePath, ["ls-files", "--cached", "--others", "--exclude-standard"])
  ).stdout
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.endsWith(".js"));
  const declaration = new RegExp(`(?:const|let|var)\\s+${governedPolicy.constantName}\\s*=`);
  let definitions = 0;
  let referencingFiles = 0;
  for (const file of tracked) {
    const contents = await readFile(path.join(workspace.workspacePath, file), "utf8").catch(
      () => "",
    );
    if (!contents.includes(governedPolicy.constantName)) continue;
    referencingFiles += 1;
    if (declaration.test(contents)) definitions += 1;
  }
  return { definitions, referencingFiles };
}

/**
 * Runs the held-out behaviour checks out of process. Each governed function must
 * follow the new policy and each decoy must keep the old one.
 */
async function evaluateHiddenBehaviour(workspace: LargeScaleWorkspace): Promise<{
  passed: number;
  total: number;
  regressionFailures: number;
  failures: readonly string[];
}> {
  const checks = workspace.repository.functions.flatMap((item) =>
    [
      { failureIndex: 1, baseMs: 100 },
      { failureIndex: 4, baseMs: 100 },
      { failureIndex: 12, baseMs: 100 },
    ].map((input) => ({
      file: item.file,
      name: item.name,
      governed: item.governed,
      ...input,
      expected: item.governed
        ? expectedGovernedDelay(input.failureIndex, input.baseMs)
        : expectedDecoyDelay(input.failureIndex, input.baseMs),
    })),
  );

  const script = `
    const checks = ${JSON.stringify(checks)};
    let passed = 0;
    let regressionFailures = 0;
    const failures = [];
    for (const check of checks) {
      try {
        const loaded = await import(${JSON.stringify(`${pathToFileUrlPrefix(workspace.workspacePath)}/`)} + check.file);
        const actual = loaded[check.name](check.failureIndex, check.baseMs);
        if (actual === check.expected) passed += 1;
        else {
          failures.push(check.name + "(" + check.failureIndex + "): expected " + check.expected + " got " + actual);
          if (!check.governed) regressionFailures += 1;
        }
      } catch (error) {
        failures.push(check.name + ": " + String(error && error.message));
        if (!check.governed) regressionFailures += 1;
      }
    }
    process.stdout.write(JSON.stringify({ passed, total: checks.length, regressionFailures, failures: failures.slice(0, 8) }));
  `;
  const { stdout } = await execFileAsync(process.execPath, ["--input-type=module", "-e", script], {
    cwd: workspace.workspacePath,
    encoding: "utf8",
    maxBuffer: 20 * 1024 * 1024,
  });
  return JSON.parse(stdout) as {
    passed: number;
    total: number;
    regressionFailures: number;
    failures: readonly string[];
  };
}

function pathToFileUrlPrefix(workspacePath: string): string {
  return `file://${workspacePath}`;
}

async function run(
  cwd: string,
  command: string,
  args: readonly string[],
): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  try {
    const result = await execFileAsync(command, args, {
      cwd,
      encoding: "utf8",
      maxBuffer: 20 * 1024 * 1024,
    });
    return { exitCode: 0, stdout: result.stdout, stderr: result.stderr };
  } catch (error) {
    const failure = error as { code?: number; stdout?: string; stderr?: string; message: string };
    return {
      exitCode: typeof failure.code === "number" ? failure.code : 1,
      stdout: failure.stdout ?? "",
      stderr: failure.stderr ?? failure.message,
    };
  }
}

async function git(cwd: string, args: readonly string[]) {
  return run(cwd, "git", args);
}

export { subsystems };
