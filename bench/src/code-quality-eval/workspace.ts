import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";
import type { CodeQualityScenario, HiddenEvaluationResult } from "./contracts.js";

const execFileAsync = promisify(execFile);

export interface ScenarioWorkspace {
  readonly workspacePath: string;
  readonly baseRevision: string;
}

export interface WorkspaceCommandResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
}

export async function createScenarioWorkspace(
  scenario: CodeQualityScenario,
): Promise<ScenarioWorkspace> {
  const workspacePath = await mkdtemp(
    path.join(tmpdir(), `kontext-code-eval-${scenario.scenarioId}-`),
  );
  await mkdir(path.join(workspacePath, "src"), { recursive: true });
  await mkdir(path.join(workspacePath, "test"), { recursive: true });
  await writeFile(
    path.join(workspacePath, "package.json"),
    `${JSON.stringify(
      {
        name: `code-quality-${scenario.scenarioId}`,
        version: "1.0.0",
        private: true,
        type: "module",
        scripts: { test: testCommand(scenario) },
      },
      null,
      2,
    )}\n`,
  );
  await writeFile(path.join(workspacePath, scenario.sourceFile), scenario.initialSource);
  await writeFile(path.join(workspacePath, testFileFor(scenario)), scenario.publicTestSource);
  await writeFile(
    path.join(workspacePath, "TASK.md"),
    `# Task\n\n${scenario.publicPrompt}\n\nOnly edit \`${scenario.sourceFile}\`.\n`,
  );
  await git(workspacePath, ["init", "-q"]);
  await git(workspacePath, ["add", "."]);
  await git(workspacePath, [
    "-c",
    "user.name=Kontext Code Eval",
    "-c",
    "user.email=code-eval@example.invalid",
    "commit",
    "-qm",
    "fixture baseline",
  ]);
  return {
    workspacePath,
    baseRevision: (await git(workspacePath, ["rev-parse", "HEAD"])).stdout.trim(),
  };
}

export function isPythonScenario(scenario: CodeQualityScenario): boolean {
  return /\.py$/i.test(scenario.sourceFile);
}

export function testFileFor(scenario: CodeQualityScenario): string {
  return (
    scenario.testFile ??
    (isPythonScenario(scenario) ? "test/public_test.py" : "test/public.test.js")
  );
}

function testCommand(scenario: CodeQualityScenario): string {
  // Python has no npm test runner, so the workspace script shells out to
  // unittest while keeping one uniform `npm test` entry point for the agent.
  // unittest discovery needs the start directory to be an importable package,
  // so the workspace runs the test module directly instead.
  return isPythonScenario(scenario) ? `python3 ${testFileFor(scenario)}` : "node --test";
}

export async function evaluateWorkspace(
  scenario: CodeQualityScenario,
  workspacePath: string,
): Promise<{
  readonly publicTestsPassed: boolean;
  readonly hidden: HiddenEvaluationResult;
  readonly source: string;
  readonly changedPaths: readonly string[];
  readonly patch: string;
}> {
  // --test-reporter is a node --test flag; unittest rejects it as an unknown
  // argument and the public test would look like a failure.
  const publicTests = await runWorkspaceCommand(
    workspacePath,
    "npm",
    isPythonScenario(scenario) ? ["test"] : ["test", "--", "--test-reporter=spec"],
  );
  const sourcePath = path.join(workspacePath, scenario.sourceFile);
  const source = await readFile(sourcePath, "utf8").catch(() => "");
  let hidden: HiddenEvaluationResult;
  try {
    hidden = isPythonScenario(scenario)
      ? await evaluatePythonChecks(scenario, workspacePath)
      : await (async () => {
          const moduleUrl = `${pathToFileURL(sourcePath).href}?eval=${Date.now()}-${Math.random()}`;
          const loaded = (await import(moduleUrl)) as Readonly<Record<string, unknown>>;
          return scenario.evaluateHidden(loaded);
        })();
  } catch (error) {
    hidden = {
      assertions: [
        {
          assertionId: "module-load",
          passed: false,
          diagnostic: error instanceof Error ? error.message : String(error),
        },
      ],
    };
  }
  const changedPaths = (await git(workspacePath, ["status", "--short"])).stdout
    .split("\n")
    .map((line) => line.slice(3).trim())
    .filter(Boolean)
    .sort();
  const patch = (await git(workspacePath, ["diff", "--binary", "HEAD", "--"])).stdout;
  return {
    publicTestsPassed: publicTests.exitCode === 0,
    hidden,
    source,
    changedPaths,
    patch,
  };
}

/**
 * Held-out checks for a Python scenario are declared as data and executed by a
 * driver inside the workspace, because the evaluator cannot import a Python
 * module the way it imports a JavaScript one. The driver is written outside the
 * repository tree under test and removed afterwards so it never appears as a
 * changed path.
 */
async function evaluatePythonChecks(
  scenario: CodeQualityScenario,
  workspacePath: string,
): Promise<HiddenEvaluationResult> {
  const checks = scenario.hiddenChecks;
  if (!checks || checks.length === 0) {
    throw new Error(`Python scenario ${scenario.scenarioId} declares no hidden checks`);
  }
  const driverDirectory = await mkdtemp(path.join(tmpdir(), "kontext-python-driver-"));
  const driverPath = path.join(driverDirectory, "driver.py");
  const specPath = path.join(driverDirectory, "checks.json");
  const moduleDirectory = path.join(workspacePath, path.dirname(scenario.sourceFile));
  const moduleName = path.basename(scenario.sourceFile).replace(/\.py$/i, "");
  await writeFile(specPath, JSON.stringify({ checks }), "utf8");
  await writeFile(driverPath, pythonDriverSource, "utf8");
  try {
    const result = await runWorkspaceCommand(workspacePath, "python3", [
      driverPath,
      moduleDirectory,
      moduleName,
      specPath,
    ]);
    const parsed = JSON.parse(result.stdout) as HiddenEvaluationResult;
    return parsed;
  } finally {
    await rm(driverDirectory, { recursive: true, force: true });
  }
}

const pythonDriverSource = `import importlib.util, json, sys

module_directory, module_name, spec_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, module_directory)
with open(spec_path) as handle:
    checks = json.load(handle)["checks"]

assertions = []
try:
    spec = importlib.util.spec_from_file_location(
        module_name, module_directory + "/" + module_name + ".py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
except Exception as error:
    print(json.dumps({"assertions": [
        {"assertionId": "module-load", "passed": False,
         "diagnostic": type(error).__name__ + ": " + str(error)}
    ]}))
    sys.exit(0)

for check in checks:
    assertion_id = check["assertionId"]
    try:
        function = getattr(module, check["functionName"])
    except AttributeError:
        assertions.append({"assertionId": assertion_id, "passed": False,
                           "diagnostic": "Missing function " + check["functionName"]})
        continue
    expects_throw = check.get("throws")
    try:
        actual = function(*check["args"])
        if expects_throw:
            assertions.append({"assertionId": assertion_id, "passed": False,
                               "diagnostic": "Expected " + expects_throw})
        elif actual == check.get("expected"):
            assertions.append({"assertionId": assertion_id, "passed": True})
        else:
            assertions.append({"assertionId": assertion_id, "passed": False,
                               "diagnostic": "Expected " + json.dumps(check.get("expected")) +
                                             ", received " + repr(actual)})
    except Exception as error:
        name = type(error).__name__
        if expects_throw and name == expects_throw:
            assertions.append({"assertionId": assertion_id, "passed": True})
        else:
            assertions.append({"assertionId": assertion_id, "passed": False,
                               "diagnostic": name + ": " + str(error)})

print(json.dumps({"assertions": assertions}))
`;

export async function runWorkspaceCommand(
  cwd: string,
  command: string,
  args: readonly string[],
  environment: NodeJS.ProcessEnv = process.env,
): Promise<WorkspaceCommandResult> {
  try {
    const result = await execFileAsync(command, args, {
      cwd,
      env: environment,
      encoding: "utf8",
      maxBuffer: 10 * 1024 * 1024,
    });
    return { exitCode: 0, stdout: result.stdout, stderr: result.stderr };
  } catch (error) {
    if (isExecError(error)) {
      return {
        exitCode: typeof error.code === "number" ? error.code : 1,
        stdout: error.stdout ?? "",
        stderr: error.stderr ?? error.message,
      };
    }
    throw error;
  }
}

async function git(cwd: string, args: readonly string[]): Promise<WorkspaceCommandResult> {
  return runWorkspaceCommand(cwd, "git", args);
}

function isExecError(error: unknown): error is Error & {
  readonly code?: number | string;
  readonly stdout?: string;
  readonly stderr?: string;
} {
  return error instanceof Error;
}
