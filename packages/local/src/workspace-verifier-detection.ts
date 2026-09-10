import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import type { VerifierKind } from "@kontext-brain/spec";

/**
 * Verifiers a workspace already declares through its own manifests — package
 * scripts, pyproject, go.mod, Cargo.toml, a Makefile — so a project needs no
 * `.kontext/verifiers.json` unless its checks are unusual. The rule that matters
 * is unchanged: the workspace names the command, never the agent.
 */

export interface DetectedVerifier {
  readonly kind: VerifierKind;
  /** Always `workspace:<kind>`, the ref planners and bundles refer to. */
  readonly ref: string;
  readonly command: string;
  readonly args: readonly string[];
  /** Which manifest declared it, for the UI and for diagnostics. */
  readonly origin: string;
}

type Kind = "lint" | "test" | "typecheck" | "build";

const SCRIPT_NAMES: Readonly<Record<Kind, readonly string[]>> = {
  lint: ["lint", "eslint", "biome", "oxlint"],
  test: ["test"],
  typecheck: ["typecheck", "type-check", "check-types", "tsc", "tc"],
  build: ["build"],
};

async function readText(filePath: string): Promise<string | undefined> {
  try {
    return await readFile(filePath, "utf8");
  } catch {
    return undefined;
  }
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await stat(filePath);
    return true;
  } catch {
    return false;
  }
}

function verifier(kind: Kind, command: string, args: readonly string[], origin: string) {
  return { kind, ref: `workspace:${kind}`, command, args, origin } satisfies DetectedVerifier;
}

/** The package manager the lockfile says the project uses; the field wins when present. */
export async function detectPackageManager(
  workspacePath: string,
  packageManagerField: string | undefined,
): Promise<"pnpm" | "yarn" | "bun" | "npm"> {
  const declared = packageManagerField?.split("@", 1)[0];
  if (declared === "pnpm" || declared === "yarn" || declared === "bun" || declared === "npm") {
    return declared;
  }
  if (await exists(path.join(workspacePath, "pnpm-lock.yaml"))) return "pnpm";
  if (await exists(path.join(workspacePath, "yarn.lock"))) return "yarn";
  if (
    (await exists(path.join(workspacePath, "bun.lockb"))) ||
    (await exists(path.join(workspacePath, "bun.lock")))
  ) {
    return "bun";
  }
  return "npm";
}

async function detectNode(workspacePath: string): Promise<DetectedVerifier[]> {
  const raw = await readText(path.join(workspacePath, "package.json"));
  if (raw === undefined) return [];
  let manifest: { packageManager?: unknown; scripts?: unknown };
  try {
    manifest = JSON.parse(raw) as { packageManager?: unknown; scripts?: unknown };
  } catch {
    return [];
  }
  const scripts =
    typeof manifest.scripts === "object" && manifest.scripts !== null
      ? (manifest.scripts as Record<string, unknown>)
      : {};
  const manager = await detectPackageManager(
    workspacePath,
    typeof manifest.packageManager === "string" ? manifest.packageManager : undefined,
  );
  const found: DetectedVerifier[] = [];
  for (const kind of Object.keys(SCRIPT_NAMES) as Kind[]) {
    const script = SCRIPT_NAMES[kind].find((name) => typeof scripts[name] === "string");
    if (!script) continue;
    found.push(verifier(kind, manager, ["run", script], `package.json scripts.${script}`));
  }
  return found;
}

async function detectPython(workspacePath: string): Promise<DetectedVerifier[]> {
  const pyproject = (await readText(path.join(workspacePath, "pyproject.toml"))) ?? "";
  const hasProject =
    pyproject !== "" ||
    (await exists(path.join(workspacePath, "setup.py"))) ||
    (await exists(path.join(workspacePath, "setup.cfg")));
  if (!hasProject) return [];
  const found: DetectedVerifier[] = [];
  const pytestConfigured =
    /\[tool\.pytest/.test(pyproject) ||
    (await exists(path.join(workspacePath, "pytest.ini"))) ||
    (await exists(path.join(workspacePath, "tests"))) ||
    (await exists(path.join(workspacePath, "test")));
  if (pytestConfigured) {
    found.push(verifier("test", "python3", ["-m", "pytest", "-q"], "pytest"));
  }
  if (/\[tool\.ruff/.test(pyproject) || (await exists(path.join(workspacePath, "ruff.toml")))) {
    found.push(verifier("lint", "ruff", ["check", "."], "ruff"));
  }
  if (/\[tool\.mypy/.test(pyproject) || (await exists(path.join(workspacePath, "mypy.ini")))) {
    found.push(verifier("typecheck", "python3", ["-m", "mypy", "."], "mypy"));
  }
  return found;
}

async function detectGo(workspacePath: string): Promise<DetectedVerifier[]> {
  if (!(await exists(path.join(workspacePath, "go.mod")))) return [];
  return [
    verifier("typecheck", "go", ["vet", "./..."], "go.mod"),
    verifier("test", "go", ["test", "./..."], "go.mod"),
    verifier("build", "go", ["build", "./..."], "go.mod"),
  ];
}

async function detectRust(workspacePath: string): Promise<DetectedVerifier[]> {
  if (!(await exists(path.join(workspacePath, "Cargo.toml")))) return [];
  return [
    verifier("lint", "cargo", ["clippy", "--all-targets"], "Cargo.toml"),
    verifier("test", "cargo", ["test"], "Cargo.toml"),
    verifier("build", "cargo", ["build"], "Cargo.toml"),
  ];
}

async function detectMake(workspacePath: string): Promise<DetectedVerifier[]> {
  const makefile =
    (await readText(path.join(workspacePath, "Makefile"))) ??
    (await readText(path.join(workspacePath, "makefile")));
  if (makefile === undefined) return [];
  const targets = new Set(
    Array.from(makefile.matchAll(/^([A-Za-z][\w.-]*)\s*:(?!=)/gm), (match) => match[1] ?? ""),
  );
  const found: DetectedVerifier[] = [];
  for (const kind of ["lint", "test", "typecheck", "build"] as const) {
    if (targets.has(kind)) found.push(verifier(kind, "make", [kind], `Makefile ${kind}`));
  }
  return found;
}

/**
 * Manifest-declared verifiers, one per kind. The first ecosystem that declares a
 * kind wins in the order Node, Python, Go, Rust, Make, which matches how a mixed
 * repository is usually driven.
 */
export async function detectWorkspaceVerifiers(
  workspacePath: string,
): Promise<readonly DetectedVerifier[]> {
  const byKind = new Map<string, DetectedVerifier>();
  for (const detector of [detectNode, detectPython, detectGo, detectRust, detectMake]) {
    for (const found of await detector(workspacePath)) {
      if (!byKind.has(found.kind)) byKind.set(found.kind, found);
    }
  }
  return Array.from(byKind.values());
}

/**
 * Installed-dependency directories a runtime worktree needs linked from the
 * source checkout, inferred from the manifests when the workspace declares none.
 */
export async function defaultLinkedDirectories(workspacePath: string): Promise<readonly string[]> {
  const candidates: string[] = [];
  if (await exists(path.join(workspacePath, "package.json"))) candidates.push("node_modules");
  for (const venv of [".venv", "venv"]) {
    if (await exists(path.join(workspacePath, venv))) candidates.push(venv);
  }
  return candidates;
}
