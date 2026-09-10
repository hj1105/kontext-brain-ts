import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  defaultLinkedDirectories,
  detectWorkspaceVerifiers,
  listDeclaredWorkspaceVerifiers,
  readWorkspaceLinkedDirectories,
} from "../src/index.js";

const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});

async function workspace(files: Record<string, string>, directories: string[] = []) {
  const root = await mkdtemp(path.join(tmpdir(), "kontext-verifier-detect-"));
  roots.push(root);
  for (const directory of directories) await mkdir(path.join(root, directory), { recursive: true });
  for (const [name, content] of Object.entries(files)) {
    await mkdir(path.dirname(path.join(root, name)), { recursive: true });
    await writeFile(path.join(root, name), content);
  }
  return root;
}

const byKind = (found: readonly { kind: string; command: string; args: readonly string[] }[]) =>
  Object.fromEntries(found.map((v) => [v.kind, `${v.command} ${v.args.join(" ")}`]));

describe("detectWorkspaceVerifiers", () => {
  it("reads a Node project's scripts and picks the package manager from its lockfile", async () => {
    const root = await workspace({
      "package.json": JSON.stringify({
        scripts: {
          test: "vitest",
          "type-check": "tsc -p .",
          eslint: "eslint .",
          build: "vite build",
        },
      }),
      "pnpm-lock.yaml": "",
    });
    expect(byKind(await detectWorkspaceVerifiers(root))).toEqual({
      lint: "pnpm run eslint",
      test: "pnpm run test",
      typecheck: "pnpm run type-check",
      build: "pnpm run build",
    });
  });

  it("honours an explicit packageManager field over the lockfile", async () => {
    const root = await workspace({
      "package.json": JSON.stringify({ packageManager: "yarn@4.1.0", scripts: { test: "jest" } }),
      "pnpm-lock.yaml": "",
    });
    expect(byKind(await detectWorkspaceVerifiers(root))).toEqual({ test: "yarn run test" });
  });

  it("recognises Python, Go, Rust and Makefile projects without any Kontext file", async () => {
    const python = await workspace(
      { "pyproject.toml": "[tool.ruff]\nline-length = 100\n[tool.mypy]\nstrict = true\n" },
      ["tests"],
    );
    expect(byKind(await detectWorkspaceVerifiers(python))).toEqual({
      test: "python3 -m pytest -q",
      lint: "ruff check .",
      typecheck: "python3 -m mypy .",
    });
    const go = await workspace({ "go.mod": "module example.com/x\n" });
    expect(byKind(await detectWorkspaceVerifiers(go))).toEqual({
      typecheck: "go vet ./...",
      test: "go test ./...",
      build: "go build ./...",
    });
    const rust = await workspace({ "Cargo.toml": '[package]\nname = "x"\n' });
    expect(Object.keys(byKind(await detectWorkspaceVerifiers(rust))).sort()).toEqual([
      "build",
      "lint",
      "test",
    ]);
    const make = await workspace({
      Makefile:
        "VAR := 1\n\ntest:\n\tgo test ./...\n\nlint:\n\tgolangci-lint run\n\nclean:\n\trm -rf out\n",
    });
    expect(byKind(await detectWorkspaceVerifiers(make))).toEqual({
      lint: "make lint",
      test: "make test",
    });
  });

  it("lets the first ecosystem that declares a kind win, per kind", async () => {
    const root = await workspace({
      "package.json": JSON.stringify({ scripts: { test: "vitest" } }),
      Makefile: "lint:\n\tshellcheck *.sh\n",
    });
    expect(byKind(await detectWorkspaceVerifiers(root))).toEqual({
      test: "npm run test",
      lint: "make lint",
    });
  });

  it("finds nothing in a directory without manifests", async () => {
    const root = await workspace({ "README.md": "# hi\n" });
    expect(await detectWorkspaceVerifiers(root)).toEqual([]);
  });
});

describe("zero-config defaults reach the planner and the runtime worktree", () => {
  it("lists detected verifiers beside declared ones", async () => {
    const root = await workspace({
      "package.json": JSON.stringify({ scripts: { lint: "oxlint" } }),
      ".kontext/verifiers.json": JSON.stringify({
        schemaVersion: 1,
        verifiers: [{ kind: "test", ref: "e2e", command: "npx", args: ["playwright", "test"] }],
      }),
    });
    expect(await listDeclaredWorkspaceVerifiers(root)).toEqual([
      { kind: "test", ref: "e2e" },
      { kind: "lint", ref: "workspace:lint" },
    ]);
  });

  it("links installed dependencies into runtime worktrees without being asked", async () => {
    const root = await workspace({ "package.json": "{}" }, [".venv"]);
    expect(await defaultLinkedDirectories(root)).toEqual(["node_modules", ".venv"]);
    expect(await readWorkspaceLinkedDirectories(root)).toEqual(["node_modules", ".venv"]);
    const declared = await workspace({
      "package.json": "{}",
      ".kontext/verifiers.json": JSON.stringify({
        schemaVersion: 1,
        verifiers: [],
        linkedDirectories: ["vendor/bundle", "node_modules"],
      }),
    });
    expect(await readWorkspaceLinkedDirectories(declared)).toEqual([
      "vendor/bundle",
      "node_modules",
    ]);
  });
});
