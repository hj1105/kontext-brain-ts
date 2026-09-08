import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import {
  chmod,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readlink,
  realpath,
  rename,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { devNull } from "node:os";
import path from "node:path";
import { subscriptionRuntimeEnvironment } from "./subscription-runtime-environment.js";

export interface CodingWorkspaceSeed {
  readonly workspacePath: string;
  readonly repositoryPath: string;
  readonly codeRevision: string;
}
type Entry = { name: string; mode: "100644" | "100755" | "120000"; bytes: Buffer };
const maxBytes = 64 * 1024 * 1024;
const maxFiles = 10_000;
const revisionPattern = /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/;
const hash = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
const line = (value: Buffer) => value.toString("utf8").replace(/\r?\n$/, "");

/** Captures working files, never the source index; all Git writes stay in private storage. */
export async function prepareCodingWorkspaceSeed(
  dataDirectory: string,
  sourcePath: string,
  environment: NodeJS.ProcessEnv = process.env,
): Promise<CodingWorkspaceSeed> {
  const workspacePath = await realpath(sourcePath);
  const store = path.resolve(dataDirectory, "coding-workspace-seeds", hash(workspacePath));
  if (within(workspacePath, store))
    throw new Error("Coding seed storage must be outside the source workspace");
  await mkdir(store, { recursive: true, mode: 0o700 });
  const canonicalStore = await realpath(store);
  if (within(workspacePath, canonicalStore))
    throw new Error("Coding seed storage resolves inside the source workspace");
  const stage = await mkdtemp(path.join(canonicalStore, ".capture-"));
  await chmod(stage, 0o700);
  const git = runner(dataDirectory, environment);
  try {
    const before = await sourceIdentity(workspacePath, git);
    await git(stage, [
      "init",
      "-q",
      ...(before.base?.length === 64 ? ["--object-format=sha256"] : []),
    ]);
    const entries = await capture(workspacePath, stage, before.git, git);
    const after = await capture(workspacePath, stage, before.git, git);
    if (
      fingerprint(entries) !== fingerprint(after) ||
      JSON.stringify(before) !== JSON.stringify(await sourceIdentity(workspacePath, git))
    ) {
      throw new Error("Coding workspace changed during seed capture; review it again");
    }
    if (before.base) {
      await git(stage, [
        "fetch",
        "--quiet",
        "--depth=1",
        "--no-tags",
        "--no-recurse-submodules",
        workspacePath,
        before.base,
      ]);
    }
    const index: Buffer[] = [];
    for (const entry of entries) {
      const target = path.join(stage, entry.name);
      await mkdir(path.dirname(target), { recursive: true });
      if (entry.mode === "120000") await symlink(entry.bytes.toString("utf8"), target);
      else await writeFile(target, entry.bytes, { mode: entry.mode === "100755" ? 0o755 : 0o644 });
    }
    // Raw blobs bypass source attributes, clean filters and line-ending conversion.
    for (let offset = 0; offset < entries.length; offset += 100) {
      const batch = entries.slice(offset, offset + 100);
      const regular = batch.filter((entry) => entry.mode !== "120000");
      const ids = regular.length
        ? line(
            await git(stage, [
              "hash-object",
              "-w",
              "--no-filters",
              "--",
              ...regular.map((entry) => entry.name),
            ]),
          ).split("\n")
        : [];
      if (ids.length !== regular.length || ids.some((id) => !revisionPattern.test(id)))
        throw new Error("Cannot hash coding seed files");
      for (const entry of batch) {
        const id =
          entry.mode === "120000"
            ? line(await git(stage, ["hash-object", "-w", "--stdin"], entry.bytes))
            : ids[regular.indexOf(entry)];
        index.push(Buffer.from(`${entry.mode} ${id}\t${entry.name}\0`));
      }
    }
    await git(stage, ["update-index", "-z", "--index-info"], Buffer.concat(index));
    const tree = line(await git(stage, ["write-tree"]));
    const codeRevision = line(
      await git(stage, [
        "commit-tree",
        tree,
        ...(before.base ? ["-p", before.base] : []),
        "-m",
        `Kontext reviewed working files v1\n\nCodebase: ${before.codebaseId}`,
      ]),
    );
    if (!revisionPattern.test(codeRevision)) throw new Error("Cannot commit coding workspace seed");
    await git(stage, ["update-ref", "refs/heads/kontext-seed", codeRevision]);
    await git(stage, ["symbolic-ref", "HEAD", "refs/heads/kontext-seed"]);
    await git(stage, ["config", "kontext.seedCodebaseId", before.codebaseId]);
    const repositoryPath = path.join(canonicalStore, codeRevision);
    const seed = { workspacePath, repositoryPath, codeRevision };
    await writeFile(path.join(stage, ".git", "kontext-seed.json"), JSON.stringify(seed), {
      mode: 0o600,
      flag: "wx",
    });
    try {
      await rename(stage, repositoryPath);
    } catch (error) {
      if (
        !error ||
        typeof error !== "object" ||
        !("code" in error) ||
        !["EEXIST", "ENOTEMPTY"].includes(String(error.code))
      )
        throw error;
    }
    await verifyCodingWorkspaceSeed(dataDirectory, seed, environment);
    return seed;
  } finally {
    await rm(stage, { recursive: true, force: true });
  }
}

export async function verifyCodingWorkspaceSeed(
  dataDirectory: string,
  seed: CodingWorkspaceSeed,
  environment: NodeJS.ProcessEnv = process.env,
): Promise<string> {
  if (!revisionPattern.test(seed.codeRevision)) throw new Error("Invalid coding seed revision");
  const store = await realpath(
    path.resolve(dataDirectory, "coding-workspace-seeds", hash(seed.workspacePath)),
  );
  const expected = path.join(store, seed.codeRevision);
  if (
    seed.repositoryPath !== expected ||
    (await realpath(expected)) !== expected ||
    !(await lstat(path.join(expected, ".git"))).isDirectory()
  ) {
    throw new Error("Coding seed repository identity changed");
  }
  await readSeedCodebaseId(expected, path.join(expected, ".git"));
  const saved = JSON.parse(
    await readFile(path.join(expected, ".git", "kontext-seed.json"), {
      encoding: "utf8",
      flag: constants.O_RDONLY | constants.O_NOFOLLOW,
    }),
  );
  if (JSON.stringify(saved) !== JSON.stringify(seed))
    throw new Error("Coding seed ownership changed");
  const git = runner(dataDirectory, environment);
  if (
    (await git(expected, ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"])).length
  )
    throw new Error("Coding seed ignored files changed");
  if (line(await git(expected, ["rev-parse", "HEAD"])) !== seed.codeRevision)
    throw new Error("Coding seed revision changed");
  const listing = await git(expected, ["ls-tree", "-r", "-z", "HEAD"]);
  const entries = await capture(expected, expected, true, git);
  const actual = new Map(entries.map((entry) => [entry.name, entry]));
  const rows = listing.toString("utf8").split("\0").filter(Boolean);
  if (rows.length !== entries.length) throw new Error("Coding seed files changed");
  for (const row of rows) {
    const match = /^(100644|100755|120000) blob ([a-f0-9]+)\t([\s\S]+)$/.exec(row);
    const entry = match && actual.get(match[3] ?? "");
    if (!match || !entry || entry.mode !== match[1]) throw new Error("Coding seed files changed");
    const algorithm = seed.codeRevision.length === 64 ? "sha256" : "sha1";
    const blob = createHash(algorithm)
      .update(`blob ${entry.bytes.length}\0`)
      .update(entry.bytes)
      .digest("hex");
    if (blob !== match[2]) throw new Error("Coding seed files changed");
  }
  return expected;
}

type Git = ReturnType<typeof runner>;
async function sourceIdentity(
  workspacePath: string,
  git: Git,
): Promise<{ git: boolean; base: string | null; codebaseId: string }> {
  try {
    const root = await realpath(line(await git(workspacePath, ["rev-parse", "--show-toplevel"])));
    if (root !== workspacePath) throw new Error("Coding workspace must be the Git root");
  } catch (error) {
    const marker = await lstat(path.join(workspacePath, ".git")).catch(() => null);
    if (marker || !(error instanceof Error) || !error.message.includes("not a git repository"))
      throw error;
    return { git: false, base: null, codebaseId: `codebase:${hash(workspacePath)}` };
  }
  if ((await git(workspacePath, ["ls-files", "--unmerged", "-z"])).length)
    throw new Error("Resolve Git conflicts before capturing a coding seed");
  const base = line(
    await git(
      workspacePath,
      ["rev-parse", "--verify", "--quiet", "HEAD^{commit}"],
      undefined,
      [0, 1],
    ),
  );
  if (base && !revisionPattern.test(base)) throw new Error("Invalid source commit");
  const remote = line(
    await git(workspacePath, ["config", "--get", "remote.origin.url"], undefined, [0, 1]),
  ).trim();
  const common = line(await git(workspacePath, ["rev-parse", "--git-common-dir"])).trim();
  return {
    git: true,
    base: base || null,
    codebaseId: `codebase:${hash(remote || path.resolve(workspacePath, common))}`,
  };
}

/** Keeps copied code in its original Codebase without configuring a push/fetch remote. */
export async function readSeedCodebaseId(
  workspacePath: string,
  commonDirectory: string,
): Promise<string | undefined> {
  const marker = path.join(commonDirectory, "kontext-seed.json");
  const metadata = await lstat(marker).catch((error) => {
    if (error?.code === "ENOENT") return null;
    throw error;
  });
  if (!metadata) return undefined;
  if (!metadata.isFile() || metadata.size > 16 * 1024)
    throw new Error("Coding seed identity is invalid");
  const seed = JSON.parse(
    await readFile(marker, { encoding: "utf8", flag: constants.O_RDONLY | constants.O_NOFOLLOW }),
  );
  if (
    !seed ||
    typeof seed.repositoryPath !== "string" ||
    typeof seed.codeRevision !== "string" ||
    !revisionPattern.test(seed.codeRevision) ||
    path.join(seed.repositoryPath, ".git") !== commonDirectory
  )
    throw new Error("Coding seed identity changed");
  const git = runner("", process.env);
  const identity = line(await git(workspacePath, ["config", "--get", "kontext.seedCodebaseId"]));
  const message = line(await git(workspacePath, ["show", "-s", "--format=%B", seed.codeRevision]));
  if (
    !/^codebase:[a-f0-9]{64}$/.test(identity) ||
    message.trimEnd() !== `Kontext reviewed working files v1\n\nCodebase: ${identity}`
  )
    throw new Error("Coding seed Codebase identity changed");
  return identity;
}

async function capture(
  workspace: string,
  stage: string,
  isGit: boolean,
  git: Git,
): Promise<Entry[]> {
  const names = await git(workspace, [
    ...(!isGit ? [`--git-dir=${path.join(stage, ".git")}`, `--work-tree=${workspace}`] : []),
    "ls-files",
    ...(isGit ? ["--cached"] : []),
    "--others",
    "--exclude-standard",
    "-z",
  ]);
  if (!Buffer.from(names.toString("utf8")).equals(names))
    throw new Error("Coding seed filenames must be valid UTF-8");
  const paths = [...new Set(names.toString("utf8").split("\0").filter(Boolean))].sort();
  if (paths.length > maxFiles) throw new Error("Coding seed exceeds its file-count limit");
  const entries: Entry[] = [];
  let bytes = 0;
  for (const name of paths) {
    if (
      path.isAbsolute(name) ||
      name
        .split(/[\\/]/)
        .some((part) => !part || part === ".." || part === "." || part.toLowerCase() === ".git")
    )
      throw new Error("Unsafe coding seed path");
    const target = path.join(workspace, name);
    const stat = await lstat(target).catch((error) => {
      if (error?.code === "ENOENT") return null;
      throw error;
    });
    if (!stat) continue;
    if (stat.size > maxBytes - bytes) throw new Error("Coding seed exceeds its byte limit");
    if (stat.isSymbolicLink()) {
      const link = await readlink(target);
      if (
        path.isAbsolute(link) ||
        !within(workspace, await realpath(target)) ||
        link.split(/[\\/]/).includes(".git")
      )
        throw new Error("Coding seed symlink escapes the selected workspace");
      const contents = Buffer.from(link);
      bytes += contents.length;
      entries.push({ name, mode: "120000", bytes: contents });
    } else {
      if (!stat.isFile() || !within(workspace, await realpath(target)))
        throw new Error(
          "Coding seeds require regular files or internal relative symlinks; nested repositories are unsupported",
        );
      const contents = await readFile(target, { flag: constants.O_RDONLY | constants.O_NOFOLLOW });
      bytes += contents.length;
      if (bytes > maxBytes) throw new Error("Coding seed exceeds its byte limit");
      entries.push({ name, mode: stat.mode & 0o111 ? "100755" : "100644", bytes: contents });
    }
  }
  return entries;
}
function fingerprint(entries: Entry[]): string {
  return hash(JSON.stringify(entries.map((entry) => [entry.name, entry.mode, hash(entry.bytes)])));
}
function within(root: string, target: string): boolean {
  const relative = path.relative(root, target);
  return (
    relative === "" ||
    (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(`..${path.sep}`))
  );
}
function runner(dataDirectory: string, environment: NodeJS.ProcessEnv) {
  return (cwd: string, args: string[], input?: Buffer, codes = [0]): Promise<Buffer> =>
    new Promise((resolve, reject) => {
      const child = execFile(
        "git",
        [
          "--no-optional-locks",
          "-c",
          "core.fsmonitor=false",
          "-c",
          `core.hooksPath=${devNull}`,
          "-c",
          "core.autocrlf=false",
          "-c",
          "uploadpack.packObjectsHook=",
          ...args,
        ],
        {
          cwd,
          encoding: "buffer",
          timeout: 15_000,
          maxBuffer: 2 * 1024 * 1024,
          windowsHide: true,
          env: {
            ...subscriptionRuntimeEnvironment(dataDirectory, environment),
            LC_ALL: "C",
            GIT_CONFIG_GLOBAL: devNull,
            GIT_CONFIG_NOSYSTEM: "1",
            GIT_TERMINAL_PROMPT: "0",
            GIT_AUTHOR_NAME: "Kontext Brain",
            GIT_COMMITTER_NAME: "Kontext Brain",
            GIT_AUTHOR_EMAIL: "kontext-brain@invalid.local",
            GIT_COMMITTER_EMAIL: "kontext-brain@invalid.local",
            GIT_AUTHOR_DATE: "2000-01-01T00:00:00Z",
            GIT_COMMITTER_DATE: "2000-01-01T00:00:00Z",
          },
        },
        (error, stdout, stderr) => {
          if (error && !codes.includes(Number(error.code)))
            reject(new Error(stderr.toString("utf8").trim() || "Coding seed Git command failed"));
          else resolve(stdout);
        },
      );
      child.stdin?.on("error", () => {});
      child.stdin?.end(input);
    });
}
