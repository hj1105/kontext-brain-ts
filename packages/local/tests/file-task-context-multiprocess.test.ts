import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { expect, it } from "vitest";
import { FileTaskContextRepository } from "../src/file-task-context-repository.js";

it("allows only one conditional publication across independent sidecar processes", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-task-process-cas-"));
  const workers: ReturnType<typeof startWorker>[] = [];
  try {
    const taskId = "task:process-race";
    const initial = {
      codeRevision: "revision:initial",
      sourceFreshnessDigest: "freshness:one",
      effectiveScopes: [],
      normativeRecords: [],
      normativeRevisionCatalog: [],
      conflicts: [],
      evidence: [],
      logicPlans: [],
    };
    const repository = new FileTaskContextRepository(directory);
    const version = await repository.publishCurrent(taskId, initial);
    await writeFile(
      path.join(directory, "request.json"),
      JSON.stringify({ initial, taskId, digest: version.digest }),
    );
    const entry = path.join(directory, "cas-worker.mjs");
    await build({
      stdin: {
        contents: `import { FileTaskContextRepository } from ${JSON.stringify(fileURLToPath(new URL("../src/file-task-context-repository.ts", import.meta.url)))};
          import { readFile } from 'node:fs/promises';
          import path from 'node:path';
          const directory = process.argv[2];
          const request = JSON.parse(await readFile(path.join(directory, 'request.json'), 'utf8'));
          process.stdout.write('ready\\n');
          await new Promise(resolve => process.stdin.once('data', resolve));
          try {
            await new FileTaskContextRepository(directory).publishCurrent(request.taskId,
              { ...request.initial, codeRevision: 'revision:' + process.pid }, { expectedDigest: request.digest });
            process.stdout.write('accepted\\n');
          } catch (error) {
            if (!error.message.includes('changed since it was read')) throw error;
            process.stdout.write('conflict\\n');
          }
          process.stdin.destroy();`,
        resolveDir: process.cwd(),
      },
      outfile: entry,
      bundle: true,
      platform: "node",
      format: "esm",
      banner: {
        js: 'import { createRequire } from "node:module"; import { fileURLToPath } from "node:url"; import { dirname } from "node:path"; const require = createRequire(import.meta.url); const __filename = fileURLToPath(import.meta.url); const __dirname = dirname(__filename);',
      },
      logLevel: "silent",
    });
    for (let index = 0; index < 6; index++) workers.push(startWorker(entry, directory));
    await Promise.all(workers.map((worker) => worker.ready));
    for (const worker of workers) worker.child.stdin.end("publish\n");
    const results = await Promise.all(workers.map((worker) => worker.result));
    expect(results.filter((result) => result.includes("accepted"))).toHaveLength(1);
    expect(results.filter((result) => result.includes("conflict"))).toHaveLength(5);
    const winner = workers.find((_, index) => results[index]?.includes("accepted"));
    expect((await repository.getCurrent(taskId)).codeRevision).toBe(
      `revision:${winner?.child.pid}`,
    );
  } finally {
    for (const worker of workers) worker.child.kill("SIGKILL");
    await Promise.allSettled(workers.map((worker) => worker.result));
    await rm(directory, { recursive: true, force: true });
  }
}, 15_000);

function startWorker(entry: string, directory: string) {
  const child = spawn(process.execPath, [entry, directory], {
    env: {
      HOME: directory,
      USERPROFILE: directory,
      PATH: "",
      ...(process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}),
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  let output = "";
  let diagnostic = "";
  let acceptReady!: () => void;
  let rejectReady!: (error: Error) => void;
  const ready = new Promise<void>((resolve, reject) => {
    acceptReady = resolve;
    rejectReady = reject;
  });
  const timeout = setTimeout(() => {
    rejectReady(new Error("Worker did not become ready"));
    child.kill("SIGKILL");
  }, 5_000);
  child.stdout.on("data", (chunk) => {
    output += chunk.toString();
    if (output.includes("ready\n")) acceptReady();
  });
  child.stderr.on("data", (chunk) => {
    diagnostic += chunk.toString();
  });
  const result = new Promise<string>((resolve, reject) => {
    child.once("error", (error) => {
      rejectReady(error);
      reject(error);
    });
    child.once("close", (code) => {
      clearTimeout(timeout);
      if (code === 0) resolve(output);
      else {
        const error = new Error(`Worker exited ${code}: ${diagnostic}`);
        rejectReady(error);
        reject(error);
      }
    });
  });
  // Readiness can fail before the caller attaches its outcome wait.
  void result.catch(() => undefined);
  return { child, ready, result };
}
