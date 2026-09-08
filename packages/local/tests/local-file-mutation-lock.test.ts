import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { afterEach, describe, expect, it, vi } from "vitest";
import { withLocalFileMutationLock } from "../src/local-file-mutation-lock.js";

const directories: string[] = [];
async function fixture() {
  const directory = await mkdtemp(path.join(tmpdir(), "kontext-file-lock-"));
  directories.push(directory);
  return { directory, lock: path.join(directory, "context.lock") };
}
afterEach(async () => {
  await Promise.all(
    directories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })),
  );
});
describe("local file mutation serialization", () => {
  it("serializes independent callers without overlapping their mutations", async () => {
    const { lock } = await fixture();
    let active = 0;
    let peak = 0;
    const values = await Promise.all(
      Array.from({ length: 24 }, (_, index) =>
        withLocalFileMutationLock(lock, async () => {
          active++;
          peak = Math.max(peak, active);
          await new Promise((resolve) => setTimeout(resolve, 2));
          active--;
          return index;
        }),
      ),
    );
    expect(peak).toBe(1);
    expect(values).toHaveLength(24);
  });

  it.each(["ENOENT", "EINVAL"])(
    "does not replay a mutation that throws %s and releases its ownership",
    async (code) => {
      const { lock } = await fixture();
      const failure = Object.assign(new Error("Selected source failure"), { code });
      const mutation = vi.fn(async () => {
        throw failure;
      });
      await expect(withLocalFileMutationLock(lock, mutation)).rejects.toBe(failure);
      expect(mutation).toHaveBeenCalledTimes(1);
      expect(await withLocalFileMutationLock(lock, async () => "next")).toBe("next");
    },
  );

  it("reclaims an empty initialization directory but preserves unrecognized owner data", async () => {
    const { lock } = await fixture();
    await mkdir(lock);
    expect(await withLocalFileMutationLock(lock, async () => "recovered")).toBe("recovered");
    await mkdir(lock);
    await writeFile(path.join(lock, "unknown-owner"), "preserve this");
    await expect(withLocalFileMutationLock(lock, async () => "unsafe")).rejects.toThrow(
      "Unrecognized",
    );
    expect(await readFile(path.join(lock, "unknown-owner"), "utf8")).toBe("preserve this");
  });

  it("never expires a live owner just because another writer waited too long", async () => {
    const { lock } = await fixture();
    let entered!: () => void;
    let finish!: () => void;
    const ready = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const held = withLocalFileMutationLock(lock, async () => {
      entered();
      await new Promise<void>((resolve) => {
        finish = resolve;
      });
    });
    await ready;
    const competing = vi.fn(async () => "must not execute");
    try {
      await expect(withLocalFileMutationLock(lock, competing)).rejects.toThrow("busy");
      expect(competing).not.toHaveBeenCalled();
    } finally {
      finish();
      await held;
    }
    expect(await withLocalFileMutationLock(lock, async () => "settled")).toBe("settled");
  });

  it("recovers after positive exit evidence for a separate owning process", async () => {
    const { directory, lock } = await fixture();
    const entry = path.join(directory, "lock-owner.mjs");
    await build({
      stdin: {
        contents: `import { withLocalFileMutationLock } from ${JSON.stringify(fileURLToPath(new URL("../src/local-file-mutation-lock.ts", import.meta.url)))};
          await withLocalFileMutationLock(process.argv[2], async () => {
            process.stdout.write('acquired\\n');
            await new Promise(() => setInterval(() => {}, 1000));
          });`,
        resolveDir: process.cwd(),
      },
      outfile: entry,
      bundle: true,
      platform: "node",
      format: "esm",
      logLevel: "silent",
    });
    const child = spawn(process.execPath, [entry, lock], {
      env: {
        HOME: directory,
        USERPROFILE: directory,
        PATH: "",
        ...(process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}),
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    const closed = once(child, "close");
    try {
      const [chunk] = await once(child.stdout, "data", { signal: AbortSignal.timeout(3_000) });
      expect(chunk.toString()).toContain("acquired");
      expect((await readdir(lock))[0]).toContain(`owner-${child.pid}-`);
    } finally {
      child.kill("SIGKILL");
      await closed;
    }
    expect(await withLocalFileMutationLock(lock, async () => "recovered after exit")).toBe(
      "recovered after exit",
    );
  });
});
