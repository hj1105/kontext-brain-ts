import { randomUUID } from "node:crypto";
import { mkdir, readdir, rmdir, unlink, writeFile } from "node:fs/promises";
import path from "node:path";

/** Serializes local writers without expiring a live writer or replaying its mutation. */
export async function withLocalFileMutationLock<T>(
  lockDirectory: string,
  mutation: () => Promise<T>,
): Promise<T> {
  await mkdir(path.dirname(lockDirectory), { recursive: true, mode: 0o700 });
  const marker = `owner-${process.pid}-${randomUUID()}`;
  const markerPath = path.join(lockDirectory, marker);
  for (let attempt = 0; attempt < 200; attempt++) {
    let created = false;
    try {
      await mkdir(lockDirectory, { mode: 0o700 });
      created = true;
    } catch (error) {
      if (!hasCode(error, "EEXIST")) throw error;
    }
    if (created) {
      let acquired = false;
      try {
        await writeFile(markerPath, "", { flag: "wx", mode: 0o600 });
        const owners = await readdir(lockDirectory);
        // A paused initializer can encounter a replacement directory; only one marker may enter.
        acquired = owners.length === 1 && owners[0] === marker;
      } catch (error) {
        await removeMarker(markerPath);
        // Exclusive open can report EINVAL when a competing initializer removes the empty parent.
        if (!hasCode(error, "ENOENT") && !hasCode(error, "EINVAL")) throw error;
      }
      if (acquired) {
        try {
          return await mutation();
        } finally {
          await removeMarker(markerPath);
          await removeEmptyDirectory(lockDirectory);
        }
      }
      await removeMarker(markerPath);
    }
    await reclaimExitedOwners(lockDirectory);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error("Local file mutation lock is busy; retry after the owning operation settles");
}

async function reclaimExitedOwners(directory: string): Promise<void> {
  let entries: string[];
  try {
    entries = await readdir(directory);
  } catch (error) {
    if (hasCode(error, "ENOENT")) return;
    throw error;
  }
  for (const entry of entries) {
    const match = /^owner-([1-9][0-9]*)-[a-f0-9-]{36}$/.exec(entry);
    if (!match) throw new Error("Unrecognized local file lock owner; refusing to remove it");
    const processId = Number(match[1]);
    if (!Number.isSafeInteger(processId)) throw new Error("Invalid local file lock process ID");
    try {
      process.kill(processId, 0);
    } catch (error) {
      if (hasCode(error, "ESRCH")) await removeMarker(path.join(directory, entry));
      // Permission failures and unknown process state never prove the owner exited.
    }
  }
  await removeEmptyDirectory(directory);
}

async function removeMarker(marker: string): Promise<void> {
  try {
    await unlink(marker);
  } catch (error) {
    if (!hasCode(error, "ENOENT")) throw error;
  }
}

async function removeEmptyDirectory(directory: string): Promise<void> {
  try {
    await rmdir(directory);
  } catch (error) {
    // rmdir cannot remove a replacement directory once its owner's marker is present.
    if (!hasCode(error, "ENOENT") && !hasCode(error, "ENOTEMPTY") && !hasCode(error, "EEXIST"))
      throw error;
  }
}

function hasCode(error: unknown, code: string): boolean {
  return error instanceof Error && "code" in error && error.code === code;
}
