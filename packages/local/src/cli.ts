import { readFile } from "node:fs/promises";
import path from "node:path";
import { FileTaskContextRepository } from "./file-task-context-repository.js";
import { resolvePluginDataDirectory } from "./plugin-data-directory.js";
import {
  type TaskContextStateAssemblyInput,
  assembleCurrentTaskContextState,
} from "./task-context-state-assembler.js";

async function main(): Promise<void> {
  const [command, argument] = process.argv.slice(2);
  const repository = new FileTaskContextRepository(resolvePluginDataDirectory());
  if (command === "publish-task-state" && argument) {
    const input = JSON.parse(
      await readFile(path.resolve(argument), "utf8"),
    ) as TaskContextStateAssemblyInput;
    const state = assembleCurrentTaskContextState(input);
    const result = await repository.publishCurrent(input.taskId, state);
    process.stdout.write(
      `${JSON.stringify(
        {
          taskId: input.taskId,
          digest: result.digest,
          created: result.created,
          currentStateFile: repository.currentStateFilePath(input.taskId),
          sourceFreshnessDigest: state.sourceFreshnessDigest,
          conflicts: state.conflicts,
        },
        null,
        2,
      )}\n`,
    );
    return;
  }
  if (command === "show-task-state" && argument) {
    process.stdout.write(`${JSON.stringify(await repository.getCurrent(argument), null, 2)}\n`);
    return;
  }
  throw new Error(
    "Usage: kontext-sidecar <publish-task-state assembly.json | show-task-state task-id>",
  );
}

main().catch((error) => {
  process.stderr.write(
    `kontext-sidecar error: ${error instanceof Error ? error.message : error}\n`,
  );
  process.exit(1);
});
