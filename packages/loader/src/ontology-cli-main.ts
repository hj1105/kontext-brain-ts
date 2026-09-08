import { runOntologyCli } from "./ontology-cli.js";

/**
 * A source connector keeps its transport open — a stdio server's pipes, an SSE
 * request — and nothing in the agent closes them, so the event loop stays alive
 * after the command is done. Exiting explicitly, once stdout has drained, keeps
 * a finished build from being reported as a timeout by whoever spawned us.
 */
function exitAfterFlush(code: number): void {
  const done = (): void => process.exit(code);
  if (process.stdout.write("")) {
    done();
    return;
  }
  process.stdout.once("drain", done);
  // A pipe nobody reads would otherwise hold the process open forever.
  setTimeout(done, 5_000).unref();
}

runOntologyCli(process.argv.slice(2))
  .then((code) => {
    exitAfterFlush(code);
  })
  .catch((error: unknown) => {
    process.stderr.write(
      `kontext-ontology error: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    exitAfterFlush(1);
  });
