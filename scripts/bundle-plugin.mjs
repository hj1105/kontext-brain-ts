#!/usr/bin/env node
/**
 * Builds plugins/kontext-brain/server.mjs from packages/tool-server/src/task-cli.ts.
 *
 * The bundle used to be committed with no way to rebuild it, so a fix landing in
 * packages/ never reached the plugin. A measurement run then executed a server
 * that was four merged pull requests behind its own source, which is how a
 * Python scenario could fail on a language the source already supported.
 *
 * Run `node scripts/bundle-plugin.mjs --check` in CI to fail when the committed
 * or generated bundle no longer matches the current source.
 */
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const repositoryRoot = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const entryPoint = path.join(repositoryRoot, "packages/tool-server/src/task-cli.ts");
const outputPath = path.join(repositoryRoot, "plugins/kontext-brain/server.mjs");
const banner = `#!/usr/bin/env node
import { createRequire as __kontextCreateRequire } from "node:module";
import { dirname as __kontextDirname } from "node:path";
import { fileURLToPath as __kontextFileURLToPath } from "node:url";
const require = __kontextCreateRequire(import.meta.url);
const __filename = __kontextFileURLToPath(import.meta.url);
const __dirname = __kontextDirname(__filename);`;

async function bundle() {
  const result = await build({
    entryPoints: [entryPoint],
    outfile: outputPath,
    bundle: true,
    platform: "node",
    format: "esm",
    target: "node20",
    // The plugin ships as one file that a host runs directly, so every
    // workspace package and dependency has to be inlined.
    packages: "bundle",
    banner: { js: banner },
    logLevel: "warning",
    write: false,
    metafile: false,
  });
  const output = result.outputFiles?.[0];
  if (!output) throw new Error("esbuild produced no output");
  return output.text;
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

const check = process.argv.includes("--check");
const generated = await bundle();

if (check) {
  const existing = await readFile(outputPath, "utf8").catch(() => undefined);
  if (existing === undefined) {
    process.stderr.write(`Missing ${path.relative(repositoryRoot, outputPath)}\n`);
    process.exit(1);
  }
  if (digest(existing) !== digest(generated)) {
    process.stderr.write(
      "The plugin bundle is stale. Run `node scripts/bundle-plugin.mjs` and commit the result.\n",
    );
    process.exit(1);
  }
  process.stdout.write("The plugin bundle matches its source.\n");
} else {
  await writeFile(outputPath, generated, { mode: 0o755 });
  process.stdout.write(
    `Wrote ${path.relative(repositoryRoot, outputPath)} (${(generated.length / 1_048_576).toFixed(1)} MiB)\n`,
  );
}
