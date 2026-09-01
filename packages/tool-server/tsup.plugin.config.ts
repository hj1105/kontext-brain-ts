import { defineConfig } from "tsup";

export default defineConfig({
  entry: { server: "src/task-cli.ts" },
  outDir: "../../plugins/kontext-brain",
  outExtension: () => ({ js: ".mjs" }),
  format: ["esm"],
  platform: "node",
  target: "node20",
  bundle: true,
  splitting: false,
  clean: false,
  dts: false,
  sourcemap: false,
  noExternal: [/.*/],
  banner: {
    js: [
      "#!/usr/bin/env node",
      'import { createRequire as __kontextCreateRequire } from "node:module";',
      'import { dirname as __kontextDirname } from "node:path";',
      'import { fileURLToPath as __kontextFileURLToPath } from "node:url";',
      "const require = __kontextCreateRequire(import.meta.url);",
      "const __filename = __kontextFileURLToPath(import.meta.url);",
      "const __dirname = __kontextDirname(__filename);",
    ].join("\n"),
  },
});
