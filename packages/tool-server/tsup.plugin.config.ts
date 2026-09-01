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
  banner: { js: "#!/usr/bin/env node" },
});
