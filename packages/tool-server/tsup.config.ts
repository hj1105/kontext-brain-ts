import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/cli.ts", "src/task-cli.ts"],
  format: ["esm"],
  dts: { entry: "src/index.ts" },
  clean: true,
  tsconfig: "./tsconfig.json",
  external: [
    "@kontext-brain/context",
    "@kontext-brain/core",
    "@kontext-brain/local",
    "@kontext-brain/loader",
    "@kontext-brain/orchestrator",
    "@kontext-brain/runtime-claude",
    "@kontext-brain/runtime-codex",
    "@kontext-brain/spec",
    "@modelcontextprotocol/sdk",
  ],
  banner: { js: "#!/usr/bin/env node" },
});
