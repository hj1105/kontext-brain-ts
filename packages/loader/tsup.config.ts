import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/ontology-cli-main.ts"],
  format: ["esm"],
  dts: true,
  clean: true,
  tsconfig: "./tsconfig.json",
  external: ["@kontext-brain/core", "@kontext-brain/llm", "@kontext-brain/mcp"],
});
