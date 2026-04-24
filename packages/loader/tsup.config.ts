import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  dts: true,
  clean: true,
  tsconfig: "./tsconfig.json",
  external: ["@kontext-brain/core", "@kontext-brain/llm", "@kontext-brain/mcp"],
});
