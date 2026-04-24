import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/cli.ts"],
  format: ["esm"],
  dts: { entry: "src/index.ts" },
  clean: true,
  tsconfig: "./tsconfig.json",
  external: [
    "@kontext-brain/core",
    "@kontext-brain/loader",
    "@modelcontextprotocol/sdk",
  ],
  banner: { js: "#!/usr/bin/env node" },
});
