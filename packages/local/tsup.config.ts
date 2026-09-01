import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/cli.ts"],
  format: ["esm"],
  dts: true,
  clean: true,
  tsconfig: "./tsconfig.json",
  banner: { js: "#!/usr/bin/env node" },
});
