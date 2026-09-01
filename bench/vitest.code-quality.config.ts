import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: false,
    include: ["bench/src/code-quality-eval/**/*.test.ts"],
    testTimeout: 30_000,
  },
});
