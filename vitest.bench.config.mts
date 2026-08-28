import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: false,
    include: ["bench/src/rag-eval-v2/**/*.test.ts"],
    testTimeout: 30000,
  },
});
