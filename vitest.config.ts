import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: false,
    include: ["packages/*/tests/**/*.test.ts", "tests/**/*.test.ts"],
    testTimeout: 30000,
  },
});
