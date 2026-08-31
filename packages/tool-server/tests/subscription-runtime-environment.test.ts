import { describe, expect, it } from "vitest";
import { subscriptionRuntimeEnvironment } from "../src/index.js";

describe("subscriptionRuntimeEnvironment", () => {
  it("passes only CLI configuration and Kontext data while stripping API credentials", () => {
    const environment = subscriptionRuntimeEnvironment("/private/kontext", {
      PATH: "/usr/bin",
      HOME: "/home/user",
      CODEX_HOME: "/home/user/.codex",
      CLAUDE_CONFIG_DIR: "/home/user/.claude",
      CODEX_API_KEY: "codex-secret",
      ANTHROPIC_API_KEY: "claude-secret",
      DATABASE_URL: "database-secret",
    });

    expect(environment).toEqual({
      PATH: "/usr/bin",
      HOME: "/home/user",
      CODEX_HOME: "/home/user/.codex",
      CLAUDE_CONFIG_DIR: "/home/user/.claude",
      KONTEXT_PLUGIN_DATA: "/private/kontext",
    });
    expect(environment).not.toHaveProperty("CODEX_API_KEY");
    expect(environment).not.toHaveProperty("ANTHROPIC_API_KEY");
    expect(environment).not.toHaveProperty("DATABASE_URL");
  });
});
