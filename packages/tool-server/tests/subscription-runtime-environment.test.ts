import { describe, expect, it } from "vitest";
import { takeHostKnowledgeCapability } from "../src/host-knowledge-tools.js";
import { subscriptionRuntimeEnvironment } from "../src/index.js";

describe("subscriptionRuntimeEnvironment", () => {
  it("takes management authority out of the inherited environment before any verifier or Git process can start", () => {
    const environment = { KONTEXT_HOST_MANAGEMENT_TOKEN: "a".repeat(64), PATH: "/fixture/bin" };
    expect(takeHostKnowledgeCapability(environment)).toBe("a".repeat(64));
    expect(environment).toEqual({ PATH: "/fixture/bin" });
    expect(takeHostKnowledgeCapability(environment)).toBeUndefined();
  });
  it("rejects malformed management authority without retaining it in the environment", () => {
    const environment = { KONTEXT_HOST_MANAGEMENT_TOKEN: "invalid" };
    expect(() => takeHostKnowledgeCapability(environment)).toThrow("Invalid host-management");
    expect(environment).toEqual({});
  });
  it("passes only CLI configuration and Kontext data while stripping API credentials", () => {
    const environment = subscriptionRuntimeEnvironment("/private/kontext", {
      PATH: "/usr/bin",
      HOME: "/home/user",
      CODEX_HOME: "/home/user/.codex",
      CLAUDE_CONFIG_DIR: "/home/user/.claude",
      CODEX_API_KEY: "codex-secret",
      ANTHROPIC_API_KEY: "claude-secret",
      DATABASE_URL: "database-secret",
      KONTEXT_HOST_MANAGEMENT_TOKEN: "host-management-secret",
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
    expect(environment).not.toHaveProperty("KONTEXT_HOST_MANAGEMENT_TOKEN");
  });
});
