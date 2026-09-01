import { describe, expect, it } from "vitest";
import type {
  AgentRuntimePort,
  RuntimeCapabilitySnapshot,
  RuntimeSession,
  RuntimeWorkInput,
} from "../src/index.js";
import { IndependentReviewCoordinator, createRuntimeCapabilitySnapshot } from "../src/index.js";

describe("IndependentReviewCoordinator", () => {
  it("selects a subscription runtime that did not author the change", async () => {
    const codex = runtime("codex", '{"verdict":"failed","findings":[]}');
    const claude = runtime("claude", '{"verdict":"passed","findings":[]}');
    const coordinator = new IndependentReviewCoordinator([codex, claude]);

    const result = await coordinator.review(request());

    expect(result.reviewerProvider).toBe("claude");
    expect(result.verificationRun.result).toBe("passed");
    expect(claude.inputs[0]?.executionRole).toBe("independent_review");
    expect(codex.inputs).toHaveLength(0);
  });

  it("turns malformed or unavailable cross-runtime review into inconclusive proof", async () => {
    const result = await new IndependentReviewCoordinator([runtime("codex", "not json")]).review(
      request(),
    );

    expect(result.reviewerProvider).toBeUndefined();
    expect(result.verificationRun.result).toBe("inconclusive");
    expect(result.diagnostic).toContain("No eligible non-author");
  });

  it("does not accept malformed output from an otherwise eligible reviewer", async () => {
    const result = await new IndependentReviewCoordinator([runtime("codex", "not json")]).review({
      ...request(),
      authorProviders: ["claude"],
    });

    expect(result.reviewerProvider).toBe("codex");
    expect(result.verificationRun.result).toBe("inconclusive");
    expect(result.diagnostic).toContain("Invalid independent review output");
  });

  it("does not send the review packet to a provider excluded by context egress", async () => {
    const claude = runtime("claude", '{"verdict":"passed","findings":[]}');
    const result = await new IndependentReviewCoordinator([claude]).review({
      ...request(),
      eligibleProviders: ["codex"],
    });

    expect(result.verificationRun.result).toBe("inconclusive");
    expect(claude.inputs).toHaveLength(0);
  });
});

function request() {
  return {
    contract: {
      taskId: "task:review",
      intent: "Review the integrated behavior.",
      acceptance: [],
      nonGoals: [],
      targets: ["symbol:handler"],
      risk: "medium" as const,
    },
    snapshot: {
      taskId: "task:review",
      baseCodeRevision: "commit:base",
      effectiveScopes: [{ kind: "personal" as const, subjectId: "user:one" }],
      normativeRevisions: [],
      requiredEvidenceIds: [],
      sourceFreshnessDigest: "freshness:one",
      contextDigest: "context:one",
      createdAt: "2026-08-31T00:00:00.000Z",
    },
    workspacePath: "/workspace",
    codeRevision: "workspace:result",
    authorProviders: ["codex" as const],
    eligibleProviders: ["codex" as const, "claude" as const],
    changedSymbolIds: ["symbol:handler"],
    changedPaths: ["src/handler.ts"],
    allowedRuleRefs: [],
    allowedEvidenceIds: [],
    reviewPacket: "Review this diff.",
    reviewedAt: "2026-08-31T01:00:00.000Z",
  };
}

function runtime(provider: "codex" | "claude", output: string) {
  return new FakeRuntime(provider, output);
}

class FakeRuntime implements AgentRuntimePort {
  readonly inputs: RuntimeWorkInput[] = [];

  constructor(
    readonly provider: "codex" | "claude",
    private readonly output: string,
  ) {}

  async inspectCapabilities(): Promise<RuntimeCapabilitySnapshot> {
    return createRuntimeCapabilitySnapshot({
      provider: this.provider,
      cliPath: this.provider,
      installed: true,
      authenticated: true,
      billingPath: "subscription",
      supports: {
        structuredOutput: true,
        sessionResume: true,
        mcp: true,
        hooks: true,
        workspaceSandbox: true,
      },
      inspectedAt: "2026-08-31T00:00:00.000Z",
    });
  }

  async start(input: RuntimeWorkInput): Promise<RuntimeSession> {
    this.inputs.push(input);
    return {
      sessionId: `session:${this.provider}`,
      provider: this.provider,
      status: "completed",
      output: this.output,
      events: [],
      startedAt: "2026-08-31T01:00:00.000Z",
      completedAt: "2026-08-31T01:01:00.000Z",
    };
  }

  async resume(): Promise<RuntimeSession> {
    throw new Error("not used");
  }

  async terminate(): Promise<void> {}
}
