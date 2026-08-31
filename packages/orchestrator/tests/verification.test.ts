import type { LogicWorkItem, TaskContract } from "@kontext-brain/spec";
import { describe, expect, it } from "vitest";
import {
  VerificationCoordinator,
  VerifierInfrastructureError,
  VerifierRegistry,
  createFastVerificationPlan,
  createFullVerificationPlan,
  createTargetedVerificationPlan,
} from "../src/index.js";

const workItem: LogicWorkItem = {
  workItemId: "work-item:verify",
  taskId: "task:verify",
  plannedSymbolIds: ["symbol:b", "symbol:a"],
  dependsOn: [],
  allowedPaths: ["packages/example/src/index.ts"],
  requiredVerifiers: [
    { kind: "test", ref: "example:test" },
    { kind: "test", ref: "example:test" },
  ],
  capabilityId: "capability:verify",
};

const contract: TaskContract = {
  taskId: workItem.taskId,
  intent: "Verify a Task against its exact revision and context.",
  acceptance: [
    {
      criterionId: "acceptance:example",
      statement: "Example tests pass.",
      verifier: { kind: "test", ref: "example:test" },
    },
  ],
  nonGoals: [],
  targets: ["symbol:a"],
  risk: "low",
};

const binding = {
  workspacePath: "/workspace",
  codeRevision: "commit:result",
  contextDigest: "context:current",
  observedAt: "2026-08-28T03:00:00.000Z",
};

describe("verification plans", () => {
  it("plans all four mandatory fast symbol checks", () => {
    const plan = createFastVerificationPlan({ affectedSymbolIds: ["symbol:b", "symbol:a"] });

    expect(plan.tier).toBe("fast");
    expect(plan.requirements.map((requirement) => requirement.verifier.ref)).toEqual([
      "kontext:domain-term-check",
      "kontext:graph-query-check",
      "kontext:semantic-sync",
      "kontext:stable-symbol-identity",
    ]);
    expect(plan.requirements[0]?.subjectIds).toEqual(["symbol:a", "symbol:b"]);
  });

  it("deduplicates targeted Work Item and bound Invariant verifiers", () => {
    const plan = createTargetedVerificationPlan({
      workItem,
      boundInvariantVerifiers: [
        { kind: "test", ref: "example:test" },
        { kind: "query", ref: "invariant:no-egress" },
      ],
    });

    expect(plan.requirements).toHaveLength(2);
    expect(plan.requirements.every((requirement) => requirement.tier === "targeted")).toBe(true);
  });

  it("includes executable integrated suites, acceptance, and Invariants for low risk", () => {
    const plan = createFullVerificationPlan({
      contract,
      boundInvariantVerifiers: [{ kind: "query", ref: "invariant:no-egress" }],
    });

    expect(plan.requirements.map((requirement) => requirement.verifier)).toEqual(
      expect.arrayContaining([
        { kind: "typecheck", ref: "workspace:typecheck" },
        { kind: "test", ref: "workspace:test" },
        { kind: "build", ref: "workspace:build" },
        { kind: "lint", ref: "workspace:lint" },
        { kind: "test", ref: "example:test" },
        { kind: "query", ref: "invariant:no-egress" },
      ]),
    );
    expect(plan.requirements).not.toContainEqual(
      expect.objectContaining({
        verifier: { kind: "query", ref: "kontext:manifest-audit" },
      }),
    );
    expect(plan.requirements).not.toContainEqual(
      expect.objectContaining({
        verifier: { kind: "manual_review", ref: "kontext:independent-review" },
      }),
    );
  });

  it("adds independent review to medium and high risk full verification", () => {
    for (const risk of ["medium", "high"] as const) {
      expect(
        createFullVerificationPlan({ contract: { ...contract, risk } }).requirements,
      ).toContainEqual(
        expect.objectContaining({
          verifier: { kind: "manual_review", ref: "kontext:independent-review" },
        }),
      );
    }
  });
});

describe("VerificationCoordinator", () => {
  it("prefers an exact verifier and otherwise resolves the registered kind fallback", async () => {
    const registry = new VerifierRegistry();
    registry.registerFallback("test", {
      execute: async () => ({ result: "failed", output: "fallback" }),
    });
    registry.register(
      { kind: "test", ref: "example:test" },
      { execute: async () => ({ result: "passed", output: "exact" }) },
    );
    const coordinator = new VerificationCoordinator(registry);
    const exact = await coordinator.execute(
      {
        tier: "targeted",
        verifier: { kind: "test", ref: "example:test" },
        subjectIds: ["symbol:a"],
      },
      binding,
    );
    const fallback = await coordinator.execute(
      {
        tier: "targeted",
        verifier: { kind: "test", ref: "other:test" },
        subjectIds: ["symbol:a"],
      },
      binding,
    );

    expect(exact.run.result).toBe("passed");
    expect(fallback.run.result).toBe("failed");
  });

  it("binds a passing result to tier, exact revision, context, subjects, and output digest", async () => {
    const registry = new VerifierRegistry();
    registry.register(
      { kind: "test", ref: "example:test" },
      {
        execute: async () => ({ result: "passed", output: { tests: 12 } }),
      },
    );
    const coordinator = new VerificationCoordinator(registry);
    const requirement = createTargetedVerificationPlan({ workItem }).requirements[0];
    if (!requirement) throw new Error("expected targeted requirement");

    const execution = await coordinator.execute(requirement, binding);

    expect(execution.disposition).toBe("settled");
    expect(execution.run).toEqual(
      expect.objectContaining({
        tier: "targeted",
        verifierKind: "test",
        verifierRef: "example:test",
        codeRevision: binding.codeRevision,
        contextDigest: binding.contextDigest,
        subjectIds: ["symbol:a", "symbol:b", workItem.workItemId],
        result: "passed",
        outputDigest: expect.stringMatching(/^sha256:/),
      }),
    );
  });

  it("records missing adapters and infrastructure failures as retryable inconclusive runs", async () => {
    const registry = new VerifierRegistry();
    registry.register(
      { kind: "query", ref: "kontext:semantic-sync" },
      {
        execute: async () => {
          throw new VerifierInfrastructureError("semantic provider unavailable");
        },
      },
    );
    const coordinator = new VerificationCoordinator(registry);
    const requirements = createFastVerificationPlan({
      affectedSymbolIds: ["symbol:a"],
    }).requirements;
    const registered = requirements.find(
      (requirement) => requirement.verifier.ref === "kontext:semantic-sync",
    );
    const missing = requirements.find(
      (requirement) => requirement.verifier.ref === "kontext:domain-term-check",
    );
    if (!registered || !missing) throw new Error("expected fast requirements");

    const [first, second] = await Promise.all([
      coordinator.execute(registered, binding),
      coordinator.execute(missing, binding),
    ]);

    expect(first.run.result).toBe("inconclusive");
    expect(first.disposition).toBe("retryable");
    expect(second.run.result).toBe("inconclusive");
    expect(second.disposition).toBe("retryable");
  });
});
