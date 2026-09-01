import { describe, expect, it } from "vitest";
import { codeQualityScenarios } from "./scenarios.js";

describe("code-quality hidden evaluators", () => {
  it("accepts implementations that satisfy every held-out policy", async () => {
    const implementations: Readonly<Record<string, Readonly<Record<string, unknown>>>> = {
      "retry-policy": {
        computeRetryDelay(failureIndex: number, baseMs: number): number {
          if (
            !Number.isInteger(failureIndex) ||
            failureIndex < 0 ||
            !Number.isInteger(baseMs) ||
            baseMs < 0
          ) {
            throw new RangeError("invalid retry input");
          }
          return Math.min(baseMs * 3 ** failureIndex, 4_500);
        },
      },
      "order-cancellation": {
        cancellationOutcome(order: {
          state: string;
          shipmentId: string | null;
          fraudHold: boolean;
        }): string {
          return order.state === "confirmed" && !order.shipmentId && order.fraudHold !== true
            ? "revocable"
            : "locked";
        },
      },
      "service-credit-allocation": {
        allocateServiceCredit(totalCents: number, accountIds: readonly string[]): object {
          if (!Number.isInteger(totalCents) || totalCents < 0)
            throw new RangeError("invalid total");
          if (
            accountIds.length === 0 ||
            accountIds.some((id) => !id) ||
            new Set(accountIds).size !== accountIds.length
          ) {
            throw new Error("invalid accounts");
          }
          const sorted = [...accountIds].sort();
          const base = Math.floor(totalCents / sorted.length);
          let remainder = totalCents % sorted.length;
          const values = new Map(sorted.map((id) => [id, base]));
          for (let index = sorted.length - 1; remainder > 0; index -= 1) {
            const id = sorted[index];
            if (!id) throw new Error("missing account");
            values.set(id, base + 1);
            remainder -= 1;
          }
          return Object.fromEntries(sorted.map((id) => [id, values.get(id)]));
        },
      },
    };

    for (const scenario of codeQualityScenarios) {
      const implementation = implementations[scenario.scenarioId];
      if (!implementation) throw new Error(`Missing implementation for ${scenario.scenarioId}`);
      const result = await scenario.evaluateHidden(implementation);
      expect(result.assertions.every((assertion) => assertion.passed)).toBe(true);
    }
  });
});
