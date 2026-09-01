import { describe, expect, it } from "vitest";
import { referenceImplementations } from "./scenarios-reference.js";
import { codeQualityScenarios } from "./scenarios.js";

describe("code-quality hidden evaluators", () => {
  it("accepts implementations that satisfy every held-out policy", async () => {
    for (const scenario of codeQualityScenarios) {
      const implementation = referenceImplementations[scenario.scenarioId];
      if (!implementation) throw new Error(`Missing implementation for ${scenario.scenarioId}`);
      const result = await scenario.evaluateHidden(implementation);
      expect(result.assertions.every((assertion) => assertion.passed)).toBe(true);
    }
  });
});
