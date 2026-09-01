import { describe, expect, it } from "vitest";
import { referenceImplementations } from "./scenarios-reference.js";
import { codeQualityScenarios } from "./scenarios.js";

describe("code-quality hidden evaluators", () => {
  // Python scenarios declare hiddenChecks and are executed by their driver, and
  // scenarios-validity.test.ts covers every scenario end to end in a real
  // workspace. This case keeps the in-process JavaScript evaluators honest.
  it("accepts JavaScript implementations that satisfy every held-out policy", async () => {
    const inProcess = codeQualityScenarios.filter(
      (scenario) => scenario.sourceFile.endsWith(".js") && !scenario.hiddenChecks,
    );
    expect(inProcess.length).toBeGreaterThan(0);
    for (const scenario of inProcess) {
      const implementation = referenceImplementations[scenario.scenarioId];
      if (!implementation) throw new Error(`Missing implementation for ${scenario.scenarioId}`);
      const result = await scenario.evaluateHidden(implementation);
      expect(result.assertions.every((assertion) => assertion.passed)).toBe(true);
    }
  });
});
