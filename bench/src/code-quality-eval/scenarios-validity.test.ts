import { rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { naiveImplementations, referenceImplementations } from "./scenarios-reference.js";
import { codeQualityScenarios } from "./scenarios.js";
import { createScenarioWorkspace, evaluateWorkspace } from "./workspace.js";

const temporaryDirectories: string[] = [];

afterAll(async () => {
  await Promise.all(
    temporaryDirectories
      .splice(0)
      .map((directory) => rm(directory, { recursive: true, force: true })),
  );
});

/**
 * Renders one of the shared implementations as the module under test. The
 * functions stringify as method shorthand, which is not a valid initializer on
 * its own, so they are rebuilt inside an object literal and re-exported.
 */
async function writeImplementation(
  workspacePath: string,
  sourceFile: string,
  implementation: Readonly<Record<string, unknown>>,
): Promise<void> {
  const names = Object.keys(implementation);
  const methods = names.map((name) => String(implementation[name])).join(",\n");
  const exports = names.map((name) => `export const ${name} = __impl.${name};`).join("\n");
  await writeFile(
    path.join(workspacePath, sourceFile),
    `const __impl = {\n${methods}\n};\n${exports}\n`,
  );
}

describe("code-quality scenarios are valid benchmark items", () => {
  it("covers enough scenarios for the harness pilot threshold", () => {
    expect(codeQualityScenarios.length).toBeGreaterThanOrEqual(10);
    const ids = codeQualityScenarios.map((scenario) => scenario.scenarioId);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("gives every scenario a distinct planned symbol and canonical term", () => {
    const symbols = codeQualityScenarios.map((scenario) => scenario.plannedSymbolId);
    expect(new Set(symbols).size).toBe(symbols.length);
    const terms = codeQualityScenarios.flatMap((scenario) => scenario.canonicalTerms);
    expect(new Set(terms).size).toBe(terms.length);
    for (const scenario of codeQualityScenarios) {
      expect(scenario.canonicalTerms.length).toBeGreaterThan(0);
      // A canonical term the public prompt already contains would not be held out.
      for (const term of scenario.canonicalTerms) {
        expect(scenario.publicPrompt).not.toContain(term);
        expect(scenario.publicTestSource).not.toContain(term);
        expect(scenario.initialSource).not.toContain(term);
      }
    }
  });

  it("states each policy only in the private sidecar rules", () => {
    for (const scenario of codeQualityScenarios) {
      const kinds = new Set(scenario.rules.map((rule) => rule.kind));
      expect(kinds).toContain("decision");
      expect(kinds).toContain("domain_term");
      expect(kinds).toContain("invariant");
      for (const rule of scenario.rules) {
        expect(rule.evidenceId).toBeTruthy();
        expect(rule.evidenceText).toBeTruthy();
      }
    }
  });

  for (const scenario of codeQualityScenarios) {
    it(`${scenario.scenarioId}: the policy-correct implementation passes everything`, async () => {
      const implementation = referenceImplementations[scenario.scenarioId];
      expect(implementation, `missing reference for ${scenario.scenarioId}`).toBeDefined();
      const workspace = await createScenarioWorkspace(scenario);
      temporaryDirectories.push(workspace.workspacePath);
      await writeImplementation(
        workspace.workspacePath,
        scenario.sourceFile,
        implementation as Readonly<Record<string, unknown>>,
      );

      const result = await evaluateWorkspace(scenario, workspace.workspacePath);
      expect(result.publicTestsPassed).toBe(true);
      const failed = result.hidden.assertions.filter((assertion) => !assertion.passed);
      expect(
        failed.map((assertion) => `${assertion.assertionId}: ${assertion.diagnostic ?? ""}`),
      ).toEqual([]);
    });

    it(`${scenario.scenarioId}: the naive implementation passes the public test but not the policy`, async () => {
      const implementation = naiveImplementations[scenario.scenarioId];
      expect(implementation, `missing naive impl for ${scenario.scenarioId}`).toBeDefined();
      const workspace = await createScenarioWorkspace(scenario);
      temporaryDirectories.push(workspace.workspacePath);
      await writeImplementation(
        workspace.workspacePath,
        scenario.sourceFile,
        implementation as Readonly<Record<string, unknown>>,
      );

      const result = await evaluateWorkspace(scenario, workspace.workspacePath);
      // The public surface must not reveal the policy, so the natural
      // implementation has to satisfy it.
      expect(result.publicTestsPassed).toBe(true);
      // And it must leave real headroom, or the scenario measures nothing.
      const failed = result.hidden.assertions.filter((assertion) => !assertion.passed);
      expect(failed.length).toBeGreaterThan(0);
    });
  }
});
