import { execFile } from "node:child_process";
import { rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { afterAll, describe, expect, it } from "vitest";
import { naiveImplementations, referenceImplementations } from "./scenarios-reference.js";
import { codeQualityScenarios } from "./scenarios.js";
import { createScenarioWorkspace } from "./workspace.js";

const execFileAsync = promisify(execFile);
const temporaryDirectories: string[] = [];
const benchDirectory = fileURLToPath(new URL("../..", import.meta.url));

/**
 * Evaluates in a child process rather than importing the module here. The real
 * harness runs under plain node, where a workspace .ts file loads natively,
 * while Vitest's transform pipeline refuses a file outside the project root.
 * Running the module out of process also keeps model-generated code out of the
 * evaluator.
 */
async function evaluateOutOfProcess(
  scenarioId: string,
  workspacePath: string,
): Promise<{ publicTestsPassed: boolean; failed: string[] }> {
  const script = `
    const { evaluateWorkspace } = await import("./src/code-quality-eval/workspace.ts");
    const { codeQualityScenarios } = await import("./src/code-quality-eval/scenarios.ts");
    const scenario = codeQualityScenarios.find((item) => item.scenarioId === ${JSON.stringify(scenarioId)});
    const result = await evaluateWorkspace(scenario, ${JSON.stringify(workspacePath)});
    process.stdout.write(JSON.stringify({
      publicTestsPassed: result.publicTestsPassed,
      failed: result.hidden.assertions
        .filter((assertion) => !assertion.passed)
        .map((assertion) => assertion.assertionId + ": " + (assertion.diagnostic ?? "")),
    }));
  `;
  const { stdout } = await execFileAsync(
    process.execPath,
    ["--import", "tsx", "--input-type=module", "-e", script],
    { cwd: benchDirectory, encoding: "utf8", maxBuffer: 10 * 1024 * 1024 },
  );
  return JSON.parse(stdout) as { publicTestsPassed: boolean; failed: string[] };
}

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
  // A class stringifies as a valid expression, but a method shorthand does not,
  // so only the latter is rebuilt inside an object literal.
  const parts = Object.entries(implementation).map(([name, value]) => {
    const source = String(value);
    return source.startsWith("class")
      ? `export const ${name} = ${source};`
      : `const __${name} = {\n${source}\n};\nexport const ${name} = __${name}.${name};`;
  });
  await writeFile(path.join(workspacePath, sourceFile), `${parts.join("\n")}\n`);
}

async function writeSource(
  workspacePath: string,
  sourceFile: string,
  source: string,
): Promise<void> {
  await writeFile(path.join(workspacePath, sourceFile), source);
}

/**
 * Implementations that cannot be recovered by stringifying a live function
 * carry their source text instead. Python has no JavaScript function to
 * stringify, and a TypeScript function stringifies with the transpiler's
 * injected helpers, which do not exist in the workspace.
 */
const sourceImplementations: Readonly<
  Record<string, { readonly reference: string; readonly naive: string }>
> = {
  "ledger-posting": {
    reference: `const DISPUTE_HOLD_CENTS = 2500;

export type PostingOutcome = "settled" | "pending" | "rejected";

export interface Posting {
  readonly amountCents: number;
  readonly clearedAt: string | null;
  readonly disputed: boolean;
}

export function classifyPosting(posting: Posting): PostingOutcome {
  if (posting.amountCents <= 0) return "rejected";
  if (posting.disputed && posting.amountCents > DISPUTE_HOLD_CENTS) return "pending";
  return posting.clearedAt ? "settled" : "pending";
}
`,
    naive: `export type PostingOutcome = "settled" | "pending" | "rejected";

export interface Posting {
  readonly amountCents: number;
  readonly clearedAt: string | null;
  readonly disputed: boolean;
}

export function classifyPosting(posting: Posting): PostingOutcome {
  if (posting.amountCents <= 0) return "rejected";
  if (posting.disputed) return "pending";
  return posting.clearedAt ? "settled" : "pending";
}
`,
  },
  "membership-tier": {
    reference: `const TENURE_CREDIT_CENTS = 1000;

export type MembershipTier = "standard" | "silver" | "gold";

export class MembershipPolicy {
  resolve(monthsActive: number, lifetimeSpendCents: number): MembershipTier {
    if (monthsActive < 0 || lifetimeSpendCents < 0) throw new RangeError("negative input");
    const credited = lifetimeSpendCents + Math.floor(monthsActive / 12) * TENURE_CREDIT_CENTS;
    if (credited >= 50000) return "gold";
    if (credited >= 20000) return "silver";
    return "standard";
  }
}
`,
    naive: `export type MembershipTier = "standard" | "silver" | "gold";

export class MembershipPolicy {
  resolve(monthsActive: number, lifetimeSpendCents: number): MembershipTier {
    if (lifetimeSpendCents >= 50000 || monthsActive >= 60) return "gold";
    if (lifetimeSpendCents >= 20000 || monthsActive >= 24) return "silver";
    return "standard";
  }
}
`,
  },
  "python-volume-pricing": {
    reference: `MARGINAL_TIER_BREAKS = ((10, 100), (100, 90), (None, 80))


def price_order(units, unit_price_cents):
    if not isinstance(units, int) or isinstance(units, bool) or units < 0:
        raise ValueError("invalid units")
    if not isinstance(unit_price_cents, int) or unit_price_cents < 0:
        raise ValueError("invalid price")
    total = 0
    remaining = units
    previous = 0
    for limit, percent in MARGINAL_TIER_BREAKS:
        span = remaining if limit is None else min(remaining, limit - previous)
        if span <= 0:
            break
        total += (span * unit_price_cents * percent) // 100
        remaining -= span
        previous = limit if limit is not None else previous
    return total
`,
    naive: `def price_order(units, unit_price_cents):
    if units > 100:
        percent = 80
    elif units > 10:
        percent = 90
    else:
        percent = 100
    return (units * unit_price_cents * percent) // 100
`,
  },
  "python-approval-quorum": {
    reference: `QUORUM_FLOOR = 3


def quorum_met(approvals, total_reviewers, risk):
    if approvals < 0 or total_reviewers < 0 or approvals > total_reviewers:
        raise ValueError("invalid approvals")
    if risk == "high":
        required = max(QUORUM_FLOOR, -(-total_reviewers * 2 // 3))
    else:
        required = total_reviewers // 2 + 1
    return approvals >= required
`,
    naive: `def quorum_met(approvals, total_reviewers, risk):
    ratio = 2 / 3 if risk == "high" else 0.5
    return approvals > total_reviewers * ratio
`,
  },
};

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
      const sourceForm = sourceImplementations[scenario.scenarioId];
      const implementation = referenceImplementations[scenario.scenarioId];
      expect(
        sourceForm ?? implementation,
        `missing reference for ${scenario.scenarioId}`,
      ).toBeDefined();
      const workspace = await createScenarioWorkspace(scenario);
      temporaryDirectories.push(workspace.workspacePath);
      if (sourceForm) {
        await writeSource(workspace.workspacePath, scenario.sourceFile, sourceForm.reference);
      } else {
        await writeImplementation(
          workspace.workspacePath,
          scenario.sourceFile,
          implementation as Readonly<Record<string, unknown>>,
        );
      }

      const result = await evaluateOutOfProcess(scenario.scenarioId, workspace.workspacePath);
      expect(result.publicTestsPassed).toBe(true);
      expect(result.failed).toEqual([]);
    });

    it(`${scenario.scenarioId}: the naive implementation passes the public test but not the policy`, async () => {
      const sourceForm = sourceImplementations[scenario.scenarioId];
      const implementation = naiveImplementations[scenario.scenarioId];
      expect(
        sourceForm ?? implementation,
        `missing naive impl for ${scenario.scenarioId}`,
      ).toBeDefined();
      const workspace = await createScenarioWorkspace(scenario);
      temporaryDirectories.push(workspace.workspacePath);
      if (sourceForm) {
        await writeSource(workspace.workspacePath, scenario.sourceFile, sourceForm.naive);
      } else {
        await writeImplementation(
          workspace.workspacePath,
          scenario.sourceFile,
          implementation as Readonly<Record<string, unknown>>,
        );
      }

      const result = await evaluateOutOfProcess(scenario.scenarioId, workspace.workspacePath);
      // The public surface must not reveal the policy, so the natural
      // implementation has to satisfy it.
      expect(result.publicTestsPassed).toBe(true);
      // And it must leave real headroom, or the scenario measures nothing.
      expect(result.failed.length).toBeGreaterThan(0);
    });
  }
});
