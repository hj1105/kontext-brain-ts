import { readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { governedPolicy, governedSubsystem } from "./generator.js";
import {
  type LargeScaleWorkspace,
  createLargeScaleWorkspace,
  gradeLargeScaleWorkspace,
} from "./workspace.js";

const created: string[] = [];

afterAll(async () => {
  await Promise.all(created.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function workspace(): Promise<LargeScaleWorkspace> {
  const result = await createLargeScaleWorkspace();
  created.push(result.workspacePath);
  return result;
}

const ceilingExpression = /Math\.min\(baseMs \* 2 \*\* failureIndex, 30_?000\)/;

/**
 * Applies the policy the way it is meant to be applied: the ceiling is declared
 * once in a shared module and imported by each governed file.
 */
async function applyPolicy(
  built: LargeScaleWorkspace,
  predicate: (subsystem: string) => boolean,
  options: { readonly copyConstant?: boolean } = {},
): Promise<void> {
  const files = new Set(
    built.repository.functions.filter((item) => predicate(item.subsystem)).map((item) => item.file),
  );
  if (!options.copyConstant) {
    await writeFile(
      path.join(built.workspacePath, governedPolicy.sharedModule),
      `export const ${governedPolicy.constantName} = ${governedPolicy.capMs};\n`,
    );
  }
  for (const file of files) {
    const absolute = path.join(built.workspacePath, file);
    const original = await readFile(absolute, "utf8");
    const header = options.copyConstant
      ? `const ${governedPolicy.constantName} = ${governedPolicy.capMs};\n\n`
      : `import { ${governedPolicy.constantName} } from "./${path.basename(governedPolicy.sharedModule)}";\n\n`;
    const body = original.replace(
      ceilingExpression,
      `Math.min(baseMs * ${governedPolicy.factor} ** failureIndex, ${governedPolicy.constantName})`,
    );
    await writeFile(absolute, `${header}${body}`);
  }
}

describe("large-scale fixture is a valid benchmark item", () => {
  it("builds a repository large enough that the governed set is not obvious", async () => {
    const built = await workspace();
    expect(built.repository.functions.length).toBeGreaterThanOrEqual(30);
    expect(built.repository.governedNames.length).toBeGreaterThanOrEqual(8);
    // Lookalikes must outnumber the governed set, or discrimination is easy.
    expect(built.repository.decoyNames.length).toBeGreaterThan(
      built.repository.governedNames.length * 2,
    );
    expect(built.repository.files.size).toBeGreaterThanOrEqual(8);
  });

  it("never states the policy on the public surface", async () => {
    const built = await workspace();
    const task = await readFile(path.join(built.workspacePath, "TASK.md"), "utf8");
    const publicTest = await readFile(
      path.join(built.workspacePath, "test", "public.test.js"),
      "utf8",
    );
    for (const surface of [task, publicTest]) {
      expect(surface).not.toContain(governedPolicy.constantName);
      expect(surface).not.toContain(String(governedPolicy.capMs));
    }
  });

  it("scores a no-op as passing the public test with no recall", async () => {
    const built = await workspace();
    const grade = await gradeLargeScaleWorkspace(built);
    // The public surface must not reveal the policy, so doing nothing passes it.
    expect(grade.publicTestsPassed).toBe(true);
    expect(grade.targetRecall).toBe(0);
    expect(grade.regressionFailures).toBe(0);
    expect(grade.canonicalTermPresent).toBe(false);
    expect(grade.sharedConstantHonoured).toBe(false);
    // Only the governed functions are wrong, so hidden checks partly pass.
    expect(grade.hiddenPassed).toBeLessThan(grade.hiddenTotal);
    expect(grade.hiddenPassed).toBeGreaterThan(0);
  });

  it("scores the policy-correct change as full recall and full precision", async () => {
    const built = await workspace();
    await applyPolicy(built, (subsystem) => subsystem === governedSubsystem);
    const grade = await gradeLargeScaleWorkspace(built);
    expect(grade.targetRecall).toBe(1);
    expect(grade.collateralPrecision).toBe(1);
    expect(grade.decoysChanged).toEqual([]);
    expect(grade.regressionFailures).toBe(0);
    expect(grade.canonicalTermPresent).toBe(true);
    expect(grade.sharedConstantHonoured).toBe(true);
    expect(grade.constantDefinitionCount).toBe(1);
    expect(grade.hiddenPassed).toBe(grade.hiddenTotal);
  });

  it("distinguishes a shared ceiling from the same constant copied per file", async () => {
    const built = await workspace();
    await applyPolicy(built, (subsystem) => subsystem === governedSubsystem, {
      copyConstant: true,
    });
    const grade = await gradeLargeScaleWorkspace(built);
    // Behaviour is right, so only the structural check separates the two.
    expect(grade.hiddenPassed).toBe(grade.hiddenTotal);
    expect(grade.targetRecall).toBe(1);
    expect(grade.constantDefinitionCount).toBeGreaterThan(1);
    expect(grade.sharedConstantHonoured).toBe(false);
  });

  it("punishes rewriting every lookalike, which recall alone would reward", async () => {
    const built = await workspace();
    await applyPolicy(built, () => true);
    const grade = await gradeLargeScaleWorkspace(built);
    // Recall is satisfied, so precision and the decoy suite are what catch it.
    expect(grade.targetRecall).toBe(1);
    expect(grade.collateralPrecision).toBeLessThan(1);
    expect(grade.decoysChanged.length).toBeGreaterThan(0);
    expect(grade.regressionFailures).toBeGreaterThan(0);
    expect(grade.publicTestsPassed).toBe(false);
  });
});
