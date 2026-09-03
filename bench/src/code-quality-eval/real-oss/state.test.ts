import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { flaskBlueprintNameTask } from "./manifest.js";
import { buildRealOssStateAssembly } from "./state.js";

const cleanup = new Set<string>();

afterEach(async () => {
  await Promise.all([...cleanup].map((entry) => rm(entry, { recursive: true, force: true })));
  cleanup.clear();
});

describe("real OSS ontology assembly", () => {
  it("indexes real Python symbols and resolves provenance-backed governance", async () => {
    const workspacePath = await mkdtemp(path.join(tmpdir(), "kontext-real-oss-state-test-"));
    cleanup.add(workspacePath);
    await mkdir(path.join(workspacePath, "src", "flask"), { recursive: true });
    await writeFile(
      path.join(workspacePath, "src", "flask", "blueprints.py"),
      `class Blueprint:
    def __init__(self, name):
        self.name = name

    def register(self):
        return self.name
`,
    );
    await writeFile(
      path.join(workspacePath, "src", "flask", "app.py"),
      "def create_app():\n    return object()\n",
    );

    const result = await buildRealOssStateAssembly({
      task: flaskBlueprintNameTask,
      workspace: { workspacePath, baseRevision: flaskBlueprintNameTask.baseCommit },
      runtime: "codex",
    });

    expect(result.ontology.codeResources).toBe(2);
    expect(result.ontology.codeSymbols).toBeGreaterThan(2);
    expect(result.ontology.behaviorBearingSymbols).toBeGreaterThan(1);
    expect(result.ontology.targetQualifiedName).toBe("Blueprint.__init__");
    expect(result.ontology.targetSymbolId).toMatch(/^code-symbol:/);
    expect(result.ontology.provenanceResources).toBe(4);
    expect(result.ontology.governingRecordIds).toEqual(
      flaskBlueprintNameTask.normativeRecords.map((record) => record.recordId).sort(),
    );
  });
});
