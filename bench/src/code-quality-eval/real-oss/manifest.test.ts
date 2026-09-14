import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { djangoAnnotationPruningTask } from "./django-manifest.js";
import { flaskBlueprintNameTask } from "./manifest.js";

describe("real OSS task manifest", () => {
  it("pins a licensed GitHub repository and an immutable base commit", () => {
    expect(flaskBlueprintNameTask.repository).toBe("pallets/flask");
    expect(flaskBlueprintNameTask.repositoryUrl).toBe("https://github.com/pallets/flask.git");
    expect(flaskBlueprintNameTask.license).toBe("BSD-3-Clause");
    expect(flaskBlueprintNameTask.baseCommit).toMatch(/^[0-9a-f]{40}$/);
  });

  it("keeps the upstream source patch out of every agent-visible source", () => {
    const visible = JSON.stringify({
      publicPrompt: flaskBlueprintNameTask.publicPrompt,
      sourceDocuments: flaskBlueprintNameTask.sourceDocuments,
      normativeRecords: flaskBlueprintNameTask.normativeRecords,
    });
    expect(visible).not.toContain("if not name:");
    expect(visible).not.toContain("test_empty_name_not_allowed");
  });

  it("pins the held-out SWE-bench test patch by digest", () => {
    expect(createHash("sha256").update(flaskBlueprintNameTask.hiddenTest.patch).digest("hex")).toBe(
      flaskBlueprintNameTask.hiddenTest.patchSha256,
    );
    expect(flaskBlueprintNameTask.hiddenTest.failToPass).toEqual([
      "tests/test_blueprints.py::test_empty_name_not_allowed",
    ]);
    expect(flaskBlueprintNameTask.hiddenTest.passToPass).toHaveLength(59);
  });

  it("models the Django change as nine behavior-bearing targets", () => {
    expect(djangoAnnotationPruningTask.repository).toBe("django/django");
    expect(djangoAnnotationPruningTask.baseCommit).toBe("321ecb40f4da842926e1bc07e11df4aabe53ca4b");
    expect(djangoAnnotationPruningTask.targets).toHaveLength(9);
    expect(
      djangoAnnotationPruningTask.targets.filter((target) => target.binding === "planned"),
    ).toHaveLength(3);
    expect(
      new Set(djangoAnnotationPruningTask.targets.map((target) => target.plannedSymbolId)).size,
    ).toBe(9);
  });

  it("keeps the Django solution and hidden tests out of agent-visible evidence", () => {
    const visible = JSON.stringify({
      publicPrompt: djangoAnnotationPruningTask.publicPrompt,
      sourceDocuments: djangoAnnotationPruningTask.sourceDocuments,
      normativeRecords: djangoAnnotationPruningTask.normativeRecords,
    });
    expect(visible).not.toContain("def get_refs");
    expect(visible).not.toContain("set_annotation_mask(annotation_mask)");
    expect(visible).not.toContain("test_unused_aliased_aggregate_pruned");
    expect(
      createHash("sha256").update(djangoAnnotationPruningTask.hiddenTest.patch).digest("hex"),
    ).toBe(djangoAnnotationPruningTask.hiddenTest.patchSha256);
    expect(djangoAnnotationPruningTask.hiddenTest.failToPass).toHaveLength(3);
    expect(djangoAnnotationPruningTask.hiddenTest.passToPass).toHaveLength(100);
    expect(djangoAnnotationPruningTask.publicTest.args).toContain("queries");
  });
});
