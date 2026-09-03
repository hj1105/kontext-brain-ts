import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
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
});
