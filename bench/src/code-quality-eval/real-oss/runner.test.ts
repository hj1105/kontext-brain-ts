import { describe, expect, it } from "vitest";
import { flaskBlueprintNameTask } from "./manifest.js";
import { ragRealOssPrompt, realOssPrompt, renderRawSourceContext } from "./runner.js";

describe("real OSS prompts", () => {
  it("keeps baseline free of sidecar and held-out test content", () => {
    const prompt = realOssPrompt({
      task: flaskBlueprintNameTask,
      workspacePath: "/tmp/workspace",
      runtime: "codex",
    });
    expect(prompt).not.toContain("kontext_prepare_task");
    expect(prompt).not.toContain("test_empty_name_not_allowed");
    expect(prompt).not.toContain("if not name:");
  });

  it("gives RAG the raw provenance documents, not extracted records", () => {
    const prompt = ragRealOssPrompt({
      task: flaskBlueprintNameTask,
      workspacePath: "/tmp/workspace",
    });
    expect(prompt).toContain("https://github.com/pallets/flask/issues/5010");
    expect(prompt).toContain("Modular Applications with Blueprints");
    expect(prompt).not.toContain("decision:flask-blueprint-name-required");
    expect(renderRawSourceContext(flaskBlueprintNameTask)).not.toContain(
      "test_empty_name_not_allowed",
    );
  });

  it("requires one ontology lookup per behavior-bearing symbol and both checks", () => {
    const prompt = realOssPrompt({
      task: flaskBlueprintNameTask,
      workspacePath: "/tmp/workspace",
      runtime: "codex",
      targets: flaskBlueprintNameTask.targets.map((target) => ({
        workItemId: target.workItemId,
        plannedSymbolId: target.plannedSymbolId,
      })),
      createdAt: "2026-09-02T00:00:00.000Z",
    });
    expect(prompt).toContain("kontext_prepare_task");
    expect(prompt).toContain("one at a time and in order");
    expect(prompt).toContain("kontext_begin_logic once");
    expect(prompt).toContain("tier=fast");
    expect(prompt).toContain("tier=targeted");
    expect(prompt).toContain(flaskBlueprintNameTask.targets[0]?.plannedSymbolId);
    expect(prompt).not.toContain("test_empty_name_not_allowed");
  });
});
