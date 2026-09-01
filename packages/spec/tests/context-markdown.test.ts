import { describe, expect, it } from "vitest";
import type { DomainTermRevision, NormativeActivation, NormativeManifest } from "../src/index.js";
import {
  importContextMarkdown,
  projectDomainTermsToContextMarkdown,
  replaceGeneratedDomainTerms,
} from "../src/index.js";

const term: DomainTermRevision = {
  kind: "domain_term",
  organizationId: "org:acme",
  recordId: "domain-term:code-symbol",
  revisionId: "revision:code-symbol:1",
  scope: { kind: "workspace", workspaceId: "workspace:local" },
  evidence: [{ evidenceId: "evidence:design", sourceSpan: "§4.1" }],
  egress: {
    dataClassification: "internal",
    allowedRuntimeProviders: ["codex", "claude"],
  },
  authoredBy: "user:owner",
  authoredAt: "2026-08-28T00:00:00.000Z",
  term: "Code Symbol",
  definition: "A stable Kontext domain record for a language-facing code declaration.",
  avoid: ["AST Symbol"],
};
const activation: NormativeActivation = {
  organizationId: "org:acme",
  kind: "domain_term",
  recordId: term.recordId,
  revisionId: term.revisionId,
  scope: term.scope,
  state: "accepted_local",
  acceptedBy: "user:owner",
  acceptedAt: "2026-08-28T00:01:00.000Z",
};
const manifest: NormativeManifest = {
  schemaVersion: 1,
  organizationId: "org:acme",
  revisions: [term],
  activations: [activation],
};

describe("CONTEXT.md Domain Term projection", () => {
  it("renders accepted terms and marks the generated boundary", () => {
    const markdown = projectDomainTermsToContextMarkdown(manifest);

    expect(markdown).toContain("<!-- kontext-domain-terms:start -->");
    expect(markdown).toContain("## Code Symbol");
    expect(markdown).toContain("Avoid: `AST Symbol`.");
  });

  it("updates only the generated block in a mixed human-authored file", () => {
    const existing = `# Context

Human-maintained introduction.

<!-- kontext-domain-terms:start -->

## Old Term

Old definition.

<!-- kontext-domain-terms:end -->

Human-maintained footer.
`;
    const updated = replaceGeneratedDomainTerms(
      existing,
      projectDomainTermsToContextMarkdown(manifest),
    );

    expect(updated).toContain("Human-maintained introduction.");
    expect(updated).toContain("Human-maintained footer.");
    expect(updated).toContain("## Code Symbol");
    expect(updated).not.toContain("## Old Term");
  });

  it("imports an existing glossary once with file Evidence", () => {
    const imported = importContextMarkdown(
      "# Context\n\n## Task\n\nA bounded unit of intended work.\n",
      {
        organizationId: "org:acme",
        scope: { kind: "workspace", workspaceId: "workspace:local" },
        evidenceId: "evidence:file:context-md",
        egress: {
          dataClassification: "internal",
          allowedRuntimeProviders: ["codex", "claude"],
        },
        authoredBy: "user:owner",
        authoredAt: "2026-08-28T00:00:00.000Z",
      },
    );

    expect(imported).toEqual([
      expect.objectContaining({
        kind: "domain_term",
        recordId: "domain-term:task",
        term: "Task",
        definition: "A bounded unit of intended work.",
        evidence: [
          {
            evidenceId: "evidence:file:context-md",
            sourceSpan: "## Task",
          },
        ],
      }),
    ]);
    expect(
      importContextMarkdown("# Context\n\n## Task\n\nA bounded unit of intended work.\n", {
        organizationId: "org:acme",
        scope: { kind: "workspace", workspaceId: "workspace:local" },
        evidenceId: "evidence:file:context-md",
        egress: {
          dataClassification: "internal",
          allowedRuntimeProviders: ["codex", "claude"],
        },
        authoredBy: "user:owner",
        authoredAt: "2026-08-28T00:00:00.000Z",
      })[0]?.revisionId,
    ).toBe(imported[0]?.revisionId);
  });
});
