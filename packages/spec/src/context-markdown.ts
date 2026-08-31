import { createHash } from "node:crypto";
import type {
  DomainTermRevision,
  GovernanceScope,
  NormativeManifest,
  RuntimeEgressPolicy,
} from "./domain.js";

const GENERATED_START = "<!-- kontext-domain-terms:start -->";
const GENERATED_END = "<!-- kontext-domain-terms:end -->";

export interface ContextMarkdownImportInput {
  readonly organizationId: string;
  readonly scope: GovernanceScope;
  readonly evidenceId: string;
  readonly egress: RuntimeEgressPolicy;
  readonly authoredBy: string;
  readonly authoredAt: string;
}

export function projectDomainTermsToContextMarkdown(manifest: NormativeManifest): string {
  const revisions = new Map(
    manifest.revisions
      .filter((revision): revision is DomainTermRevision => revision.kind === "domain_term")
      .map((revision) => [revision.revisionId, revision]),
  );
  const activeTerms = manifest.activations
    .filter(
      (activation) =>
        activation.kind === "domain_term" &&
        (activation.state === "accepted" || activation.state === "accepted_local"),
    )
    .map((activation) => revisions.get(activation.revisionId))
    .filter((revision): revision is DomainTermRevision => Boolean(revision))
    .sort(
      (left, right) =>
        left.term.localeCompare(right.term) || left.revisionId.localeCompare(right.revisionId),
    );
  const sections = activeTerms.map((revision) => {
    const avoid =
      revision.avoid && revision.avoid.length > 0
        ? `\n\nAvoid: ${revision.avoid.map((term) => `\`${term}\``).join(", ")}.`
        : "";
    return `## ${revision.term}\n\n${revision.definition}${avoid}`;
  });
  return [
    "# Context",
    "",
    "Generated from accepted Kontext Brain Domain Term revisions. Do not edit this block directly.",
    "",
    GENERATED_START,
    ...sections.flatMap((section) => ["", section]),
    "",
    GENERATED_END,
    "",
  ].join("\n");
}

export function replaceGeneratedDomainTerms(markdown: string, generatedProjection: string): string {
  const generatedBlock = extractGeneratedBlock(generatedProjection);
  const start = markdown.indexOf(GENERATED_START);
  const end = markdown.indexOf(GENERATED_END);
  if (start < 0 && end < 0) {
    return `${markdown.trimEnd()}\n\n${generatedBlock}\n`;
  }
  if (start < 0 || end < start) {
    throw new Error("CONTEXT.md has an invalid generated Domain Term boundary");
  }
  return `${markdown.slice(0, start)}${generatedBlock}${markdown.slice(
    end + GENERATED_END.length,
  )}`;
}

export function importContextMarkdown(
  markdown: string,
  input: ContextMarkdownImportInput,
): readonly DomainTermRevision[] {
  const sections = markdown.split(/^## /m).slice(1);
  return sections.flatMap((section) => {
    const newline = section.indexOf("\n");
    const term = (newline < 0 ? section : section.slice(0, newline)).trim();
    const body = newline < 0 ? "" : section.slice(newline + 1).trim();
    if (!term || !body) return [];
    const definition = body.replace(GENERATED_START, "").replace(GENERATED_END, "").trim();
    const recordId = `domain-term:${slug(term)}`;
    const revisionId = `revision:${digest(
      JSON.stringify([recordId, input.scope, definition]),
    ).slice(0, 24)}`;
    return [
      {
        kind: "domain_term" as const,
        organizationId: input.organizationId,
        recordId,
        revisionId,
        scope: input.scope,
        evidence: [{ evidenceId: input.evidenceId, sourceSpan: `## ${term}` }],
        egress: input.egress,
        authoredBy: input.authoredBy,
        authoredAt: input.authoredAt,
        term,
        definition,
      },
    ];
  });
}

function extractGeneratedBlock(markdown: string): string {
  const start = markdown.indexOf(GENERATED_START);
  const end = markdown.indexOf(GENERATED_END);
  if (start < 0 || end < start) {
    throw new Error("Generated CONTEXT.md projection is missing its boundary");
  }
  return markdown.slice(start, end + GENERATED_END.length);
}

function slug(value: string): string {
  const normalized = value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-|-$/g, "");
  return normalized || digest(value).slice(0, 16);
}

function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
