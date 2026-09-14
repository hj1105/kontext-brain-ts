import { governedPolicy, governedSubsystem, subsystems } from "./generator.js";

/**
 * A real organization's knowledge is not a list of one-sentence rules. It is
 * issue threads, specification documents, and runbooks, most of which are
 * irrelevant to any given task.
 *
 * Splitting the surfaces this way is what makes the measurement mean something:
 *
 * - the issue body is public, because that is what a contributor is handed, and
 *   it deliberately does not state the factor, the ceiling, the constant name,
 *   or which subsystem is governed
 * - the specification and the decisions are sidecar-only, because that is the
 *   maintainer knowledge the product claims to supply
 *
 * The documents are also long prose rather than terse rules, which is harder for
 * similarity search than a one-line policy and is the realistic case.
 */
export interface KnowledgeDocument {
  readonly documentId: string;
  readonly kind: "issue" | "spec" | "runbook";
  readonly title: string;
  readonly body: string;
  /** Which subsystem the document concerns, or undefined for corpus noise. */
  readonly subsystem?: string;
}

/**
 * The public task. A contributor gets a symptom and a request, not the policy.
 */
export const publicIssue = {
  number: 4127,
  title: "Payment retries give up too early during downstream degradation",
  body: `## What happened

During the incident on 2026-08-14 a downstream processor was degraded for
about eleven minutes. Our retries stopped climbing well before the processor
recovered, so a batch of operations failed permanently and had to be replayed
by hand the next morning.

## What we expect

Retry delays should keep growing long enough to ride out an outage of that
length instead of flattening out early.

## Notes

The retry helpers were written per operation and have drifted, so please make
the change consistently rather than patching one call site. There is an
approved decision covering the exact numbers and the naming convention — apply
it rather than choosing values yourself.

Do not change subsystems that are not covered by that decision.
`,
} as const;

/** The specification the maintainers hold, which states the numbers. */
function governedSpec(): KnowledgeDocument {
  return {
    documentId: "spec:billing-retry-recovery",
    kind: "spec",
    subsystem: governedSubsystem,
    title: "Billing retry recovery specification",
    body: `# Billing retry recovery

## Scope

This specification covers retry delay computation for the billing subsystem
only. The notify, sync, and media subsystems each have their own approved
specification and are explicitly out of scope here. Changing them as part of a
billing change has caused rollback problems before and is not permitted.

## Delay curve

A retry delay is min(baseMs * ${governedPolicy.factor} ** failureIndex, ceiling).

The factor was raised to ${governedPolicy.factor} so that the curve reaches the
ceiling within the window a degraded processor typically needs, rather than
flattening out after a few attempts.

The ceiling is ${governedPolicy.capMs} milliseconds. Beyond that a caller is
expected to shed the operation to the replay queue instead of waiting longer.

## Where the ceiling lives

The ceiling is declared exactly once, in ${governedPolicy.sharedModule}, and
imported by every billing retry function. It must not be copied into the
individual operation files. The previous per-file copies drifted during the
2026-06 audit and that is the reason for this rule.

## Naming

The exported constant is named ${governedPolicy.constantName}. Do not name it a
max delay, a timeout, or a backoff limit; those terms mean other things in the
billing domain.
`,
  };
}

/** Sibling specifications that state values which must not change. */
function siblingSpecs(): readonly KnowledgeDocument[] {
  return subsystems
    .filter((subsystem) => subsystem.name !== governedSubsystem)
    .map((subsystem) => ({
      documentId: `spec:${subsystem.name}-retry`,
      kind: "spec" as const,
      subsystem: subsystem.name,
      title: `${subsystem.name} retry specification`,
      body: `# ${subsystem.name} retry

This specification covers retry delay computation for the ${subsystem.name}
subsystem only.

A retry delay is min(baseMs * ${subsystem.factor} ** failureIndex, ${subsystem.capMs}).

These values are current and approved. They were reviewed in the 2026-07 cycle
and deliberately left unchanged. Do not alter them while working on another
subsystem's retry policy.
`,
    }));
}

const noiseAreas = [
  "checkout",
  "catalog",
  "search",
  "identity",
  "audit",
  "reporting",
  "scheduling",
  "storage",
  "telemetry",
  "pricing",
  "tax",
  "shipping",
  "returns",
  "fraud",
  "consent",
  "export",
  "session",
  "quota",
] as const;

/**
 * Plausible, adjacent, and wrong. Several noise specifications also describe a
 * retry curve and a ceiling, so a query about retry behaviour does not isolate
 * the governing document by topic alone.
 */
function noiseDocuments(): readonly KnowledgeDocument[] {
  const documents: KnowledgeDocument[] = [];
  for (const area of noiseAreas) {
    documents.push({
      documentId: `spec:${area}-retry`,
      kind: "spec",
      title: `${area} retry specification`,
      body: `# ${area} retry\n\nA retry delay in the ${area} subsystem is min(baseMs * 2 ** failureIndex, 20000).\nThe ceiling is declared in the ${area} configuration module. These values were\napproved in the current review cycle and are not under revision.\n`,
    });
    documents.push({
      documentId: `issue:${area}-latency`,
      kind: "issue",
      title: `${area} latency spikes during peak hours`,
      body: `Requests in the ${area} subsystem occasionally take several seconds during\npeak hours. Investigation suggested queue depth rather than retry behaviour.\nNo policy change was made.\n`,
    });
    documents.push({
      documentId: `runbook:${area}-oncall`,
      kind: "runbook",
      title: `${area} on-call runbook`,
      body: `# ${area} on-call\n\nCheck the queue depth dashboard first. If retries are climbing to their\nceiling, shed to the replay queue rather than raising the ceiling. Escalate to\nthe ${area} owner before changing any approved value.\n`,
    });
  }
  return documents;
}

export function allDocuments(): readonly KnowledgeDocument[] {
  return [governedSpec(), ...siblingSpecs(), ...noiseDocuments()];
}

export function governingDocumentIds(): readonly string[] {
  return [governedSpec().documentId];
}
