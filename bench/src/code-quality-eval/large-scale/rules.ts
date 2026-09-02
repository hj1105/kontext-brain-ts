import { governedPolicy, governedSubsystem, subsystems } from "./generator.js";

/**
 * The rule corpus is the variable that decides whether the retrieval arm is a
 * control or an oracle.
 *
 * At 30 documents, rule recall@5 was 100% and the retrieval arm tied the
 * treatment arm on nearly every scenario. At 42 it had already fallen to 95.2%.
 * A real codebase carries hundreds of approved decisions, so the corpus is
 * padded to that order of magnitude with policies that are plausible, adjacent,
 * and wrong for this task. Retrieval then has to discriminate rather than
 * recall.
 */
export interface LargeScaleRule {
  readonly kind: "decision" | "domain_term" | "invariant";
  readonly recordId: string;
  readonly revisionId: string;
  readonly text: string;
  readonly evidenceId: string;
  readonly evidenceText: string;
  /** Which subsystem the rule governs, or undefined for corpus noise. */
  readonly subsystem?: string;
}

/** The rules that actually govern the task. */
export function governingRules(): readonly LargeScaleRule[] {
  return [
    {
      kind: "decision",
      recordId: "decision:billing-retry-ceiling",
      revisionId: "revision:billing-retry-ceiling:3",
      subsystem: governedSubsystem,
      text: `Every retry delay in the billing subsystem uses a factor of ${governedPolicy.factor} and a ceiling of ${governedPolicy.capMs} milliseconds, so the delay is min(baseMs * ${governedPolicy.factor} ** failureIndex, ceiling). This applies only to the billing subsystem; notify, sync, and media keep their own approved factors and ceilings. The ceiling is declared exactly once, in ${governedPolicy.sharedModule}, and imported by every billing retry function. Copying the value into each file is not accepted.`,
      evidenceId: "evidence:billing:retry-ceiling",
      evidenceText:
        "The payments owner raised the billing factor and ceiling after a settlement outage, and required a single shared declaration so a future change cannot drift between files.",
    },
    {
      kind: "domain_term",
      recordId: "domain-term:recovery-ceiling",
      revisionId: "revision:recovery-ceiling:1",
      subsystem: governedSubsystem,
      text: `Recovery Ceiling is the maximum billing retry delay. Implementation code names the exported constant ${governedPolicy.constantName}. Do not call it a max delay or a timeout.`,
      evidenceId: "evidence:domain:recovery-ceiling",
      evidenceText: `The billing domain owner standardized Recovery Ceiling and the code term ${governedPolicy.constantName}.`,
    },
    {
      kind: "invariant",
      recordId: "invariant:billing-retry-bounded",
      revisionId: "revision:billing-retry-bounded:1",
      subsystem: governedSubsystem,
      text: "No billing retry delay exceeds the Recovery Ceiling, and no subsystem outside billing changes its retry behaviour as part of this work.",
      evidenceId: "evidence:billing:retry-bounded",
      evidenceText:
        "Operations requires the billing ceiling to hold and requires unrelated subsystems to stay byte-identical so the change can be rolled back independently.",
    },
  ];
}

/**
 * One decision per non-governed subsystem, stating the factor and ceiling that
 * must not change. These are the closest distractors in the corpus: an arm that
 * retrieves by similarity alone has little to separate them from the governing
 * decision.
 */
export function siblingRules(): readonly LargeScaleRule[] {
  return subsystems
    .filter((subsystem) => subsystem.name !== governedSubsystem)
    .map((subsystem) => ({
      kind: "decision" as const,
      recordId: `decision:${subsystem.name}-retry-ceiling`,
      revisionId: `revision:${subsystem.name}-retry-ceiling:1`,
      subsystem: subsystem.name,
      text: `Every retry delay in the ${subsystem.name} subsystem uses a factor of ${subsystem.factor} and a ceiling of ${subsystem.capMs} milliseconds. This value is current and approved; do not change it.`,
      evidenceId: `evidence:${subsystem.name}:retry-ceiling`,
      evidenceText: `The ${subsystem.name} owner confirmed the current factor and ceiling remain approved.`,
    }));
}

const noiseAreas = [
  "checkout",
  "catalog",
  "search",
  "identity",
  "audit",
  "reporting",
  "webhook",
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
  "import",
  "session",
  "quota",
] as const;

const noiseTopics = [
  {
    slug: "retry-ceiling",
    text: (area: string) =>
      `Retry delays in the ${area} subsystem use a factor of 2 and a ceiling of 20000 milliseconds.`,
  },
  {
    slug: "timeout-budget",
    text: (area: string) =>
      `The ${area} subsystem allows a total request timeout budget of 15000 milliseconds across all attempts.`,
  },
  {
    slug: "batch-size",
    text: (area: string) =>
      `The ${area} subsystem processes records in batches of 250 and never exceeds four concurrent batches.`,
  },
  {
    slug: "cache-ttl",
    text: (area: string) =>
      `The ${area} subsystem caches lookups for 300 seconds and revalidates on write.`,
  },
  {
    slug: "audit-retention",
    text: (area: string) =>
      `The ${area} subsystem retains audit records for 400 days before deletion.`,
  },
  {
    slug: "rounding",
    text: (area: string) =>
      `Monetary amounts in the ${area} subsystem round half to even at the line level.`,
  },
] as const;

/**
 * Padding that is plausible and adjacent. Several noise entries also describe a
 * retry ceiling, so a query about retry behaviour does not trivially isolate the
 * governing decision.
 */
export function noiseRules(): readonly LargeScaleRule[] {
  const rules: LargeScaleRule[] = [];
  for (const area of noiseAreas) {
    for (const topic of noiseTopics) {
      rules.push({
        kind: "decision",
        recordId: `decision:${area}-${topic.slug}`,
        revisionId: `revision:${area}-${topic.slug}:1`,
        text: topic.text(area),
        evidenceId: `evidence:${area}:${topic.slug}`,
        evidenceText: `The ${area} owner approved this value in the current review cycle.`,
      });
    }
  }
  return rules;
}

export function allRules(): readonly LargeScaleRule[] {
  return [...governingRules(), ...siblingRules(), ...noiseRules()];
}

export const retrievalQueryText =
  "Apply the current approved retry policy. Update the retry delay behaviour to match the approved decision.";
