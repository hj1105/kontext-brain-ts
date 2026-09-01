import type {
  ArmComparison,
  ArmSummary,
  CodeQualityArm,
  CodeQualityReport,
  CodeQualityRunConfig,
  CodeQualityRunResult,
  PairedOutcomeSummary,
} from "./contracts.js";

export function buildCodeQualityReport(input: {
  readonly config: CodeQualityRunConfig;
  readonly scenarios: readonly string[];
  readonly runs: readonly CodeQualityRunResult[];
  readonly generatedAt?: string;
}): CodeQualityReport {
  const baseline = summarizeArm("baseline", input.runs);
  const rag = summarizeArm("rag", input.runs);
  const kontext = summarizeArm("kontext", input.runs);
  const paired = summarizePairs(input.runs);
  const comparisons = [
    compareArms(input.runs, "kontext", "baseline"),
    compareArms(input.runs, "kontext", "rag"),
    compareArms(input.runs, "rag", "baseline"),
  ].filter((comparison) => comparison.pairs > 0);
  const evidenceStrength = classifyEvidenceStrength(input.scenarios.length, paired.pairs);
  const hiddenAssertionUplift = kontext.hiddenAssertionPassRate - baseline.hiddenAssertionPassRate;
  const taskSuccessUplift = kontext.taskSuccessRate - baseline.taskSuccessRate;
  return {
    schemaVersion: 1,
    generatedAt: input.generatedAt ?? new Date().toISOString(),
    config: input.config,
    scenarios: [...input.scenarios],
    runs: [...input.runs],
    summaries: rag.runs > 0 ? [baseline, rag, kontext] : [baseline, kontext],
    paired,
    comparisons,
    hiddenAssertionUplift,
    taskSuccessUplift,
    evidenceStrength,
    verdict: verdict({
      evidenceStrength,
      hiddenAssertionUplift,
      taskSuccessUplift,
      paired,
    }),
    limitations: limitations(evidenceStrength, input.scenarios.length, paired.pairs),
  };
}

export function summarizeArm(
  arm: CodeQualityArm,
  allRuns: readonly CodeQualityRunResult[],
): ArmSummary {
  const runs = allRuns.filter((run) => run.arm === arm);
  const eligibleRuns = runs.filter((run) => run.evaluationEligible);
  const hiddenAssertions = eligibleRuns.flatMap((run) => run.hiddenAssertions);
  const expectedTerms = eligibleRuns.reduce(
    (total, run) => total + run.canonicalTermsPresent.length + run.canonicalTermsMissing.length,
    0,
  );
  const summary: ArmSummary = {
    arm,
    runs: runs.length,
    eligibleRuns: eligibleRuns.length,
    runtimeCompletionRate: mean(runs.map((run) => Number(run.runtimeExitCode === 0))),
    publicTestPassRate: mean(eligibleRuns.map((run) => Number(run.publicTestsPassed))),
    hiddenAssertionPassRate: mean(hiddenAssertions.map((assertion) => Number(assertion.passed))),
    taskSuccessRate: mean(eligibleRuns.map((run) => Number(isSuccessfulRun(run)))),
    canonicalTermPassRate:
      expectedTerms === 0
        ? 0
        : eligibleRuns.reduce((total, run) => total + run.canonicalTermsPresent.length, 0) /
          expectedTerms,
    scopeComplianceRate: mean(eligibleRuns.map((run) => Number(run.outOfScopePaths.length === 0))),
    ...(arm === "kontext"
      ? {
          contextConsultationRate: mean(runs.map((run) => Number(run.contextConsulted))),
        }
      : {}),
    meanDurationMilliseconds: mean(eligibleRuns.map((run) => run.durationMilliseconds)),
    ...optionalMean(
      "meanInputTokens",
      eligibleRuns.map((run) => run.inputTokens),
    ),
    ...optionalMean(
      "meanOutputTokens",
      eligibleRuns.map((run) => run.outputTokens),
    ),
  };
  return summary;
}

/**
 * The retrieval arm separates holding the policy from the governance workflow
 * around it, so kontext against rag is the comparison that isolates the
 * workflow while kontext against baseline measures both together.
 */
export function compareArms(
  runs: readonly CodeQualityRunResult[],
  treatment: CodeQualityArm,
  control: CodeQualityArm,
): ArmComparison {
  const grouped = groupByPair(runs);
  let treatmentWins = 0;
  let controlWins = 0;
  let ties = 0;
  let pairs = 0;
  for (const pair of grouped.values()) {
    const left = pair[treatment];
    const right = pair[control];
    if (!left?.evaluationEligible || !right?.evaluationEligible) continue;
    pairs += 1;
    const comparison = compareRuns(left, right);
    if (comparison > 0) treatmentWins += 1;
    else if (comparison < 0) controlWins += 1;
    else ties += 1;
  }
  const nonTies = treatmentWins + controlWins;
  return {
    treatment,
    control,
    pairs,
    treatmentWins,
    controlWins,
    ties,
    hiddenAssertionUplift:
      summarizeArm(treatment, runs).hiddenAssertionPassRate -
      summarizeArm(control, runs).hiddenAssertionPassRate,
    ...(nonTies === 0
      ? {}
      : { twoSidedSignTestPValue: exactTwoSidedSignTest(treatmentWins, controlWins) }),
  };
}

function groupByPair(
  runs: readonly CodeQualityRunResult[],
): Map<string, Partial<Record<CodeQualityArm, CodeQualityRunResult>>> {
  const pairs = new Map<string, Partial<Record<CodeQualityArm, CodeQualityRunResult>>>();
  for (const run of runs) {
    const key = `${run.scenarioId}\u0000${run.repetition}`;
    const pair = pairs.get(key) ?? {};
    pair[run.arm] = run;
    pairs.set(key, pair);
  }
  return pairs;
}

export function summarizePairs(runs: readonly CodeQualityRunResult[]): PairedOutcomeSummary {
  const pairs = new Map<string, Partial<Record<CodeQualityArm, CodeQualityRunResult>>>();
  for (const run of runs) {
    const key = `${run.scenarioId}\u0000${run.repetition}`;
    const pair = pairs.get(key) ?? {};
    pair[run.arm] = run;
    pairs.set(key, pair);
  }
  let kontextWins = 0;
  let baselineWins = 0;
  let ties = 0;
  let completePairs = 0;
  for (const pair of pairs.values()) {
    if (!pair.baseline?.evaluationEligible || !pair.kontext?.evaluationEligible) continue;
    completePairs += 1;
    const comparison = compareRuns(pair.kontext, pair.baseline);
    if (comparison > 0) kontextWins += 1;
    else if (comparison < 0) baselineWins += 1;
    else ties += 1;
  }
  const nonTies = kontextWins + baselineWins;
  return {
    pairs: completePairs,
    kontextWins,
    baselineWins,
    ties,
    ...(nonTies === 0
      ? {}
      : { twoSidedSignTestPValue: exactTwoSidedSignTest(kontextWins, baselineWins) }),
  };
}

export function exactTwoSidedSignTest(successes: number, failures: number): number {
  const trials = successes + failures;
  if (trials === 0) return 1;
  const tailEnd = Math.min(successes, failures);
  let probability = 2 ** -trials;
  let cumulative = probability;
  for (let successesInTail = 1; successesInTail <= tailEnd; successesInTail += 1) {
    probability *= (trials - successesInTail + 1) / successesInTail;
    cumulative += probability;
  }
  return Math.min(1, 2 * cumulative);
}

export function renderCodeQualityMarkdown(report: CodeQualityReport): string {
  const baseline = report.summaries.find((summary) => summary.arm === "baseline");
  const kontext = report.summaries.find((summary) => summary.arm === "kontext");
  if (!baseline || !kontext) throw new Error("The baseline and kontext summaries are required");
  const percentage = (value: number): string => `${(value * 100).toFixed(1)}%`;
  const tokens = (value: number | undefined): string =>
    value === undefined ? "n/a" : Math.round(value).toLocaleString("en-US");
  const pValue =
    report.paired.twoSidedSignTestPValue === undefined
      ? "n/a"
      : report.paired.twoSidedSignTestPValue.toFixed(4);
  return `# Kontext Brain code-quality A/B evaluation

Generated: ${report.generatedAt}

This is a **${report.evidenceStrength}** evaluation. Verdict: **${report.verdict}**.

| Metric | Baseline | Kontext | Delta |
|---|---:|---:|---:|
| Eligible runs | ${baseline.eligibleRuns}/${baseline.runs} | ${kontext.eligibleRuns}/${kontext.runs} | — |
| Runtime completion | ${percentage(baseline.runtimeCompletionRate)} | ${percentage(kontext.runtimeCompletionRate)} | ${percentage(kontext.runtimeCompletionRate - baseline.runtimeCompletionRate)} |
| Public test pass | ${percentage(baseline.publicTestPassRate)} | ${percentage(kontext.publicTestPassRate)} | ${percentage(kontext.publicTestPassRate - baseline.publicTestPassRate)} |
| Hidden policy assertions | ${percentage(baseline.hiddenAssertionPassRate)} | ${percentage(kontext.hiddenAssertionPassRate)} | ${percentage(report.hiddenAssertionUplift)} |
| Whole-task success | ${percentage(baseline.taskSuccessRate)} | ${percentage(kontext.taskSuccessRate)} | ${percentage(report.taskSuccessUplift)} |
| Canonical domain terms | ${percentage(baseline.canonicalTermPassRate)} | ${percentage(kontext.canonicalTermPassRate)} | ${percentage(kontext.canonicalTermPassRate - baseline.canonicalTermPassRate)} |
| Exact-path scope | ${percentage(baseline.scopeComplianceRate)} | ${percentage(kontext.scopeComplianceRate)} | ${percentage(kontext.scopeComplianceRate - baseline.scopeComplianceRate)} |
| Mean input tokens | ${tokens(baseline.meanInputTokens)} | ${tokens(kontext.meanInputTokens)} | — |
| Mean output tokens | ${tokens(baseline.meanOutputTokens)} | ${tokens(kontext.meanOutputTokens)} | — |
| Mean duration | ${Math.round(baseline.meanDurationMilliseconds)} ms | ${Math.round(kontext.meanDurationMilliseconds)} ms | ${Math.round(kontext.meanDurationMilliseconds - baseline.meanDurationMilliseconds)} ms |

Paired outcomes: Kontext ${report.paired.kontextWins} wins, baseline ${report.paired.baselineWins} wins, ${report.paired.ties} ties across ${report.paired.pairs} pairs (two-sided exact sign-test p=${pValue}).

Kontext context consultation rate: ${percentage(kontext.contextConsultationRate ?? 0)}.

## Per-run evidence

| Scenario | Repetition | Arm | Eligible | Hidden | Public | Terms | Scope | Context | Duration |
|---|---:|---|---|---:|---|---|---|---|---:|
${report.runs
  .map((run) => {
    const hiddenPassed = run.hiddenAssertions.filter((assertion) => assertion.passed).length;
    return `| ${run.scenarioId} | ${run.repetition} | ${run.arm} | ${run.evaluationEligible ? "yes" : "no"} | ${hiddenPassed}/${run.hiddenAssertions.length} | ${run.publicTestsPassed ? "pass" : "fail"} | ${run.canonicalTermsMissing.length === 0 ? "pass" : `missing ${run.canonicalTermsMissing.join(", ")}`} | ${run.outOfScopePaths.length === 0 ? "pass" : `fail: ${run.outOfScopePaths.join(", ")}`} | ${run.contextConsulted ? "yes" : "no"} | ${Math.round(run.durationMilliseconds)} ms |`;
  })
  .join("\n")}

## Limitations

${report.limitations.map((limitation) => `- ${limitation}`).join("\n")}
`;
}

function isSuccessfulRun(run: CodeQualityRunResult): boolean {
  return (
    run.runtimeExitCode === 0 &&
    run.publicTestsPassed &&
    run.hiddenAssertions.length > 0 &&
    run.hiddenAssertions.every((assertion) => assertion.passed) &&
    run.canonicalTermsMissing.length === 0 &&
    run.outOfScopePaths.length === 0
  );
}

function compareRuns(left: CodeQualityRunResult, right: CodeQualityRunResult): number {
  const leftScore = runScore(left);
  const rightScore = runScore(right);
  for (let index = 0; index < leftScore.length; index += 1) {
    const delta = (leftScore[index] ?? 0) - (rightScore[index] ?? 0);
    if (delta !== 0) return Math.sign(delta);
  }
  return 0;
}

function runScore(run: CodeQualityRunResult): readonly number[] {
  const hiddenRate = mean(run.hiddenAssertions.map((assertion) => Number(assertion.passed)));
  const totalTerms = run.canonicalTermsPresent.length + run.canonicalTermsMissing.length;
  return [
    hiddenRate,
    Number(isSuccessfulRun(run)),
    totalTerms === 0 ? 0 : run.canonicalTermsPresent.length / totalTerms,
    Number(run.outOfScopePaths.length === 0),
    Number(run.publicTestsPassed),
    Number(run.runtimeExitCode === 0),
  ];
}

function classifyEvidenceStrength(
  scenarioCount: number,
  pairCount: number,
): CodeQualityReport["evidenceStrength"] {
  if (scenarioCount >= 10 && pairCount >= 100) return "release";
  if (scenarioCount >= 10 && pairCount >= 30) return "pilot";
  return "smoke";
}

function verdict(input: {
  readonly evidenceStrength: CodeQualityReport["evidenceStrength"];
  readonly hiddenAssertionUplift: number;
  readonly taskSuccessUplift: number;
  readonly paired: PairedOutcomeSummary;
}): CodeQualityReport["verdict"] {
  if (input.evidenceStrength !== "release") return "inconclusive";
  const significant = (input.paired.twoSidedSignTestPValue ?? 1) <= 0.05;
  if (significant && input.hiddenAssertionUplift > 0 && input.taskSuccessUplift >= 0) {
    return "improvement";
  }
  if (significant && input.hiddenAssertionUplift < 0 && input.taskSuccessUplift <= 0) {
    return "regression";
  }
  return "no_detected_difference";
}

function limitations(
  evidenceStrength: CodeQualityReport["evidenceStrength"],
  scenarioCount: number,
  pairCount: number,
): readonly string[] {
  const values = [
    "The hidden evaluator measures deterministic functional and terminology requirements, not general maintainability or human preference.",
    "Codex generation is stochastic; repeated paired runs are required before a release claim.",
    "The treatment has extra MCP latency and tokens, so quality and cost must be considered together.",
  ];
  if (evidenceStrength !== "release") {
    values.unshift(
      `Only ${scenarioCount} scenarios and ${pairCount} complete pairs were measured; this can show directional behavior but cannot establish a statistically supported release claim.`,
    );
  }
  return values;
}

function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function optionalMean<Key extends "meanInputTokens" | "meanOutputTokens">(
  key: Key,
  values: readonly (number | undefined)[],
): Partial<Record<Key, number>> {
  const present = values.filter((value): value is number => value !== undefined);
  return present.length === 0 ? {} : ({ [key]: mean(present) } as Partial<Record<Key, number>>);
}
