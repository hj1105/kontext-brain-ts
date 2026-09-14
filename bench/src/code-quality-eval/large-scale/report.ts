import type { CodeQualityArm } from "../contracts.js";
import { exactTwoSidedSignTest } from "../report.js";
import type {
  LargeScaleArmSummary,
  LargeScalePairSummary,
  LargeScaleReport,
  LargeScaleRunConfig,
  LargeScaleRunResult,
} from "./contracts.js";

export function buildLargeScaleReport(input: {
  readonly config: LargeScaleRunConfig;
  readonly runs: readonly LargeScaleRunResult[];
  readonly generatedAt?: string;
}): LargeScaleReport {
  const comparisons = [
    compareLargeScaleArms(input.runs, "kontext", "baseline"),
    compareLargeScaleArms(input.runs, "kontext", "rag"),
    compareLargeScaleArms(input.runs, "rag", "baseline"),
  ].filter((comparison) => comparison.pairs > 0);
  const primaryPairs =
    comparisons.find(
      (comparison) => comparison.treatment === "kontext" && comparison.control === "baseline",
    )?.pairs ?? 0;
  const evidenceStrength = primaryPairs >= 20 ? "release" : primaryPairs >= 3 ? "pilot" : "smoke";
  return {
    schemaVersion: 1,
    benchmark: "large-scale-code-quality",
    generatedAt: input.generatedAt ?? new Date().toISOString(),
    config: input.config,
    summaries: input.config.arms.map((arm) => summarizeLargeScaleArm(arm, input.runs)),
    comparisons,
    runs: input.runs,
    evidenceStrength,
    limitations: [
      "The benchmark uses one generated repository family; release claims require additional independently authored repositories and tasks.",
      "The hidden grader knows the generator manifest, while agents can see only the fixture repository and their assigned context surface.",
      "Subscription runtime load and model nondeterminism remain uncontrolled; arm order rotates between repetitions.",
      ...(input.config.arms.includes("rag")
        ? [
            "The RAG control embeds realistic source documents, while Kontext consumes provenance-linked normative records extracted from those sources.",
          ]
        : [
            "The RAG control was not run, so this report cannot isolate retrieval quality from governance workflow quality.",
          ]),
    ],
  };
}

export function summarizeLargeScaleArm(
  arm: CodeQualityArm,
  runs: readonly LargeScaleRunResult[],
): LargeScaleArmSummary {
  const all = runs.filter((run) => run.arm === arm);
  const eligible = all.filter((run) => run.evaluationEligible);
  return {
    arm,
    runs: all.length,
    eligibleRuns: eligible.length,
    runtimeCompletionRate: mean(all.map((run) => Number(run.runtimeExitCode === 0))),
    taskSuccessRate: mean(eligible.map((run) => Number(run.taskSuccess))),
    meanTargetRecall: mean(eligible.map((run) => run.grade.targetRecall)),
    meanCollateralPrecision: mean(eligible.map((run) => run.grade.collateralPrecision)),
    meanHiddenPassRate: mean(
      eligible.map((run) =>
        run.grade.hiddenTotal === 0 ? 0 : run.grade.hiddenPassed / run.grade.hiddenTotal,
      ),
    ),
    regressionFreeRate: mean(eligible.map((run) => Number(run.grade.regressionFailures === 0))),
    canonicalTermPassRate: mean(eligible.map((run) => Number(run.grade.canonicalTermPresent))),
    sharedConstantPassRate: mean(eligible.map((run) => Number(run.grade.sharedConstantHonoured))),
    ...(arm === "kontext"
      ? {
          contextConsultationRate: mean(all.map((run) => Number(run.contextConsulted))),
        }
      : {}),
    meanDurationMilliseconds: mean(eligible.map((run) => run.durationMilliseconds)),
    ...optionalMean(
      "meanInputTokens",
      eligible.map((run) => run.inputTokens),
    ),
    ...optionalMean(
      "meanOutputTokens",
      eligible.map((run) => run.outputTokens),
    ),
  };
}

export function compareLargeScaleArms(
  runs: readonly LargeScaleRunResult[],
  treatment: CodeQualityArm,
  control: CodeQualityArm,
): LargeScalePairSummary {
  const byRepetition = new Map<number, Partial<Record<CodeQualityArm, LargeScaleRunResult>>>();
  for (const run of runs) {
    const group = byRepetition.get(run.repetition) ?? {};
    group[run.arm] = run;
    byRepetition.set(run.repetition, group);
  }
  let treatmentWins = 0;
  let controlWins = 0;
  let ties = 0;
  let pairs = 0;
  for (const group of byRepetition.values()) {
    const left = group[treatment];
    const right = group[control];
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
    ...(nonTies === 0
      ? {}
      : { twoSidedSignTestPValue: exactTwoSidedSignTest(treatmentWins, controlWins) }),
  };
}

export function renderLargeScaleMarkdown(report: LargeScaleReport): string {
  const percentage = (value: number): string => `${(value * 100).toFixed(1)}%`;
  const token = (value: number | undefined): string =>
    value === undefined ? "n/a" : Math.round(value).toLocaleString("en-US");
  return `# Kontext Brain large-scale code-quality benchmark

Generated: ${report.generatedAt}

Evidence strength: **${report.evidenceStrength}**. A successful run requires full governed-symbol recall, zero collateral edits, all held-out behavior checks, no regressions, the canonical term, and one shared policy constant.

| Arm | Eligible | Success | Target recall | Precision | Hidden | Regression-free | Domain term | Shared constant | Context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
${report.summaries
  .map(
    (summary) =>
      `| ${summary.arm} | ${summary.eligibleRuns}/${summary.runs} | ${percentage(summary.taskSuccessRate)} | ${percentage(summary.meanTargetRecall)} | ${percentage(summary.meanCollateralPrecision)} | ${percentage(summary.meanHiddenPassRate)} | ${percentage(summary.regressionFreeRate)} | ${percentage(summary.canonicalTermPassRate)} | ${percentage(summary.sharedConstantPassRate)} | ${summary.contextConsultationRate === undefined ? "n/a" : percentage(summary.contextConsultationRate)} |`,
  )
  .join("\n")}

## Paired comparisons

| Treatment | Control | Pairs | Wins | Losses | Ties | Sign-test p |
|---|---|---:|---:|---:|---:|---:|
${report.comparisons
  .map(
    (comparison) =>
      `| ${comparison.treatment} | ${comparison.control} | ${comparison.pairs} | ${comparison.treatmentWins} | ${comparison.controlWins} | ${comparison.ties} | ${comparison.twoSidedSignTestPValue?.toFixed(4) ?? "n/a"} |`,
  )
  .join("\n")}

## Per-run evidence

| Repetition | Arm | Eligible | Success | Governed | Decoys | Hidden | Term | Shared | Logic context | Tokens in/out | Duration |
|---:|---|---|---|---:|---:|---:|---|---|---:|---:|---:|
${report.runs
  .map(
    (run) =>
      `| ${run.repetition} | ${run.arm} | ${run.evaluationEligible ? "yes" : "no"} | ${run.taskSuccess ? "yes" : "no"} | ${run.grade.governedChanged.length} | ${run.grade.decoysChanged.length} | ${run.grade.hiddenPassed}/${run.grade.hiddenTotal} | ${run.grade.canonicalTermPresent ? "yes" : "no"} | ${run.grade.sharedConstantHonoured ? "yes" : "no"} | ${run.observedLogicConsultations}/${run.expectedLogicConsultations} | ${token(run.inputTokens)}/${token(run.outputTokens)} | ${Math.round(run.durationMilliseconds)} ms |`,
  )
  .join("\n")}

## Limitations

${report.limitations.map((limitation) => `- ${limitation}`).join("\n")}
`;
}

function compareRuns(left: LargeScaleRunResult, right: LargeScaleRunResult): number {
  const leftScore = score(left);
  const rightScore = score(right);
  for (let index = 0; index < leftScore.length; index += 1) {
    const difference = (leftScore[index] ?? 0) - (rightScore[index] ?? 0);
    if (difference !== 0) return Math.sign(difference);
  }
  return 0;
}

function score(run: LargeScaleRunResult): readonly number[] {
  return [
    Number(run.taskSuccess),
    run.grade.hiddenTotal === 0 ? 0 : run.grade.hiddenPassed / run.grade.hiddenTotal,
    run.grade.targetRecall,
    run.grade.collateralPrecision,
    Number(run.grade.regressionFailures === 0),
    Number(run.grade.sharedConstantHonoured),
    Number(run.grade.canonicalTermPresent),
    Number(run.grade.publicTestsPassed),
  ];
}

function mean(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function optionalMean<Key extends "meanInputTokens" | "meanOutputTokens">(
  key: Key,
  values: readonly (number | undefined)[],
): Partial<Record<Key, number>> {
  const present = values.filter((value): value is number => value !== undefined);
  return present.length === 0 ? {} : ({ [key]: mean(present) } as Partial<Record<Key, number>>);
}
