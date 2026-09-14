import type {
  DeepSweArm,
  DeepSwePreparationManifest,
  DeepSweReport,
  DeepSweTrialResult,
} from "./contracts.js";
import { compareDeepSweArms, summarizeDeepSweArm } from "./metrics.js";

export function buildDeepSweReport(input: {
  readonly manifest: DeepSwePreparationManifest;
  readonly trials: readonly DeepSweTrialResult[];
  readonly generatedAt?: string;
}): DeepSweReport {
  const arms = input.manifest.arms.map((arm) => arm.arm);
  return {
    schemaVersion: 1,
    benchmark: "deepswe-kontext-ab",
    generatedAt: input.generatedAt ?? new Date().toISOString(),
    manifest: input.manifest,
    summaries: arms.map((arm) => summarizeDeepSweArm(arm, input.trials)),
    comparisons: controls(arms).map((control) =>
      compareDeepSweArms("kontext", control, input.trials),
    ),
    trials: input.trials,
    limitations: [
      "This changes the official mini-swe-agent scaffold by adding an arm-stable context command, so it is a DeepSWE-based paired A/B result rather than an official leaderboard score.",
      "Functional verifier success does not by itself measure maintainability, security, evidence correctness, or terminology consistency.",
      "Normative records are a derived Kontext projection; their evidence closure and snapshot date are validated, but task authorship independence still requires external review.",
    ],
  };
}

export function renderDeepSweMarkdown(report: DeepSweReport): string {
  const percentage = (value: number): string => `${(value * 100).toFixed(1)}%`;
  const interval = (value: readonly [number, number] | undefined): string =>
    value ? `${percentage(value[0])}–${percentage(value[1])}` : "n/a";
  return `# DeepSWE-based Kontext paired evaluation

- Generated: ${report.generatedAt}
- DeepSWE revision: \`${report.manifest.deepSweRevision}\`
- Pier revision: \`${report.manifest.pierRevision}\`
- Kontext adapter revision: \`${report.manifest.adapterRevision}\`
- Model: \`${report.manifest.model}\` at \`${report.manifest.reasoningEffort}\`
- Tasks: ${report.manifest.tasks.length}; attempts per task: ${report.manifest.attempts}

## Arm summaries

| Arm | Tasks | Eligible | Excluded | Protocol complete | pass@1 | pass@4 | Run-to-run 95% | Median output tokens | Median duration | Median cost | Median steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
${report.summaries
  .map(
    (summary) =>
      `| ${summary.arm} | ${summary.tasks} | ${summary.eligibleTrials} | ${summary.excludedTrials} | ${summary.protocolCompleteTrials}/${summary.eligibleTrials + summary.excludedTrials} | ${percentage(summary.passAt1)} | ${percentage(summary.passAt4)} | ${interval(summary.runToRun95)} | ${summary.medianOutputTokens ?? "n/a"} | ${summary.medianDurationMilliseconds === undefined ? "n/a" : `${Math.round(summary.medianDurationMilliseconds / 1000)} s`} | ${summary.medianCostUsd === undefined ? "n/a" : `$${summary.medianCostUsd.toFixed(4)}`} | ${summary.medianAgentSteps ?? "n/a"} |`,
  )
  .join("\n")}

## Paired comparisons

| Treatment | Control | Comparable tasks | pass@1 delta | pass@1 task-bootstrap 95% | pass@4 delta | pass@4 task-bootstrap 95% |
| --- | --- | ---: | ---: | --- | ---: | --- |
${report.comparisons
  .map(
    (comparison) =>
      `| ${comparison.treatment} | ${comparison.control} | ${comparison.comparableTasks} | ${percentage(comparison.passAt1Delta)} | ${interval(comparison.passAt1ClusterBootstrap95)} | ${percentage(comparison.passAt4Delta)} | ${interval(comparison.passAt4ClusterBootstrap95)} |`,
  )
  .join("\n")}

## Limitations

${report.limitations.map((limitation) => `- ${limitation}`).join("\n")}
`;
}

function controls(arms: readonly DeepSweArm[]): ("baseline" | "rag")[] {
  if (!arms.includes("kontext")) return [];
  return (["baseline", "rag"] as const).filter((arm) => arms.includes(arm));
}
