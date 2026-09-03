import type { CodeQualityArm } from "../contracts.js";
import type {
  RealOssArmSummary,
  RealOssReport,
  RealOssRunConfig,
  RealOssRunResult,
  RealOssTask,
} from "./contracts.js";

export function buildRealOssReport(input: {
  readonly task: RealOssTask;
  readonly config: RealOssRunConfig;
  readonly runs: readonly RealOssRunResult[];
  readonly generatedAt?: string;
}): RealOssReport {
  return {
    schemaVersion: 1,
    benchmark: "real-oss-code-quality",
    generatedAt: input.generatedAt ?? new Date().toISOString(),
    task: {
      instanceId: input.task.instanceId,
      repository: input.task.repository,
      repositoryUrl: input.task.repositoryUrl,
      license: input.task.license,
      baseCommit: input.task.baseCommit,
      upstreamIssueUrl: input.task.upstreamIssueUrl,
      upstreamPullRequestUrl: input.task.upstreamPullRequestUrl,
      hiddenTestPatchSha256: input.task.hiddenTest.patchSha256,
    },
    config: {
      runtime: input.config.runtime,
      model: input.config.model,
      reasoningEffort: input.config.reasoningEffort,
      repetitions: input.config.repetitions,
      timeoutMilliseconds: input.config.timeoutMilliseconds,
      arms: input.config.arms,
      cacheDirectory: input.config.cacheDirectory,
    },
    summaries: input.config.arms.map((arm) => summarizeArm(arm, input.runs)),
    runs: input.runs,
    evidenceStrength: "smoke",
    limitations: [
      "This run covers one real SWE-bench Verified task from one public library; it is a smoke test, not an external-validity claim.",
      "The task is an upstream historical replay. The agent sees the pinned pre-fix commit, while the grader alone applies the upstream regression-test patch.",
      "The public issue states the requested behavior directly, so this task validates real-repository integration and governance discipline more than difficult knowledge retrieval.",
      "Subscription runtime load and model nondeterminism remain uncontrolled; arm order rotates between repetitions.",
    ],
  };
}

export function summarizeArm(
  arm: CodeQualityArm,
  runs: readonly RealOssRunResult[],
): RealOssArmSummary {
  const all = runs.filter((run) => run.arm === arm);
  const eligible = all.filter((run) => run.evaluationEligible);
  return {
    arm,
    runs: all.length,
    eligibleRuns: eligible.length,
    taskSuccessRate: mean(eligible.map((run) => Number(run.taskSuccess))),
    failToPassRate: mean(
      eligible.map((run) => ratio(run.grade.failToPassPassed, run.grade.failToPassTotal)),
    ),
    passToPassRate: mean(
      eligible.map((run) => ratio(run.grade.passToPassPassed, run.grade.passToPassTotal)),
    ),
    allowedPathRate: mean(eligible.map((run) => Number(run.grade.allowedPathsOnly))),
    ...(arm === "kontext"
      ? { contextConsultationRate: mean(all.map((run) => Number(run.contextConsulted))) }
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

export function renderRealOssMarkdown(report: RealOssReport): string {
  const percentage = (value: number): string => `${(value * 100).toFixed(1)}%`;
  const token = (value: number | undefined): string =>
    value === undefined ? "n/a" : Math.round(value).toLocaleString("en-US");
  const ontology = report.runs.find((run) => run.ontology)?.ontology;
  return `# Kontext Brain real-OSS code-quality benchmark

Generated: ${report.generatedAt}

Repository: [${report.task.repository}](${report.task.repositoryUrl.replace(/\.git$/, "")}) at \`${report.task.baseCommit}\` (${report.task.license}).
Task: [${report.task.instanceId}](${report.task.upstreamIssueUrl}); upstream fix: [PR](${report.task.upstreamPullRequestUrl}).
Evidence strength: **${report.evidenceStrength}**.

| Arm | Eligible | Success | FAIL_TO_PASS | PASS_TO_PASS | Allowed paths | Context | Tokens in/out | Duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
${report.summaries
  .map(
    (summary) =>
      `| ${summary.arm} | ${summary.eligibleRuns}/${summary.runs} | ${percentage(summary.taskSuccessRate)} | ${percentage(summary.failToPassRate)} | ${percentage(summary.passToPassRate)} | ${percentage(summary.allowedPathRate)} | ${summary.contextConsultationRate === undefined ? "n/a" : percentage(summary.contextConsultationRate)} | ${token(summary.meanInputTokens)}/${token(summary.meanOutputTokens)} | ${Math.round(summary.meanDurationMilliseconds)} ms |`,
  )
  .join("\n")}

## Ontology ingestion

${
  ontology
    ? `The Kontext arm indexed ${ontology.codeResources} Python code Resources, ${ontology.codeSymbols} symbols (${ontology.behaviorBearingSymbols} behavior-bearing), ${ontology.provenanceResources} provenance Resources, and ${ontology.normativeRecords} effective normative records. Target: \`${ontology.targetQualifiedName}\` (\`${ontology.targetSymbolId}\`). Governing records: ${ontology.governingRecordIds.map((id) => `\`${id}\``).join(", ")}.`
    : "No eligible Kontext ontology assembly was recorded."
}

## Per-run evidence

| Repetition | Arm | Eligible | Success | Public | F2P | P2P | Paths | Logic context | Changed files |
|---:|---|---|---|---|---:|---:|---|---:|---|
${report.runs
  .map(
    (run) =>
      `| ${run.repetition} | ${run.arm} | ${run.evaluationEligible ? "yes" : "no"} | ${run.taskSuccess ? "yes" : "no"} | ${run.grade.publicTestsPassed ? "pass" : "fail"} | ${run.grade.failToPassPassed}/${run.grade.failToPassTotal} | ${run.grade.passToPassPassed}/${run.grade.passToPassTotal} | ${run.grade.allowedPathsOnly ? "pass" : "fail"} | ${run.observedLogicConsultations}/${run.expectedLogicConsultations} | ${run.grade.changedFiles.map((file) => `\`${file}\``).join(", ")} |`,
  )
  .join("\n")}

## Limitations

${report.limitations.map((limitation) => `- ${limitation}`).join("\n")}
`;
}

function ratio(numerator: number, denominator: number): number {
  return denominator === 0 ? 0 : numerator / denominator;
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
