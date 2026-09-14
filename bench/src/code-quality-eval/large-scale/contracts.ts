import type { KontextToolCall } from "../codex-runner.js";
import type { CodeQualityArm, CodeQualityRunConfig } from "../contracts.js";
import type { LargeScaleGrade } from "./workspace.js";

export interface LargeScaleRunConfig extends CodeQualityRunConfig {
  readonly arms: readonly CodeQualityArm[];
}

export interface LargeScaleRunResult {
  readonly runId: string;
  readonly repetition: number;
  readonly arm: CodeQualityArm;
  readonly model: string;
  readonly reasoningEffort: CodeQualityRunConfig["reasoningEffort"];
  readonly startedAt: string;
  readonly finishedAt: string;
  readonly durationMilliseconds: number;
  readonly runtimeExitCode: number;
  readonly runtimeDiagnostic?: string;
  readonly inputTokens?: number;
  readonly outputTokens?: number;
  readonly kontextToolCalls: readonly KontextToolCall[];
  readonly expectedLogicConsultations: number;
  readonly observedLogicConsultations: number;
  readonly contextConsulted: boolean;
  readonly evaluationEligible: boolean;
  readonly taskSuccess: boolean;
  readonly retrieval?: {
    readonly documentIds: readonly string[];
    readonly governingRetrieved: number;
    readonly governingTotal: number;
  };
  readonly grade: LargeScaleGrade;
}

export interface LargeScaleArmSummary {
  readonly arm: CodeQualityArm;
  readonly runs: number;
  readonly eligibleRuns: number;
  readonly runtimeCompletionRate: number;
  readonly taskSuccessRate: number;
  readonly meanTargetRecall: number;
  readonly meanCollateralPrecision: number;
  readonly meanHiddenPassRate: number;
  readonly regressionFreeRate: number;
  readonly canonicalTermPassRate: number;
  readonly sharedConstantPassRate: number;
  readonly contextConsultationRate?: number;
  readonly meanDurationMilliseconds: number;
  readonly meanInputTokens?: number;
  readonly meanOutputTokens?: number;
}

export interface LargeScalePairSummary {
  readonly treatment: CodeQualityArm;
  readonly control: CodeQualityArm;
  readonly pairs: number;
  readonly treatmentWins: number;
  readonly controlWins: number;
  readonly ties: number;
  readonly twoSidedSignTestPValue?: number;
}

export interface LargeScaleReport {
  readonly schemaVersion: 1;
  readonly benchmark: "large-scale-code-quality";
  readonly generatedAt: string;
  readonly config: LargeScaleRunConfig;
  readonly summaries: readonly LargeScaleArmSummary[];
  readonly comparisons: readonly LargeScalePairSummary[];
  readonly runs: readonly LargeScaleRunResult[];
  readonly evidenceStrength: "smoke" | "pilot" | "release";
  readonly limitations: readonly string[];
}
