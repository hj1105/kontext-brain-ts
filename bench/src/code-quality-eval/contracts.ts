export type CodeQualityArm = "baseline" | "rag" | "kontext";

export interface CodeQualityNormativeRule {
  readonly kind: "decision" | "domain_term" | "invariant";
  readonly recordId: string;
  readonly revisionId: string;
  readonly statement?: string;
  readonly term?: string;
  readonly definition?: string;
  readonly avoid?: readonly string[];
  readonly evidenceId: string;
  readonly evidenceText: string;
}

export interface HiddenAssertionResult {
  readonly assertionId: string;
  readonly passed: boolean;
  readonly diagnostic?: string;
}

export interface HiddenEvaluationResult {
  readonly assertions: readonly HiddenAssertionResult[];
}

/**
 * A Python scenario cannot hand a JavaScript module to evaluateHidden, so its
 * held-out checks are declared as data and executed by a Python driver.
 */
export interface PythonHiddenCheck {
  readonly assertionId: string;
  readonly functionName: string;
  readonly args: readonly unknown[];
  readonly expected?: unknown;
  readonly throws?: string;
}

export interface CodeQualityScenario {
  readonly scenarioId: string;
  readonly taskId: string;
  readonly intent: string;
  readonly publicPrompt: string;
  readonly sourceFile: string;
  readonly initialSource: string;
  readonly publicTestSource: string;
  /** Defaults to test/public.test.js; Python scenarios use test/public_test.py. */
  readonly testFile?: string;
  /** Required for a Python scenario, which cannot use evaluateHidden. */
  readonly hiddenChecks?: readonly PythonHiddenCheck[];
  readonly workItemId: string;
  readonly plannedSymbolId: string;
  readonly qualifiedName: string;
  readonly capabilityId: string;
  readonly canonicalTerms: readonly string[];
  readonly rules: readonly CodeQualityNormativeRule[];
  evaluateHidden(module: Readonly<Record<string, unknown>>): Promise<HiddenEvaluationResult>;
}

export type CodeQualityRuntime = "codex" | "claude";

export interface CodeQualityRunConfig {
  readonly runtime: CodeQualityRuntime;
  readonly model: string;
  readonly reasoningEffort: "low" | "medium" | "high" | "xhigh";
  readonly repetitions: number;
  readonly timeoutMilliseconds: number;
}

export interface CodeQualityRunResult {
  readonly runId: string;
  readonly scenarioId: string;
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
  readonly publicTestsPassed: boolean;
  readonly hiddenAssertions: readonly HiddenAssertionResult[];
  readonly canonicalTermsPresent: readonly string[];
  readonly canonicalTermsMissing: readonly string[];
  readonly changedPaths: readonly string[];
  readonly outOfScopePaths: readonly string[];
  readonly kontextToolsObserved: readonly string[];
  readonly contextConsulted: boolean;
  readonly evaluationEligible: boolean;
  readonly source: string;
  readonly patch: string;
}

export interface ArmSummary {
  readonly arm: CodeQualityArm;
  readonly runs: number;
  readonly eligibleRuns: number;
  readonly runtimeCompletionRate: number;
  readonly publicTestPassRate: number;
  readonly hiddenAssertionPassRate: number;
  readonly taskSuccessRate: number;
  readonly canonicalTermPassRate: number;
  readonly scopeComplianceRate: number;
  readonly contextConsultationRate?: number;
  readonly meanDurationMilliseconds: number;
  readonly meanInputTokens?: number;
  readonly meanOutputTokens?: number;
}

export interface PairedOutcomeSummary {
  readonly pairs: number;
  readonly kontextWins: number;
  readonly baselineWins: number;
  readonly ties: number;
  readonly twoSidedSignTestPValue?: number;
}

export interface ArmComparison {
  readonly treatment: CodeQualityArm;
  readonly control: CodeQualityArm;
  readonly pairs: number;
  readonly treatmentWins: number;
  readonly controlWins: number;
  readonly ties: number;
  readonly hiddenAssertionUplift: number;
  readonly twoSidedSignTestPValue?: number;
}

export interface CodeQualityReport {
  readonly schemaVersion: 1;
  readonly generatedAt: string;
  readonly config: CodeQualityRunConfig;
  readonly scenarios: readonly string[];
  readonly runs: readonly CodeQualityRunResult[];
  readonly summaries: readonly ArmSummary[];
  readonly paired: PairedOutcomeSummary;
  readonly comparisons: readonly ArmComparison[];
  readonly hiddenAssertionUplift: number;
  readonly taskSuccessUplift: number;
  readonly evidenceStrength: "smoke" | "pilot" | "release";
  readonly verdict: "improvement" | "regression" | "no_detected_difference" | "inconclusive";
  readonly limitations: readonly string[];
}
