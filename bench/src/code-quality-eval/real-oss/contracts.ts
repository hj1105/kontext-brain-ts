import type { KontextToolCall } from "../codex-runner.js";
import type { CodeQualityArm, CodeQualityRunConfig } from "../contracts.js";

export interface RealOssSourceDocument {
  readonly documentId: string;
  readonly kind: "issue" | "pull_request" | "documentation";
  readonly title: string;
  readonly body: string;
  readonly sourceUrl: string;
  readonly sourceSpan: string;
  readonly observedAt: string;
  readonly ontologyNodeIds: readonly string[];
}

export type RealOssNormativeRecord =
  | {
      readonly kind: "decision";
      readonly recordId: string;
      readonly revisionId: string;
      readonly statement: string;
      readonly evidenceIds: readonly string[];
      readonly ontologyNodeIds: readonly string[];
    }
  | {
      readonly kind: "domain_term";
      readonly recordId: string;
      readonly revisionId: string;
      readonly term: string;
      readonly definition: string;
      readonly avoid: readonly string[];
      readonly evidenceIds: readonly string[];
      readonly ontologyNodeIds: readonly string[];
    }
  | {
      readonly kind: "invariant";
      readonly recordId: string;
      readonly revisionId: string;
      readonly statement: string;
      readonly evidenceIds: readonly string[];
      readonly ontologyNodeIds: readonly string[];
    };

export interface RealOssTaskTarget {
  readonly workItemId: string;
  readonly plannedSymbolId: string;
  readonly relativePath: string;
  readonly qualifiedName: string;
  readonly symbolKind: "function" | "method";
  readonly binding: "required" | "planned";
  readonly responsibility: string;
  readonly ontologyNodeIds: readonly string[];
  readonly dependsOn?: readonly string[];
  readonly capabilityId: string;
}

export type RealOssHiddenTestRunner =
  | {
      readonly kind: "pytest-selectors";
      readonly command: string;
      readonly args: readonly string[];
    }
  | {
      readonly kind: "django-selectors";
      readonly command: string;
      readonly args: readonly string[];
    };

export interface RealOssTask {
  readonly instanceId: string;
  readonly taskId: string;
  readonly codebaseId: string;
  readonly repository: string;
  readonly repositoryUrl: string;
  readonly license: string;
  readonly baseCommit: string;
  readonly upstreamIssueUrl: string;
  readonly upstreamPullRequestUrl: string;
  readonly publicPrompt: string;
  readonly acceptanceStatement: string;
  readonly nonGoals: readonly string[];
  readonly risk: "low" | "medium" | "high";
  readonly codeRoots: readonly string[];
  readonly allowedPaths: readonly string[];
  readonly targets: readonly RealOssTaskTarget[];
  readonly sourceIntegrity: readonly {
    readonly relativePath: string;
    readonly sha256: string;
  }[];
  readonly environment: {
    readonly pythonVersion: string;
    readonly packages: readonly string[];
  };
  readonly publicTest: {
    readonly command: string;
    readonly args: readonly string[];
  };
  readonly hiddenTest: {
    readonly patch: string;
    readonly patchSha256: string;
    readonly runner: RealOssHiddenTestRunner;
    readonly failToPass: readonly string[];
    readonly passToPass: readonly string[];
  };
  readonly sourceDocuments: readonly RealOssSourceDocument[];
  readonly normativeRecords: readonly RealOssNormativeRecord[];
  readonly limitations: readonly string[];
}

export interface RealOssWorkspace {
  readonly workspacePath: string;
  readonly baseRevision: string;
}

export interface RealOssOntologyStats {
  readonly codeResources: number;
  readonly codeSymbols: number;
  readonly behaviorBearingSymbols: number;
  readonly provenanceResources: number;
  readonly normativeRecords: number;
  readonly targetSymbols: readonly {
    readonly qualifiedName: string;
    readonly binding: RealOssTaskTarget["binding"];
    readonly symbolId?: string;
  }[];
  readonly governingRecordIds: readonly string[];
}

export interface RealOssLogicTarget {
  readonly workItemId: string;
  readonly plannedSymbolId: string;
}

export interface RealOssGrade {
  readonly publicTestsPassed: boolean;
  readonly targetChanged: boolean;
  readonly allowedPathsOnly: boolean;
  readonly changedFiles: readonly string[];
  readonly failToPassPassed: number;
  readonly failToPassTotal: number;
  readonly passToPassPassed: number;
  readonly passToPassTotal: number;
  readonly hiddenPatchApplied: boolean;
  readonly hiddenFailures: readonly string[];
  readonly patch: string;
}

export interface RealOssRunConfig extends CodeQualityRunConfig {
  readonly arms: readonly CodeQualityArm[];
  readonly sourceRepositoryPath?: string;
  readonly cacheDirectory: string;
}

export interface RealOssRunResult {
  readonly runId: string;
  readonly instanceId: string;
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
  readonly ontology?: RealOssOntologyStats;
  readonly retrievedDocumentIds?: readonly string[];
  readonly grade: RealOssGrade;
}

export interface RealOssArmSummary {
  readonly arm: CodeQualityArm;
  readonly runs: number;
  readonly eligibleRuns: number;
  readonly taskSuccessRate: number;
  readonly failToPassRate: number;
  readonly passToPassRate: number;
  readonly allowedPathRate: number;
  readonly contextConsultationRate?: number;
  readonly meanDurationMilliseconds: number;
  readonly meanInputTokens?: number;
  readonly meanOutputTokens?: number;
}

export interface RealOssReport {
  readonly schemaVersion: 2;
  readonly benchmark: "real-oss-code-quality";
  readonly generatedAt: string;
  readonly task: {
    readonly instanceId: string;
    readonly repository: string;
    readonly repositoryUrl: string;
    readonly license: string;
    readonly baseCommit: string;
    readonly upstreamIssueUrl: string;
    readonly upstreamPullRequestUrl: string;
    readonly hiddenTestPatchSha256: string;
  };
  readonly config: Omit<RealOssRunConfig, "sourceRepositoryPath">;
  readonly summaries: readonly RealOssArmSummary[];
  readonly runs: readonly RealOssRunResult[];
  readonly evidenceStrength: "smoke" | "pilot" | "release";
  readonly limitations: readonly string[];
}
