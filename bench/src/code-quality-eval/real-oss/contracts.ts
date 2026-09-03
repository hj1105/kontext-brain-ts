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
  readonly allowedPaths: readonly string[];
  readonly target: {
    readonly workItemId: string;
    readonly plannedSymbolId: string;
    readonly relativePath: string;
    readonly qualifiedName: string;
    readonly responsibility: string;
    readonly ontologyNodeIds: readonly string[];
  };
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
    readonly failToPass: readonly string[];
    readonly passToPass: readonly string[];
  };
  readonly sourceDocuments: readonly RealOssSourceDocument[];
  readonly normativeRecords: readonly RealOssNormativeRecord[];
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
  readonly targetSymbolId: string;
  readonly targetQualifiedName: string;
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
  readonly schemaVersion: 1;
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
