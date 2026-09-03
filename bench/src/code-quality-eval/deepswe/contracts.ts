import type { NormativeRevision } from "@kontext-brain/spec";
import type { CodeQualityArm } from "../contracts.js";

export type DeepSweArm = CodeQualityArm;

export interface DeepSweEvidenceSnapshot {
  readonly evidenceId: string;
  readonly resourceId: string;
  readonly chunkId: string;
  readonly title: string;
  readonly text: string;
  readonly sourceUri: string;
  readonly observedAt: string;
  readonly contentSha256: string;
  readonly ontologyNodeIds: readonly string[];
}

export interface DeepSweNormativeRecord {
  readonly revision: NormativeRevision;
  readonly symbolSelectors?: readonly {
    readonly relativePath?: string;
    readonly qualifiedName?: string;
  }[];
}

export interface DeepSweContextCorpus {
  readonly schemaVersion: 1;
  readonly taskId: string;
  readonly organizationId: string;
  readonly baseCodeRevision: string;
  readonly contextDigest: string;
  readonly sourceFreshnessDigest: string;
  readonly snapshotAt: string;
  readonly generator: {
    readonly name: "kontext-brain";
    readonly revision: string;
  };
  readonly evidence: readonly DeepSweEvidenceSnapshot[];
  readonly normativeRecords: readonly DeepSweNormativeRecord[];
}

export interface DeepSweContextBundle {
  readonly schemaVersion: 1;
  readonly arm: DeepSweArm;
  readonly taskId: string;
  readonly organizationId: string;
  readonly baseCodeRevision: string;
  readonly contextDigest: string;
  readonly sourceFreshnessDigest: string;
  readonly snapshotAt: string;
  readonly corpusSha256: string;
  readonly projectionSha256: string;
  readonly generator: DeepSweContextCorpus["generator"];
  readonly evidence: readonly DeepSweEvidenceSnapshot[];
  readonly normativeRecords: readonly DeepSweNormativeRecord[];
}

export interface DeepSweTaskSnapshot {
  readonly taskId: string;
  readonly taskPath: string;
  readonly instructionSha256: string;
  readonly taskTomlSha256: string;
  readonly baseCommit: string;
  readonly language: string;
  readonly dockerImage: string;
}

export interface DeepSwePreparedArm {
  readonly arm: DeepSweArm;
  readonly jobName: string;
  readonly jobConfigPath: string;
  readonly contextIndexPath: string;
  readonly expectedJobResultPath: string;
  readonly command: readonly string[];
}

export interface DeepSwePreparationManifest {
  readonly schemaVersion: 1;
  readonly benchmark: "deepswe-kontext-ab";
  readonly preparedAt: string;
  readonly deepSweRevision: string;
  readonly pierRevision: string;
  readonly adapterRevision: string;
  readonly model: string;
  readonly reasoningEffort: string;
  readonly attempts: number;
  readonly sampleSeed: number;
  readonly tasks: readonly DeepSweTaskSnapshot[];
  readonly arms: readonly DeepSwePreparedArm[];
  readonly corpusSha256ByTask: Readonly<Record<string, string>>;
}

export type DeepSweExclusionReason =
  | "provider_error"
  | "network_error"
  | "verifier_error"
  | "environment_error"
  | "agent_setup_error"
  | "unclassified_infrastructure_error";

export type DeepSweCapabilityFailureReason =
  | "verifier_rejected"
  | "agent_timeout"
  | "context_exhausted";

export interface DeepSweContextTelemetry {
  readonly prepareCalls: number;
  readonly searchCalls: number;
  readonly beginLogicCalls: number;
  readonly fastCheckCalls: number;
  readonly targetedCheckCalls: number;
  readonly logicSymbols: readonly string[];
  readonly fullyCheckedLogicSymbols: readonly string[];
  readonly protocolComplete: boolean;
}

export interface DeepSweTrialResult {
  readonly arm: DeepSweArm;
  readonly taskId: string;
  readonly trialName: string;
  readonly rolloutIndex: number;
  readonly eligible: boolean;
  readonly success: boolean;
  readonly reward?: number;
  readonly exclusionReason?: DeepSweExclusionReason;
  readonly capabilityFailureReason?: DeepSweCapabilityFailureReason;
  readonly exceptionType?: string;
  readonly startedAt?: string;
  readonly finishedAt?: string;
  readonly durationMilliseconds?: number;
  readonly inputTokens?: number;
  readonly cachedTokens?: number;
  readonly outputTokens?: number;
  readonly costUsd?: number;
  readonly agentSteps?: number;
  readonly peakContextTokens?: number;
  readonly patchSha256?: string;
  readonly trajectoryPath?: string;
  readonly trajectorySha256?: string;
  readonly context: DeepSweContextTelemetry;
}

export interface DeepSweArmSummary {
  readonly arm: DeepSweArm;
  readonly tasks: number;
  readonly eligibleTrials: number;
  readonly excludedTrials: number;
  readonly protocolCompleteTrials: number;
  readonly passAt1: number;
  readonly passAt4: number;
  readonly runToRun95?: readonly [number, number];
  readonly medianOutputTokens?: number;
  readonly medianDurationMilliseconds?: number;
  readonly medianCostUsd?: number;
  readonly medianAgentSteps?: number;
}

export interface DeepSwePairedComparison {
  readonly treatment: "kontext";
  readonly control: "baseline" | "rag";
  readonly comparableTasks: number;
  readonly passAt1Delta: number;
  readonly passAt4Delta: number;
  readonly passAt1ClusterBootstrap95?: readonly [number, number];
  readonly passAt4ClusterBootstrap95?: readonly [number, number];
}

export interface DeepSweReport {
  readonly schemaVersion: 1;
  readonly benchmark: "deepswe-kontext-ab";
  readonly generatedAt: string;
  readonly manifest: DeepSwePreparationManifest;
  readonly summaries: readonly DeepSweArmSummary[];
  readonly comparisons: readonly DeepSwePairedComparison[];
  readonly trials: readonly DeepSweTrialResult[];
  readonly limitations: readonly string[];
}

export interface DeepSwePrepareOptions {
  readonly repositoryRoot: string;
  readonly datasetTasksPath: string;
  readonly corpusRoot: string;
  readonly runDirectory: string;
  readonly jobsDirectory: string;
  readonly pierBinary: string;
  readonly model: string;
  readonly reasoningEffort: string;
  readonly attempts: number;
  readonly concurrency: number;
  readonly sampleSeed: number;
  readonly arms: readonly DeepSweArm[];
  readonly taskIds?: readonly string[];
  readonly taskLimit?: number;
  readonly environment: "docker" | "modal";
  readonly envFile?: string;
  readonly miniSweAgentVersion?: string;
  readonly deepSweRevision?: string;
  readonly pierRevision: string;
  readonly adapterRevision: string;
}
