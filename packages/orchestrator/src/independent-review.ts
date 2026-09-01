import { createHash } from "node:crypto";
import type {
  ReviewFinding,
  TaskContextSnapshot,
  TaskContract,
  VerificationRun,
} from "@kontext-brain/spec";
import type { AgentRuntimePort, RuntimeProvider, RuntimeSession } from "./runtime.js";
import { createVerificationRun } from "./verifier-registry.js";

export interface IndependentReviewRequest {
  readonly contract: TaskContract;
  readonly snapshot: TaskContextSnapshot;
  readonly workspacePath: string;
  readonly codeRevision: string;
  readonly authorProviders: readonly RuntimeProvider[];
  readonly eligibleProviders: readonly RuntimeProvider[];
  readonly changedSymbolIds: readonly string[];
  readonly changedPaths: readonly string[];
  readonly allowedRuleRefs: readonly string[];
  readonly allowedEvidenceIds: readonly string[];
  readonly reviewPacket: string;
  readonly reviewedAt: string;
  readonly signal?: AbortSignal;
}

export interface IndependentReviewResult {
  readonly reviewerProvider?: RuntimeProvider;
  readonly session?: RuntimeSession;
  readonly findings: readonly ReviewFinding[];
  readonly verificationRun: VerificationRun;
  readonly diagnostic?: string;
}

export class IndependentReviewCoordinator {
  constructor(private readonly runtimes: readonly AgentRuntimePort[]) {}

  async review(request: IndependentReviewRequest): Promise<IndependentReviewResult> {
    if (request.contract.risk === "low") {
      throw new Error("Low-risk completion does not require a runtime Review Finding pass");
    }
    const runtime = await this.selectReviewer(
      request.authorProviders,
      request.eligibleProviders,
    );
    if (!runtime) {
      return this.inconclusive(request, "No eligible non-author subscription runtime can review");
    }
    const session = await runtime.start({
      taskId: request.contract.taskId,
      workItem: {
        workItemId: `review:${request.contract.taskId}`,
        taskId: request.contract.taskId,
        plannedSymbolIds: request.changedSymbolIds,
        dependsOn: [],
        allowedPaths: request.changedPaths,
        requiredVerifiers: [],
        capabilityId: `capability:review:${request.contract.taskId}`,
      },
      workspacePath: request.workspacePath,
      prompt: reviewPrompt(request),
      codeRevision: request.codeRevision,
      contextDigest: request.snapshot.contextDigest,
      executionRole: "independent_review",
      signal: request.signal,
    });
    if (session.status !== "completed" || !session.output) {
      return this.inconclusive(
        request,
        session.diagnostic ?? "Independent reviewer did not return a completed result",
        runtime.provider,
        session,
      );
    }
    try {
      const response = parseReviewResponse(session.output);
      const findings = materializeFindings(request, runtime.provider, response);
      const remainingOpen = findings.filter((finding) => finding.status === "open");
      const result =
        response.verdict === "inconclusive"
          ? "inconclusive"
          : response.verdict === "passed" && remainingOpen.length === 0
            ? "passed"
            : "failed";
      return {
        reviewerProvider: runtime.provider,
        session,
        findings,
        verificationRun: reviewRun(request, result, {
          reviewerProvider: runtime.provider,
          findingIds: findings.map((finding) => finding.findingId),
          verdict: response.verdict,
        }),
      };
    } catch (error) {
      return this.inconclusive(
        request,
        `Invalid independent review output: ${error instanceof Error ? error.message : String(error)}`,
        runtime.provider,
        session,
      );
    }
  }

  private async selectReviewer(
    authorProviders: readonly RuntimeProvider[],
    eligibleProviders: readonly RuntimeProvider[],
  ): Promise<AgentRuntimePort | undefined> {
    const authors = new Set(authorProviders);
    const eligible = new Set(eligibleProviders);
    const candidates = this.runtimes
      .filter((runtime) => eligible.has(runtime.provider) && !authors.has(runtime.provider))
      .sort((left, right) => left.provider.localeCompare(right.provider));
    for (const runtime of candidates) {
      const capability = await runtime.inspectCapabilities();
      if (
        capability.installed &&
        capability.authenticated &&
        capability.billingPath === "subscription" &&
        capability.supports.structuredOutput &&
        capability.supports.workspaceSandbox
      ) {
        return runtime;
      }
    }
    return undefined;
  }

  private inconclusive(
    request: IndependentReviewRequest,
    diagnostic: string,
    reviewerProvider?: RuntimeProvider,
    session?: RuntimeSession,
  ): IndependentReviewResult {
    return {
      reviewerProvider,
      session,
      findings: [],
      verificationRun: reviewRun(request, "inconclusive", { diagnostic, reviewerProvider }),
      diagnostic,
    };
  }
}

interface ReviewResponse {
  readonly verdict: "passed" | "failed" | "inconclusive";
  readonly findings: readonly {
    readonly message: string;
    readonly symbolId?: string;
    readonly ruleRef?: string;
    readonly evidenceIds: readonly string[];
  }[];
}

function parseReviewResponse(value: string): ReviewResponse {
  const parsed: unknown = JSON.parse(value);
  if (!isRecord(parsed) || !["passed", "failed", "inconclusive"].includes(String(parsed.verdict))) {
    throw new Error("verdict must be passed, failed, or inconclusive");
  }
  if (!Array.isArray(parsed.findings)) {
    throw new Error("findings must be an array");
  }
  const findings = parsed.findings.map((finding) => {
    if (!isRecord(finding) || !nonEmpty(finding.message) || !stringArray(finding.evidenceIds)) {
      throw new Error("each finding requires message and evidenceIds");
    }
    if (finding.symbolId !== undefined && !nonEmpty(finding.symbolId)) {
      throw new Error("finding symbolId must be a non-empty string");
    }
    if (finding.ruleRef !== undefined && !nonEmpty(finding.ruleRef)) {
      throw new Error("finding ruleRef must be a non-empty string");
    }
    return {
      message: finding.message,
      symbolId: finding.symbolId as string | undefined,
      ruleRef: finding.ruleRef as string | undefined,
      evidenceIds: finding.evidenceIds,
    };
  });
  return {
    verdict: parsed.verdict as ReviewResponse["verdict"],
    findings,
  };
}

function materializeFindings(
  request: IndependentReviewRequest,
  reviewerProvider: RuntimeProvider,
  response: ReviewResponse,
): readonly ReviewFinding[] {
  const changedSymbols = new Set(request.changedSymbolIds);
  const allowedRules = new Set(request.allowedRuleRefs);
  const allowedEvidence = new Set(request.allowedEvidenceIds);
  const findings: ReviewFinding[] = [];
  for (const proposed of response.findings) {
    if (proposed.symbolId && !changedSymbols.has(proposed.symbolId)) {
      throw new Error(`finding references unchanged Code Symbol ${proposed.symbolId}`);
    }
    if (proposed.ruleRef && !allowedRules.has(proposed.ruleRef)) {
      throw new Error(`finding references unavailable rule ${proposed.ruleRef}`);
    }
    if (!proposed.evidenceIds.every((evidenceId) => allowedEvidence.has(evidenceId))) {
      throw new Error("finding references unavailable Evidence");
    }
    const identity = {
      taskId: request.contract.taskId,
      codeRevision: request.codeRevision,
      contextDigest: request.snapshot.contextDigest,
      reviewerProvider,
      message: proposed.message,
      symbolId: proposed.symbolId,
      ruleRef: proposed.ruleRef,
      evidenceIds: [...proposed.evidenceIds].sort(),
      reviewedAt: request.reviewedAt,
    };
    findings.push({
      findingId: `review-finding:${createHash("sha256")
        .update(JSON.stringify(identity))
        .digest("hex")}`,
      status: "open",
      codeRevision: request.codeRevision,
      contextDigest: request.snapshot.contextDigest,
      message: proposed.message,
      reviewerProvider,
      authorProviders: uniqueProviders(request.authorProviders),
      reviewedAt: request.reviewedAt,
      symbolId: proposed.symbolId,
      ruleRef: proposed.ruleRef,
      evidenceIds: [...proposed.evidenceIds].sort(),
    });
  }
  return findings.sort((left, right) => left.findingId.localeCompare(right.findingId));
}

function reviewPrompt(request: IndependentReviewRequest): string {
  return [
    request.reviewPacket,
    "",
    "Return exactly this JSON shape:",
    '{"verdict":"passed|failed|inconclusive","findings":[{"message":"...","symbolId":"optional","ruleRef":"optional","evidenceIds":["..."]}]}',
    "Use findings only for concrete blocking concerns. A passed verdict requires no open finding.",
  ].join("\n");
}

function reviewRun(
  request: IndependentReviewRequest,
  result: "passed" | "failed" | "inconclusive",
  output: unknown,
): VerificationRun {
  return createVerificationRun(
    {
      tier: "full",
      verifier: { kind: "manual_review", ref: "kontext:independent-review" },
      subjectIds: [request.contract.taskId, ...request.contract.targets],
    },
    {
      codeRevision: request.codeRevision,
      contextDigest: request.snapshot.contextDigest,
      observedAt: request.reviewedAt,
    },
    result,
    output,
  );
}

function uniqueProviders(values: readonly RuntimeProvider[]): RuntimeProvider[] {
  return Array.from(new Set(values)).sort((left, right) => left.localeCompare(right));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmpty(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function stringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every(nonEmpty);
}
