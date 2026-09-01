import type {
  ChangeBundleInput,
  ContextAssessment,
  InvariantEvaluation,
  ReviewFinding,
  TaskEvidence,
  TaskState,
} from "@kontext-brain/spec";
import { z } from "zod";

const nonEmptyString = z.string().min(1);
const verifierKindSchema = z.enum(["test", "typecheck", "build", "lint", "query", "manual_review"]);
const revisionRefSchema = z
  .object({
    kind: z.enum(["decision", "domain_term", "invariant"]),
    recordId: nonEmptyString,
    revisionId: nonEmptyString,
  })
  .strict();
const changeBundleSchema = z
  .object({
    taskId: nonEmptyString,
    workItemId: nonEmptyString,
    baseRevision: nonEmptyString,
    resultRevision: nonEmptyString,
    taskContextDigest: nonEmptyString,
    patchDigest: nonEmptyString,
    changedSymbolIds: z.array(nonEmptyString),
    changedPaths: z.array(nonEmptyString),
    contextReceiptIds: z.array(nonEmptyString),
    evidenceIds: z.array(nonEmptyString),
    normativeRevisions: z.array(revisionRefSchema),
    verificationRunIds: z.array(nonEmptyString),
    proposals: z.array(nonEmptyString),
    unresolved: z.array(nonEmptyString),
    submittedAt: z.string().datetime(),
  })
  .strict();
const evidenceSchema = z.discriminatedUnion("kind", [
  z
    .object({
      kind: z.literal("commit"),
      ref: nonEmptyString,
      codeRevision: nonEmptyString,
      contextDigest: nonEmptyString,
      observedAt: z.string().datetime(),
    })
    .strict(),
  z
    .object({
      kind: z.literal("approval"),
      role: z.enum(["code_owner", "domain_owner"]),
      ref: nonEmptyString,
      codeRevision: nonEmptyString,
      contextDigest: nonEmptyString,
      observedAt: z.string().datetime(),
    })
    .strict(),
]);
const invariantEvaluationSchema = z
  .object({
    invariantId: nonEmptyString,
    revisionId: nonEmptyString,
    status: z.enum(["guarded", "unguarded", "violated", "inconclusive", "retired"]),
    verificationRunIds: z.array(nonEmptyString),
  })
  .strict();
const reviewFindingSchema = z
  .object({
    findingId: nonEmptyString,
    status: z.enum(["open", "resolved", "dismissed"]),
    codeRevision: nonEmptyString,
    contextDigest: nonEmptyString,
    symbolId: nonEmptyString.optional(),
    ruleRef: nonEmptyString.optional(),
    evidenceIds: z.array(nonEmptyString),
  })
  .strict();

export const checkChangeToolShape = {
  taskId: nonEmptyString,
  workItemId: nonEmptyString,
  workspacePath: nonEmptyString,
  tier: z.enum(["fast", "targeted", "full"]),
  observedAt: z.string().datetime(),
  nextAttemptAt: z.string().datetime(),
};

export const submitChangeBundleToolShape = {
  workspacePath: nonEmptyString,
  bundle: changeBundleSchema,
};

export const proposeTransitionToolShape = {
  taskId: nonEmptyString,
  currentState: z.enum(["planned", "in_progress", "awaiting_evidence", "done", "blocked"]),
  workStarted: z.boolean(),
  completionRequested: z.boolean(),
  context: z
    .object({
      status: z.enum(["current", "stale", "conflict", "inaccessible", "unavailable"]),
      contextDigest: nonEmptyString,
    })
    .strict(),
  evidence: z.array(evidenceSchema),
  invariantEvaluations: z.array(invariantEvaluationSchema),
  reviewFindings: z.array(reviewFindingSchema),
  requestedAt: z.string().datetime(),
};

export interface CheckChangeRequest {
  readonly taskId: string;
  readonly workItemId: string;
  readonly workspacePath: string;
  readonly tier: "fast" | "targeted" | "full";
  readonly observedAt: string;
  readonly nextAttemptAt: string;
}

export interface SubmitChangeBundleRequest {
  readonly workspacePath: string;
  readonly bundle: ChangeBundleInput;
}

export interface ProposeTransitionRequest {
  readonly taskId: string;
  readonly currentState: TaskState;
  readonly workStarted: boolean;
  readonly completionRequested: boolean;
  readonly context: ContextAssessment;
  readonly evidence: readonly TaskEvidence[];
  readonly invariantEvaluations: readonly InvariantEvaluation[];
  readonly reviewFindings: readonly ReviewFinding[];
  readonly requestedAt: string;
}

export interface KontextCompletionOperations {
  checkChange(request: CheckChangeRequest): Promise<unknown>;
  submitChangeBundle(request: SubmitChangeBundleRequest): Promise<unknown>;
  proposeTransition(request: ProposeTransitionRequest): Promise<unknown>;
}

export class KontextCompletionToolRouter {
  constructor(private readonly operations: KontextCompletionOperations) {}

  async checkChange(input: unknown): Promise<unknown> {
    const parsed = z.object(checkChangeToolShape).strict().parse(input);
    return this.operations.checkChange(parsed);
  }

  async submitChangeBundle(input: unknown): Promise<unknown> {
    const parsed = z.object(submitChangeBundleToolShape).strict().parse(input);
    return this.operations.submitChangeBundle(parsed);
  }

  async proposeTransition(input: unknown): Promise<unknown> {
    const parsed = z.object(proposeTransitionToolShape).strict().parse(input);
    return this.operations.proposeTransition(parsed);
  }
}
