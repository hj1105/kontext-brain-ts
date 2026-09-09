import path from "node:path";
import {
  type VerifierAdapter,
  type VerifierAdapterResult,
  type VerifierExecutionRequest,
  VerifierInfrastructureError,
} from "@kontext-brain/orchestrator";
import type { VerifierRef } from "@kontext-brain/spec";
import type { SidecarChangeEvidence } from "./sidecar-change-evidence.js";

/**
 * The fast tier always demands four `kontext:*` query verifiers, but nothing
 * evaluated them: they fell through to the workspace command adapter, which has
 * no definition for them, so every Change Bundle was rejected for proof the
 * product could not produce. The sidecar already holds the evidence these
 * checks are about — it observes the patch, resynchronizes Code Symbols,
 * resolves Planned Symbol identities and binds the Context Receipt — so the
 * check that gathered that evidence primes this adapter and the plan reads it.
 */

export const KONTEXT_EVIDENCE_VERIFIERS: readonly VerifierRef[] = [
  { kind: "query", ref: "kontext:semantic-sync" },
  { kind: "query", ref: "kontext:stable-symbol-identity" },
  { kind: "query", ref: "kontext:domain-term-check" },
  { kind: "query", ref: "kontext:graph-query-check" },
];

export interface PrimedEvidenceBinding {
  readonly workspacePath: string;
  readonly codeRevision: string;
}

export class EvidenceBackedQueryVerifierAdapter implements VerifierAdapter {
  private readonly primed = new Map<string, SidecarChangeEvidence>();

  /** Called by the observation that produced the evidence, before its plan runs. */
  prime(binding: PrimedEvidenceBinding, evidence: SidecarChangeEvidence): void {
    this.primed.set(this.key(binding), evidence);
  }

  async execute(request: VerifierExecutionRequest): Promise<VerifierAdapterResult> {
    const evidence = this.primed.get(this.key(request));
    if (!evidence) {
      // Why: a retry after the sidecar restarted has no observation to judge; the
      // coordinator records that as inconclusive and keeps retrying, as it did before.
      throw new VerifierInfrastructureError(
        `No sidecar change evidence is primed for ${request.codeRevision} in this workspace`,
      );
    }
    switch (request.requirement.verifier.ref) {
      case "kontext:semantic-sync":
        // Every changed behavior-bearing symbol resynchronized to one the receipt authorizes.
        return {
          result: evidence.unauthorizedChangedSymbolIds.length === 0 ? "passed" : "failed",
          output: {
            changedSymbolIds: evidence.observedPatch.changedSymbolIds,
            unauthorizedChangedSymbolIds: evidence.unauthorizedChangedSymbolIds,
          },
        };
      case "kontext:stable-symbol-identity":
        // Each Planned Symbol still resolves to exactly one current Code Symbol.
        return {
          result: evidence.plannedSymbolIssues.length === 0 ? "passed" : "failed",
          output: {
            bindings: evidence.plannedSymbolBindings,
            issues: evidence.plannedSymbolIssues,
          },
        };
      case "kontext:domain-term-check":
      case "kontext:graph-query-check": {
        // The normative and graph context the worker was handed is the context this
        // verification runs under; a refreshed snapshot invalidates the receipt.
        const receipts = evidence.receipts.map((receipt) => ({
          receiptId: receipt.receiptId,
          contextDigest: receipt.contextDigest,
          normativeRevisions: receipt.normativeRevisions,
        }));
        const current = receipts.every(
          (receipt) => receipt.contextDigest === request.contextDigest,
        );
        return {
          result: receipts.length > 0 && current ? "passed" : "failed",
          output: { expectedContextDigest: request.contextDigest, receipts },
        };
      }
      default:
        throw new VerifierInfrastructureError(
          `Kontext evidence verifier does not know ${request.requirement.verifier.ref}`,
        );
    }
  }

  private key(binding: PrimedEvidenceBinding): string {
    return `${path.resolve(binding.workspacePath)}\n${binding.codeRevision}`;
  }
}
