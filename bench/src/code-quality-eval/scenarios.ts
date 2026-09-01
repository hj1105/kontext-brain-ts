import { callAssertion, requiredFunction, throwAssertion } from "./assertions.js";
import type { CodeQualityScenario } from "./contracts.js";
import { extendedCodeQualityScenarios } from "./scenarios-extended.js";

export const codeQualityScenarios: readonly CodeQualityScenario[] = [
  retryPolicyScenario(),
  cancellationScenario(),
  serviceCreditScenario(),
  ...extendedCodeQualityScenarios,
];

function retryPolicyScenario(): CodeQualityScenario {
  return {
    scenarioId: "retry-policy",
    taskId: "task:code-quality:retry-policy",
    intent: "Implement the current retry-delay policy without changing its public API.",
    publicPrompt:
      "Implement computeRetryDelay in src/policy.js according to the product's current retry policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function computeRetryDelay(_failureIndex, _baseMs) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { computeRetryDelay } from "../src/policy.js";

test("the initial failure uses the base delay", () => {
  assert.equal(computeRetryDelay(0, 100), 100);
});
`,
    workItemId: "work-item:retry-delay",
    plannedSymbolId: "planned-symbol:compute-retry-delay",
    qualifiedName: "computeRetryDelay",
    capabilityId: "capability:retry-policy",
    canonicalTerms: ["RECOVERY_WINDOW_MS"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:retry-delay-policy",
        revisionId: "revision:retry-delay-policy:1",
        statement:
          "computeRetryDelay uses zero-based failureIndex and returns min(baseMs * 3 ** failureIndex, 4500). Both inputs must be non-negative integers; otherwise throw RangeError.",
        evidenceId: "evidence:product:retry-policy",
        evidenceText:
          "The reliability owner approved a zero-based power-of-three retry curve capped at 4500 milliseconds, with RangeError for invalid integer inputs.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:recovery-window",
        revisionId: "revision:recovery-window:1",
        term: "Recovery Window",
        definition:
          "The maximum retry delay. Public implementation code names its 4500 millisecond constant RECOVERY_WINDOW_MS.",
        avoid: ["max delay"],
        evidenceId: "evidence:domain:recovery-window",
        evidenceText:
          "The domain owner named the retry cap Recovery Window and selected RECOVERY_WINDOW_MS as its code term.",
      },
      {
        kind: "invariant",
        recordId: "invariant:retry-delay-bounded",
        revisionId: "revision:retry-delay-bounded:1",
        statement: "A computed retry delay never exceeds the Recovery Window.",
        evidenceId: "evidence:product:retry-bound",
        evidenceText: "Operations requires all retry delays to stay at or below 4500 milliseconds.",
      },
    ],
    evaluateHidden: async (module) => {
      const compute = requiredFunction(module, "computeRetryDelay");
      return {
        assertions: [
          callAssertion("triples-after-first-failure", compute, [1, 100], 300),
          callAssertion("caps-at-recovery-window", compute, [5, 100], 4500),
          callAssertion("accepts-zero-base", compute, [3, 0], 0),
          throwAssertion("rejects-negative-index", compute, [-1, 100], RangeError),
          throwAssertion("rejects-fractional-base", compute, [1, 2.5], RangeError),
        ],
      };
    },
  };
}

function cancellationScenario(): CodeQualityScenario {
  return {
    scenarioId: "order-cancellation",
    taskId: "task:code-quality:order-cancellation",
    intent: "Implement the current order-cancellation outcome policy.",
    publicPrompt:
      "Implement cancellationOutcome in src/policy.js according to the current order policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function cancellationOutcome(_order) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { cancellationOutcome } from "../src/policy.js";

test("an unshipped confirmed order can be revoked", () => {
  assert.equal(
    cancellationOutcome({ state: "confirmed", shipmentId: null, fraudHold: false }),
    "revocable",
  );
});
`,
    workItemId: "work-item:cancellation-outcome",
    plannedSymbolId: "planned-symbol:cancellation-outcome",
    qualifiedName: "cancellationOutcome",
    capabilityId: "capability:order-cancellation",
    canonicalTerms: ["hasRevocationEligibility"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:order-revocation-policy",
        revisionId: "revision:order-revocation-policy:1",
        statement:
          "cancellationOutcome returns 'revocable' only for state 'confirmed', no shipmentId, and fraudHold not true. Payment capture does not affect this decision. Every other order returns 'locked'.",
        evidenceId: "evidence:product:order-revocation",
        evidenceText:
          "Order operations approved revocation for confirmed, unshipped, non-fraud-held orders even after payment capture.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:revocation-eligibility",
        revisionId: "revision:revocation-eligibility:1",
        term: "Revocation Eligibility",
        definition:
          "Whether an order may be cancelled under the approved policy. The implementation names the predicate hasRevocationEligibility.",
        avoid: ["cancelable flag"],
        evidenceId: "evidence:domain:revocation-eligibility",
        evidenceText:
          "The domain owner standardized Revocation Eligibility and the code predicate hasRevocationEligibility.",
      },
      {
        kind: "invariant",
        recordId: "invariant:shipped-order-locked",
        revisionId: "revision:shipped-order-locked:1",
        statement: "An order with a shipmentId is always locked against cancellation.",
        evidenceId: "evidence:product:shipment-lock",
        evidenceText:
          "Fulfillment requires every shipment-created order to remain cancellation-locked.",
      },
    ],
    evaluateHidden: async (module) => {
      const outcome = requiredFunction(module, "cancellationOutcome");
      return {
        assertions: [
          callAssertion(
            "captured-payment-remains-revocable",
            outcome,
            [{ state: "confirmed", shipmentId: null, fraudHold: false, paymentCaptured: true }],
            "revocable",
          ),
          callAssertion(
            "fraud-hold-locks",
            outcome,
            [{ state: "confirmed", shipmentId: null, fraudHold: true }],
            "locked",
          ),
          callAssertion(
            "shipment-locks",
            outcome,
            [{ state: "confirmed", shipmentId: "shipment:1", fraudHold: false }],
            "locked",
          ),
          callAssertion(
            "draft-locks",
            outcome,
            [{ state: "draft", shipmentId: null, fraudHold: false }],
            "locked",
          ),
        ],
      };
    },
  };
}

function serviceCreditScenario(): CodeQualityScenario {
  return {
    scenarioId: "service-credit-allocation",
    taskId: "task:code-quality:service-credit-allocation",
    intent: "Allocate an integer service credit according to the approved account policy.",
    publicPrompt:
      "Implement allocateServiceCredit in src/policy.js according to the current service-credit policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function allocateServiceCredit(_totalCents, _accountIds) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { allocateServiceCredit } from "../src/policy.js";

test("an evenly divisible credit is allocated equally", () => {
  assert.deepEqual(allocateServiceCredit(6, ["account:b", "account:a", "account:c"]), {
    "account:a": 2,
    "account:b": 2,
    "account:c": 2,
  });
});
`,
    workItemId: "work-item:service-credit-allocation",
    plannedSymbolId: "planned-symbol:allocate-service-credit",
    qualifiedName: "allocateServiceCredit",
    capabilityId: "capability:service-credit",
    canonicalTerms: ["serviceCreditByAccount"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:service-credit-allocation",
        revisionId: "revision:service-credit-allocation:1",
        statement:
          "allocateServiceCredit validates a non-negative integer totalCents and unique non-empty accountIds, sorts IDs lexicographically, divides equally, and assigns remainder cents from the lexicographically greatest ID backward. Return an object in ascending account ID insertion order.",
        evidenceId: "evidence:finance:service-credit",
        evidenceText:
          "Finance approved deterministic equal allocation with remainder assigned from greatest account ID backward while output order remains ascending.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:service-credit",
        revisionId: "revision:service-credit:1",
        term: "Service Credit",
        definition:
          "An integer customer credit allocation. The result accumulator is named serviceCreditByAccount.",
        avoid: ["refund split"],
        evidenceId: "evidence:domain:service-credit",
        evidenceText:
          "The billing domain owner standardized Service Credit and serviceCreditByAccount.",
      },
      {
        kind: "invariant",
        recordId: "invariant:service-credit-conservation",
        revisionId: "revision:service-credit-conservation:1",
        statement: "Allocated cents sum exactly to totalCents and no allocation is negative.",
        evidenceId: "evidence:finance:credit-conservation",
        evidenceText: "The ledger requires exact conservation of every service-credit cent.",
      },
    ],
    evaluateHidden: async (module) => {
      const allocate = requiredFunction(module, "allocateServiceCredit");
      return {
        assertions: [
          callAssertion(
            "remainder-starts-at-greatest-id",
            allocate,
            [8, ["account:b", "account:a", "account:c"]],
            { "account:a": 2, "account:b": 3, "account:c": 3 },
          ),
          callAssertion(
            "single-remainder-goes-to-greatest-id",
            allocate,
            [7, ["account:b", "account:a", "account:c"]],
            { "account:a": 2, "account:b": 2, "account:c": 3 },
          ),
          callAssertion(
            "zero-credit-preserves-sorted-accounts",
            allocate,
            [0, ["account:z", "account:a"]],
            { "account:a": 0, "account:z": 0 },
          ),
          throwAssertion(
            "rejects-duplicate-accounts",
            allocate,
            [5, ["account:a", "account:a"]],
            Error,
          ),
          throwAssertion("rejects-negative-total", allocate, [-1, ["account:a"]], RangeError),
        ],
      };
    },
  };
}
