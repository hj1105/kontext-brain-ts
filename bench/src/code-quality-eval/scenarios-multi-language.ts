import { callAssertion, requiredFunction, throwAssertion } from "./assertions.js";
import type { CodeQualityScenario } from "./contracts.js";

/**
 * The JavaScript scenarios all target one exported function in one file, which
 * leaves the provider's type-aware identity and its non-function symbol kinds
 * untested. These add TypeScript scenarios that carry real annotations and a
 * class method, plus Python scenarios that exercise the syntactic provider.
 */
export const multiLanguageScenarios: readonly CodeQualityScenario[] = [
  ledgerPostingScenario(),
  membershipTierScenario(),
  pythonPricingScenario(),
  pythonQuorumScenario(),
];

/** TypeScript, exercising an annotated signature and a discriminated result. */
function ledgerPostingScenario(): CodeQualityScenario {
  return {
    scenarioId: "ledger-posting",
    taskId: "task:code-quality:ledger-posting",
    intent: "Classify a ledger posting under the approved settlement policy.",
    publicPrompt:
      "Implement classifyPosting in src/policy.ts according to the current settlement policy. Keep the exported types and the public API stable, and run the tests.",
    sourceFile: "src/policy.ts",
    initialSource: `export type PostingOutcome = "settled" | "pending" | "rejected";

export interface Posting {
  readonly amountCents: number;
  readonly clearedAt: string | null;
  readonly disputed: boolean;
}

export function classifyPosting(_posting: Posting): PostingOutcome {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { classifyPosting } from "../src/policy.ts";

test("a cleared undisputed posting settles", () => {
  assert.equal(
    classifyPosting({ amountCents: 100, clearedAt: "2026-03-01T00:00:00.000Z", disputed: false }),
    "settled",
  );
});
`,
    workItemId: "work-item:ledger-posting",
    plannedSymbolId: "planned-symbol:classify-posting",
    qualifiedName: "classifyPosting",
    capabilityId: "capability:ledger-posting",
    canonicalTerms: ["DISPUTE_HOLD_CENTS"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:settlement-policy",
        revisionId: "revision:settlement-policy:1",
        statement:
          "classifyPosting returns 'rejected' only when amountCents is zero or negative. A disputed posting returns 'pending' regardless of clearedAt, except that a disputed posting at or below the 2500 cent dispute hold still settles when clearedAt is present. An undisputed posting returns 'settled' when clearedAt is present and 'pending' otherwise.",
        evidenceId: "evidence:finance:settlement-policy",
        evidenceText:
          "Treasury approved auto-settling small disputed postings at or below the dispute hold because manual review costs more than the exposure.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:dispute-hold",
        revisionId: "revision:dispute-hold:1",
        term: "Dispute Hold",
        definition:
          "The amount at or below which a disputed posting still settles. Implementation code names this constant DISPUTE_HOLD_CENTS.",
        avoid: ["small dispute limit"],
        evidenceId: "evidence:domain:dispute-hold",
        evidenceText: "The finance domain owner standardized Dispute Hold and DISPUTE_HOLD_CENTS.",
      },
      {
        kind: "invariant",
        recordId: "invariant:non-positive-rejected",
        revisionId: "revision:non-positive-rejected:1",
        statement: "A posting of zero or less is always rejected.",
        evidenceId: "evidence:finance:non-positive",
        evidenceText: "The ledger refuses to settle a non-positive posting under any condition.",
      },
    ],
    evaluateHidden: async (module) => {
      const classify = requiredFunction(module, "classifyPosting");
      const cleared = "2026-03-01T00:00:00.000Z";
      return {
        assertions: [
          callAssertion(
            "small-dispute-still-settles",
            classify,
            [{ amountCents: 2500, clearedAt: cleared, disputed: true }],
            "settled",
          ),
          callAssertion(
            "large-dispute-pends",
            classify,
            [{ amountCents: 2501, clearedAt: cleared, disputed: true }],
            "pending",
          ),
          callAssertion(
            "uncleared-dispute-pends",
            classify,
            [{ amountCents: 100, clearedAt: null, disputed: true }],
            "pending",
          ),
          callAssertion(
            "zero-is-rejected-before-any-other-rule",
            classify,
            [{ amountCents: 0, clearedAt: cleared, disputed: false }],
            "rejected",
          ),
          callAssertion(
            "uncleared-undisputed-pends",
            classify,
            [{ amountCents: 100, clearedAt: null, disputed: false }],
            "pending",
          ),
        ],
      };
    },
  };
}

/** TypeScript, where the Planned Symbol is a class method rather than a function. */
function membershipTierScenario(): CodeQualityScenario {
  return {
    scenarioId: "membership-tier",
    taskId: "task:code-quality:membership-tier",
    intent: "Resolve a membership tier under the approved retention policy.",
    publicPrompt:
      "Implement the resolve method of MembershipPolicy in src/policy.ts according to the current retention policy. Keep the class and its public API stable, and run the tests.",
    sourceFile: "src/policy.ts",
    initialSource: `export type MembershipTier = "standard" | "silver" | "gold";

export class MembershipPolicy {
  resolve(_monthsActive: number, _lifetimeSpendCents: number): MembershipTier {
    throw new Error("Not implemented");
  }
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { MembershipPolicy } from "../src/policy.ts";

test("a new low-spend member is standard", () => {
  assert.equal(new MembershipPolicy().resolve(1, 0), "standard");
});
`,
    workItemId: "work-item:membership-tier",
    plannedSymbolId: "planned-symbol:membership-policy-resolve",
    qualifiedName: "MembershipPolicy.resolve",
    capabilityId: "capability:membership-tier",
    canonicalTerms: ["TENURE_CREDIT_CENTS"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:membership-retention",
        revisionId: "revision:membership-retention:1",
        statement:
          "MembershipPolicy.resolve adds a tenure credit of 1000 cents for every complete 12 months active to lifetimeSpendCents, then compares the credited total: 50000 or more is 'gold', 20000 or more is 'silver', otherwise 'standard'. Tenure alone never promotes a member without spend, because a credited total of zero stays standard. Negative inputs throw RangeError.",
        evidenceId: "evidence:retention:membership",
        evidenceText:
          "The retention owner approved crediting tenure into the spend comparison rather than adding a separate tenure rule, so a long-tenured low-spend member is not promoted on tenure alone.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:tenure-credit",
        revisionId: "revision:tenure-credit:1",
        term: "Tenure Credit",
        definition:
          "The per-year spend credit granted for tenure. Implementation code names this constant TENURE_CREDIT_CENTS.",
        avoid: ["loyalty bonus"],
        evidenceId: "evidence:domain:tenure-credit",
        evidenceText:
          "The retention domain owner standardized Tenure Credit and TENURE_CREDIT_CENTS.",
      },
      {
        kind: "invariant",
        recordId: "invariant:tier-monotonic",
        revisionId: "revision:tier-monotonic:1",
        statement: "More spend at equal tenure never lowers a member's tier.",
        evidenceId: "evidence:retention:monotonic",
        evidenceText: "Retention requires tier assignment to be non-decreasing in spend.",
      },
    ],
    evaluateHidden: async (module) => {
      const PolicyClass = module.MembershipPolicy as new () => {
        resolve(months: number, spend: number): string;
      };
      if (typeof PolicyClass !== "function") throw new Error("Missing MembershipPolicy export");
      const resolve = (months: number, spend: number): unknown =>
        new PolicyClass().resolve(months, spend);
      const call = (...args: readonly unknown[]) => resolve(args[0] as number, args[1] as number);
      return {
        assertions: [
          // 24 months credits 2000, so 48000 becomes 50000 and reaches gold.
          callAssertion("tenure-credit-reaches-gold", call, [24, 48_000], "gold"),
          callAssertion("just-below-gold-stays-silver", call, [24, 47_999], "silver"),
          // Partial years grant nothing: 23 months credits 1000, not 1916.
          callAssertion("partial-year-grants-nothing-extra", call, [23, 48_000], "silver"),
          callAssertion("tenure-alone-stays-standard", call, [120, 0], "standard"),
          throwAssertion("rejects-negative-spend", call, [12, -1], RangeError),
        ],
      };
    },
  };
}

/** Python, exercising the syntactic provider on a module-level function. */
function pythonPricingScenario(): CodeQualityScenario {
  return {
    scenarioId: "python-volume-pricing",
    taskId: "task:code-quality:python-volume-pricing",
    intent: "Price a volume order under the approved tier policy.",
    publicPrompt:
      "Implement price_order in src/policy.py according to the current volume pricing policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.py",
    initialSource: `def price_order(units, unit_price_cents):
    raise NotImplementedError
`,
    publicTestSource: `import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from policy import price_order


class PublicPricing(unittest.TestCase):
    def test_single_unit_uses_the_list_price(self):
        self.assertEqual(price_order(1, 1000), 1000)


if __name__ == "__main__":
    unittest.main()
`,
    workItemId: "work-item:python-volume-pricing",
    plannedSymbolId: "planned-symbol:price-order",
    qualifiedName: "price_order",
    capabilityId: "capability:python-volume-pricing",
    canonicalTerms: ["MARGINAL_TIER_BREAKS"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:volume-pricing",
        revisionId: "revision:volume-pricing:1",
        statement:
          "price_order applies marginal tier pricing, not a single rate chosen by total volume. The first 10 units bill at full unit_price_cents, units 11 through 100 bill at 90 percent, and units beyond 100 bill at 80 percent. Each tier's subtotal is floored to whole cents independently before summing. units and unit_price_cents must be non-negative integers; otherwise raise ValueError.",
        evidenceId: "evidence:pricing:volume-tiers",
        evidenceText:
          "The pricing owner approved marginal tiers so that crossing a break never lowers the total bill, which a whole-order discount would allow.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:marginal-tier-break",
        revisionId: "revision:marginal-tier-break:1",
        term: "Marginal Tier Break",
        definition:
          "A unit count at which the marginal rate changes. Implementation code names this table MARGINAL_TIER_BREAKS.",
        avoid: ["discount table"],
        evidenceId: "evidence:domain:marginal-tier-break",
        evidenceText:
          "The pricing domain owner standardized Marginal Tier Break and MARGINAL_TIER_BREAKS.",
      },
      {
        kind: "invariant",
        recordId: "invariant:pricing-monotonic",
        revisionId: "revision:pricing-monotonic:1",
        statement: "Ordering one more unit never lowers the order total.",
        evidenceId: "evidence:pricing:monotonic",
        evidenceText: "Finance requires order totals to be non-decreasing in unit count.",
      },
    ],
    hiddenChecks: [
      // 10 at 1000 = 10000; the 11th unit bills at 900, so a whole-order
      // discount would give 9900 instead of 10900.
      {
        assertionId: "eleventh-unit-is-marginal",
        functionName: "price_order",
        args: [11, 1000],
        expected: 10_900,
      },
      {
        assertionId: "tier-break-is-inclusive",
        functionName: "price_order",
        args: [10, 1000],
        expected: 10_000,
      },
      {
        assertionId: "third-tier-applies-beyond-one-hundred",
        functionName: "price_order",
        args: [101, 1000],
        expected: 10_000 + 90 * 900 + 800,
      },
      {
        assertionId: "zero-units-cost-nothing",
        functionName: "price_order",
        args: [0, 1000],
        expected: 0,
      },
      {
        assertionId: "floors-each-tier-independently",
        functionName: "price_order",
        args: [11, 5],
        expected: 50 + 4,
      },
      {
        assertionId: "rejects-negative-units",
        functionName: "price_order",
        args: [-1, 1000],
        throws: "ValueError",
      },
    ],
    evaluateHidden: async () => {
      throw new Error("python-volume-pricing is evaluated through its Python driver");
    },
  };
}

/** Python, where the Planned Symbol is a method on a class. */
function pythonQuorumScenario(): CodeQualityScenario {
  return {
    scenarioId: "python-approval-quorum",
    taskId: "task:code-quality:python-approval-quorum",
    intent: "Decide an approval quorum under the approved governance policy.",
    publicPrompt:
      "Implement quorum_met in src/policy.py according to the current approval policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.py",
    initialSource: `def quorum_met(approvals, total_reviewers, risk):
    raise NotImplementedError
`,
    publicTestSource: `import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from policy import quorum_met


class PublicQuorum(unittest.TestCase):
    def test_unanimous_low_risk_meets_quorum(self):
        self.assertTrue(quorum_met(4, 4, "low"))


if __name__ == "__main__":
    unittest.main()
`,
    workItemId: "work-item:python-approval-quorum",
    plannedSymbolId: "planned-symbol:quorum-met",
    qualifiedName: "quorum_met",
    capabilityId: "capability:python-approval-quorum",
    canonicalTerms: ["QUORUM_FLOOR"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:approval-quorum",
        revisionId: "revision:approval-quorum:1",
        statement:
          "quorum_met requires a strict majority for low risk and at least two thirds for high risk, both rounded up, and additionally never fewer than the quorum floor of 3 approvals for high risk. Low risk has no floor. A reviewer pool smaller than the required count can never meet quorum. approvals must not exceed total_reviewers and neither may be negative; otherwise raise ValueError.",
        evidenceId: "evidence:governance:approval-quorum",
        evidenceText:
          "The governance owner approved an absolute floor of three approvals for high risk so that a two-person pool cannot approve a high-risk change by unanimity alone.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:quorum-floor",
        revisionId: "revision:quorum-floor:1",
        term: "Quorum Floor",
        definition:
          "The absolute minimum approvals for high risk regardless of pool size. Implementation code names this constant QUORUM_FLOOR.",
        avoid: ["minimum approvals"],
        evidenceId: "evidence:domain:quorum-floor",
        evidenceText: "The governance domain owner standardized Quorum Floor and QUORUM_FLOOR.",
      },
      {
        kind: "invariant",
        recordId: "invariant:quorum-within-pool",
        revisionId: "revision:quorum-within-pool:1",
        statement: "Approvals never exceed the reviewer pool.",
        evidenceId: "evidence:governance:pool-bound",
        evidenceText: "The approval record cannot hold more approvals than reviewers.",
      },
    ],
    hiddenChecks: [
      // Two of two is unanimous but below the floor of three.
      {
        assertionId: "high-risk-floor-blocks-small-pool",
        functionName: "quorum_met",
        args: [2, 2, "high"],
        expected: false,
      },
      {
        assertionId: "high-risk-two-thirds-rounds-up",
        functionName: "quorum_met",
        args: [3, 4, "high"],
        expected: true,
      },
      {
        assertionId: "high-risk-below-two-thirds-fails",
        functionName: "quorum_met",
        args: [2, 4, "high"],
        expected: false,
      },
      {
        assertionId: "low-risk-has-no-floor",
        functionName: "quorum_met",
        args: [2, 2, "low"],
        expected: true,
      },
      {
        assertionId: "low-risk-needs-strict-majority",
        functionName: "quorum_met",
        args: [2, 4, "low"],
        expected: false,
      },
      {
        assertionId: "rejects-approvals-above-pool",
        functionName: "quorum_met",
        args: [5, 4, "low"],
        throws: "ValueError",
      },
    ],
    evaluateHidden: async () => {
      throw new Error("python-approval-quorum is evaluated through its Python driver");
    },
  };
}
