import { callAssertion, requiredFunction, throwAssertion } from "./assertions.js";
import type { CodeQualityScenario } from "./contracts.js";

/**
 * Every policy here is deliberately not the choice a competent engineer reaches
 * by default, because a scenario whose natural implementation already matches
 * the policy hands the baseline arm free credit. Each public test is satisfied
 * by that natural implementation, so the public surface stays underdetermined
 * and only the private sidecar distinguishes the two arms.
 */
export const extendedCodeQualityScenarios: readonly CodeQualityScenario[] = [
  invoiceTaxScenario(),
  accountLockoutScenario(),
  shippingQuoteScenario(),
  prorationScenario(),
  inventoryReservationScenario(),
  moderationEscalationScenario(),
  rateQuotaScenario(),
];

/** Half-to-even at line level, not half-up on the invoice total. */
function invoiceTaxScenario(): CodeQualityScenario {
  return {
    scenarioId: "invoice-tax-rounding",
    taskId: "task:code-quality:invoice-tax-rounding",
    intent: "Compute invoice tax under the approved rounding policy.",
    publicPrompt:
      "Implement computeInvoiceTaxCents in src/policy.js according to the current invoice tax policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function computeInvoiceTaxCents(_lineAmountsCents, _taxRateBasisPoints) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { computeInvoiceTaxCents } from "../src/policy.js";

test("a single line with an exact tax amount needs no rounding", () => {
  assert.equal(computeInvoiceTaxCents([1000], 1000), 100);
});
`,
    workItemId: "work-item:invoice-tax",
    plannedSymbolId: "planned-symbol:compute-invoice-tax-cents",
    qualifiedName: "computeInvoiceTaxCents",
    capabilityId: "capability:invoice-tax",
    canonicalTerms: ["roundedLineTaxCents"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:invoice-tax-rounding",
        revisionId: "revision:invoice-tax-rounding:1",
        statement:
          "computeInvoiceTaxCents rounds each line independently, never the invoice total. A line's tax is lineAmountCents * taxRateBasisPoints / 10000 rounded half to even, and the invoice tax is the sum of those rounded line values. Every line amount must be a non-negative integer and taxRateBasisPoints must be an integer from 0 to 10000; otherwise throw RangeError.",
        evidenceId: "evidence:finance:invoice-tax-rounding",
        evidenceText:
          "The tax controller approved banker's rounding per invoice line, matching the filing system, and rejected rounding the invoice total because it drifts from the filed per-line amounts.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:rounded-line-tax",
        revisionId: "revision:rounded-line-tax:1",
        term: "Rounded Line Tax",
        definition:
          "The half-to-even rounded tax of one invoice line. Implementation code names this value roundedLineTaxCents.",
        avoid: ["tax total"],
        evidenceId: "evidence:domain:rounded-line-tax",
        evidenceText:
          "The billing domain owner standardized Rounded Line Tax and the code term roundedLineTaxCents.",
      },
      {
        kind: "invariant",
        recordId: "invariant:invoice-tax-line-sum",
        revisionId: "revision:invoice-tax-line-sum:1",
        statement: "Invoice tax always equals the sum of its Rounded Line Tax values.",
        evidenceId: "evidence:finance:line-sum",
        evidenceText:
          "Audit requires the filed invoice tax to reconcile exactly against its per-line amounts.",
      },
    ],
    evaluateHidden: async (module) => {
      const compute = requiredFunction(module, "computeInvoiceTaxCents");
      return {
        assertions: [
          // 2.5 rounds to 2 under half-to-even and to 3 under half-up.
          callAssertion("half-to-even-rounds-down-to-even", compute, [[50], 500], 2),
          // 4.5 rounds to 4 under half-to-even and to 5 under half-up.
          callAssertion("half-to-even-rounds-down-from-odd-half", compute, [[90], 500], 4),
          // Per line gives 2 + 2; rounding the 5.0 total would give 5.
          callAssertion("rounds-each-line-not-the-total", compute, [[50, 50], 500], 4),
          callAssertion("zero-rate-produces-no-tax", compute, [[1234, 9], 0], 0),
          throwAssertion("rejects-negative-line", compute, [[-1], 500], RangeError),
          throwAssertion("rejects-rate-above-full", compute, [[100], 10001], RangeError),
        ],
      };
    },
  };
}

/** Sliding window is exclusive at its edge and the lockout ladder escalates. */
function accountLockoutScenario(): CodeQualityScenario {
  return {
    scenarioId: "account-lockout",
    taskId: "task:code-quality:account-lockout",
    intent: "Evaluate account lockout under the approved failure-window policy.",
    publicPrompt:
      "Implement evaluateLockout in src/policy.js according to the current authentication lockout policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function evaluateLockout(_failureTimestamps, _nowIso) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { evaluateLockout } from "../src/policy.js";

test("a single recent failure does not lock the account", () => {
  assert.deepEqual(
    evaluateLockout(["2026-03-01T12:00:00.000Z"], "2026-03-01T12:01:00.000Z"),
    { locked: false, unlockAtIso: null },
  );
});
`,
    workItemId: "work-item:account-lockout",
    plannedSymbolId: "planned-symbol:evaluate-lockout",
    qualifiedName: "evaluateLockout",
    capabilityId: "capability:account-lockout",
    canonicalTerms: ["LOCKOUT_LADDER_MINUTES"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:account-lockout-policy",
        revisionId: "revision:account-lockout-policy:1",
        statement:
          "evaluateLockout counts only failures strictly inside the 15 minute window before nowIso, so a failure exactly 15 minutes old is excluded. Fewer than 3 counted failures returns { locked: false, unlockAtIso: null }. Otherwise the ladder is 3 to 4 failures 5 minutes, 5 to 6 failures 30 minutes, 7 or more failures 240 minutes, and unlockAtIso is the latest counted failure plus that duration. Input order is not guaranteed. A failure later than nowIso throws RangeError.",
        evidenceId: "evidence:security:lockout-policy",
        evidenceText:
          "The security owner approved a strictly-inside 15 minute window and an escalating 5/30/240 minute lockout ladder anchored on the most recent counted failure.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:lockout-ladder",
        revisionId: "revision:lockout-ladder:1",
        term: "Lockout Ladder",
        definition:
          "The escalating lockout durations in minutes. Implementation code names this table LOCKOUT_LADDER_MINUTES.",
        avoid: ["backoff table"],
        evidenceId: "evidence:domain:lockout-ladder",
        evidenceText:
          "The identity domain owner standardized Lockout Ladder and the code term LOCKOUT_LADDER_MINUTES.",
      },
      {
        kind: "invariant",
        recordId: "invariant:lockout-anchored-on-latest",
        revisionId: "revision:lockout-anchored-on-latest:1",
        statement: "An unlock time is never earlier than the most recent counted failure.",
        evidenceId: "evidence:security:lockout-anchor",
        evidenceText:
          "Incident response requires the lockout to run from the newest counted failure, never from the oldest.",
      },
    ],
    evaluateHidden: async (module) => {
      const evaluate = requiredFunction(module, "evaluateLockout");
      const at = (minute: number) => `2026-03-01T12:${String(minute).padStart(2, "0")}:00.000Z`;
      const now = "2026-03-01T12:30:00.000Z";
      return {
        assertions: [
          // 12:15 is exactly 15 minutes old, so the window excludes it and only
          // two failures count.
          callAssertion("window-edge-is-excluded", evaluate, [[at(15), at(20), at(25)], now], {
            locked: false,
            unlockAtIso: null,
          }),
          callAssertion(
            "three-failures-use-the-first-rung",
            evaluate,
            [[at(16), at(20), at(25)], now],
            { locked: true, unlockAtIso: "2026-03-01T12:30:00.000Z" },
          ),
          callAssertion(
            "five-failures-use-the-second-rung",
            evaluate,
            [[at(16), at(17), at(18), at(19), at(20)], now],
            { locked: true, unlockAtIso: "2026-03-01T12:50:00.000Z" },
          ),
          callAssertion(
            "seven-failures-use-the-top-rung",
            evaluate,
            [[at(16), at(17), at(18), at(19), at(20), at(21), at(22)], now],
            { locked: true, unlockAtIso: "2026-03-01T16:22:00.000Z" },
          ),
          // Unsorted input must still anchor on the newest counted failure.
          callAssertion(
            "unsorted-input-anchors-on-newest",
            evaluate,
            [[at(25), at(16), at(20)], now],
            { locked: true, unlockAtIso: "2026-03-01T12:30:00.000Z" },
          ),
          throwAssertion(
            "rejects-a-failure-after-now",
            evaluate,
            [["2026-03-01T12:31:00.000Z"], now],
            RangeError,
          ),
        ],
      };
    },
  };
}

/** The oversize surcharge is inside the zone multiplier, not added after it. */
function shippingQuoteScenario(): CodeQualityScenario {
  return {
    scenarioId: "shipping-quote",
    taskId: "task:code-quality:shipping-quote",
    intent: "Quote shipping under the approved weight-break and zone policy.",
    publicPrompt:
      "Implement quoteShippingCents in src/policy.js according to the current shipping rate policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function quoteShippingCents(_weightGrams, _zone) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { quoteShippingCents } from "../src/policy.js";

test("a light domestic parcel uses the base rate", () => {
  assert.equal(quoteShippingCents(500, "domestic"), 500);
});
`,
    workItemId: "work-item:shipping-quote",
    plannedSymbolId: "planned-symbol:quote-shipping-cents",
    qualifiedName: "quoteShippingCents",
    capabilityId: "capability:shipping-quote",
    canonicalTerms: ["zoneSurchargeCents"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:shipping-rate-policy",
        revisionId: "revision:shipping-rate-policy:1",
        statement:
          "quoteShippingCents uses weight breaks that include their lower bound and exclude their upper bound: under 1000 grams costs 500, 1000 to under 5000 grams costs 1200, and 5000 grams or more costs 2500. A parcel of 5000 grams or more adds an 800 cent oversize surcharge to the base rate before the zone multiplier is applied. Zone multipliers are domestic 1, regional 2, international 3. A negative or non-integer weight, or an unknown zone, throws RangeError.",
        evidenceId: "evidence:logistics:shipping-rate",
        evidenceText:
          "The logistics owner approved applying the oversize surcharge inside the zone multiplier because carriers bill oversize handling per zone, not once per shipment.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:zone-surcharge",
        revisionId: "revision:zone-surcharge:1",
        term: "Zone Surcharge",
        definition:
          "The zone-multiplied oversize amount added to a quote. Implementation code names this value zoneSurchargeCents.",
        avoid: ["oversize fee"],
        evidenceId: "evidence:domain:zone-surcharge",
        evidenceText:
          "The logistics domain owner standardized Zone Surcharge and the code term zoneSurchargeCents.",
      },
      {
        kind: "invariant",
        recordId: "invariant:shipping-quote-monotonic",
        revisionId: "revision:shipping-quote-monotonic:1",
        statement: "Within one zone a heavier parcel never quotes less than a lighter parcel.",
        evidenceId: "evidence:logistics:quote-monotonic",
        evidenceText: "Pricing requires shipping quotes to be non-decreasing in weight per zone.",
      },
    ],
    evaluateHidden: async (module) => {
      const quote = requiredFunction(module, "quoteShippingCents");
      return {
        assertions: [
          callAssertion("lower-bound-is-inclusive", quote, [1000, "domestic"], 1200),
          callAssertion("upper-bound-is-exclusive", quote, [4999, "domestic"], 1200),
          callAssertion("oversize-applies-at-the-break", quote, [5000, "domestic"], 3300),
          // Surcharge inside the multiplier gives 9900; adding it afterwards
          // would give 8300.
          callAssertion("surcharge-precedes-the-multiplier", quote, [5000, "international"], 9900),
          callAssertion("regional-doubles-the-base", quote, [999, "regional"], 1000),
          throwAssertion("rejects-unknown-zone", quote, [500, "lunar"], RangeError),
        ],
      };
    },
  };
}

/** A 30/360 month that excludes the change day itself. */
function prorationScenario(): CodeQualityScenario {
  return {
    scenarioId: "subscription-proration",
    taskId: "task:code-quality:subscription-proration",
    intent: "Prorate a subscription upgrade under the approved billing convention.",
    publicPrompt:
      "Implement prorateUpgradeCents in src/policy.js according to the current proration policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function prorateUpgradeCents(_monthlyCents, _changeDayOfMonth) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { prorateUpgradeCents } from "../src/policy.js";

test("a zero-priced plan prorates to nothing", () => {
  assert.equal(prorateUpgradeCents(0, 10), 0);
});
`,
    workItemId: "work-item:subscription-proration",
    plannedSymbolId: "planned-symbol:prorate-upgrade-cents",
    qualifiedName: "prorateUpgradeCents",
    capabilityId: "capability:subscription-proration",
    canonicalTerms: ["proratedChargeCents"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:proration-convention",
        revisionId: "revision:proration-convention:1",
        statement:
          "prorateUpgradeCents uses a 30 day billing month and charges only for days after the change day, so remaining days is 30 minus changeDayOfMonth and the change day itself is not charged. The result is floor(monthlyCents * remainingDays / 30). A changeDayOfMonth of 31 is treated as 30 and therefore prorates to zero. monthlyCents must be a non-negative integer and changeDayOfMonth an integer from 1 to 31; otherwise throw RangeError.",
        evidenceId: "evidence:billing:proration-convention",
        evidenceText:
          "The billing owner approved the 30/360 convention with the change day billed on the old plan, so the upgrade charge starts the following day.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:prorated-charge",
        revisionId: "revision:prorated-charge:1",
        term: "Prorated Charge",
        definition:
          "The floored upgrade amount for the remaining billing days. Implementation code names this value proratedChargeCents.",
        avoid: ["partial charge"],
        evidenceId: "evidence:domain:prorated-charge",
        evidenceText:
          "The billing domain owner standardized Prorated Charge and the code term proratedChargeCents.",
      },
      {
        kind: "invariant",
        recordId: "invariant:proration-never-exceeds-month",
        revisionId: "revision:proration-never-exceeds-month:1",
        statement: "A Prorated Charge never exceeds the full monthly amount.",
        evidenceId: "evidence:billing:proration-bound",
        evidenceText: "Revenue requires a prorated upgrade to stay at or below one month's price.",
      },
    ],
    evaluateHidden: async (module) => {
      const prorate = requiredFunction(module, "prorateUpgradeCents");
      return {
        assertions: [
          // Excluding the change day gives 15/30; including it would give 16/30.
          callAssertion("excludes-the-change-day", prorate, [3000, 15], 1500),
          callAssertion("day-one-charges-twenty-nine-days", prorate, [3000, 1], 2900),
          callAssertion("last-billing-day-prorates-to-zero", prorate, [3000, 30], 0),
          callAssertion("day-thirty-one-collapses-to-thirty", prorate, [3000, 31], 0),
          callAssertion("floors-a-fractional-cent", prorate, [1000, 14], 533),
          throwAssertion("rejects-day-zero", prorate, [3000, 0], RangeError),
        ],
      };
    },
  };
}

/** Priority tier outranks the promise date, and reservations are all-or-nothing. */
function inventoryReservationScenario(): CodeQualityScenario {
  return {
    scenarioId: "inventory-reservation",
    taskId: "task:code-quality:inventory-reservation",
    intent: "Reserve inventory under the approved allocation-order policy.",
    publicPrompt:
      "Implement reserveInventory in src/policy.js according to the current inventory allocation policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function reserveInventory(_availableUnits, _requests) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { reserveInventory } from "../src/policy.js";

test("a single request within stock is reserved in full", () => {
  assert.deepEqual(
    reserveInventory(10, [
      { orderId: "order:a", quantity: 4, priorityTier: 1, promisedAt: "2026-03-01T00:00:00.000Z" },
    ]),
    { "order:a": 4 },
  );
});
`,
    workItemId: "work-item:inventory-reservation",
    plannedSymbolId: "planned-symbol:reserve-inventory",
    qualifiedName: "reserveInventory",
    capabilityId: "capability:inventory-reservation",
    canonicalTerms: ["reservationLedger"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:inventory-allocation-order",
        revisionId: "revision:inventory-allocation-order:1",
        statement:
          "reserveInventory sorts requests by priorityTier ascending where 1 is highest, then promisedAt ascending, then orderId lexicographically. Each request is all-or-nothing: a request larger than the remaining units reserves 0 and allocation continues with the following requests rather than stopping. The result contains every orderId, keyed in that sorted order. A negative availableUnits or a non-positive quantity throws RangeError.",
        evidenceId: "evidence:fulfillment:allocation-order",
        evidenceText:
          "Fulfillment approved priority-first allocation with all-or-nothing reservations because partial reservations strand units that cannot ship, and skipped requests must not block smaller ones behind them.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:reservation-ledger",
        revisionId: "revision:reservation-ledger:1",
        term: "Reservation Ledger",
        definition:
          "The per-order reserved quantities produced by one allocation pass. Implementation code names this accumulator reservationLedger.",
        avoid: ["allocation map"],
        evidenceId: "evidence:domain:reservation-ledger",
        evidenceText:
          "The fulfillment domain owner standardized Reservation Ledger and the code term reservationLedger.",
      },
      {
        kind: "invariant",
        recordId: "invariant:reservation-within-stock",
        revisionId: "revision:reservation-within-stock:1",
        statement: "Reserved units never exceed the available units.",
        evidenceId: "evidence:fulfillment:stock-bound",
        evidenceText: "Warehouse control requires total reservations to stay within stock on hand.",
      },
    ],
    evaluateHidden: async (module) => {
      const reserve = requiredFunction(module, "reserveInventory");
      const request = (
        orderId: string,
        quantity: number,
        priorityTier: number,
        promisedAt: string,
      ) => ({ orderId, quantity, priorityTier, promisedAt });
      return {
        assertions: [
          // Tier 1 wins even though the tier 2 request was promised earlier.
          callAssertion(
            "priority-outranks-the-promise-date",
            reserve,
            [
              5,
              [
                request("order:late", 5, 1, "2026-03-05T00:00:00.000Z"),
                request("order:early", 5, 2, "2026-03-01T00:00:00.000Z"),
              ],
            ],
            { "order:late": 5, "order:early": 0 },
          ),
          // The oversized request takes nothing and the next one still fills.
          callAssertion(
            "all-or-nothing-skips-without-blocking",
            reserve,
            [
              4,
              [
                request("order:big", 9, 1, "2026-03-01T00:00:00.000Z"),
                request("order:small", 3, 1, "2026-03-02T00:00:00.000Z"),
              ],
            ],
            { "order:big": 0, "order:small": 3 },
          ),
          callAssertion(
            "ties-break-on-order-id",
            reserve,
            [
              3,
              [
                request("order:b", 3, 1, "2026-03-01T00:00:00.000Z"),
                request("order:a", 3, 1, "2026-03-01T00:00:00.000Z"),
              ],
            ],
            { "order:a": 3, "order:b": 0 },
          ),
          callAssertion(
            "exact-stock-is-fully-reserved",
            reserve,
            [6, [request("order:a", 6, 1, "2026-03-01T00:00:00.000Z")]],
            { "order:a": 6 },
          ),
          throwAssertion(
            "rejects-negative-stock",
            reserve,
            [-1, [request("order:a", 1, 1, "2026-03-01T00:00:00.000Z")]],
            RangeError,
          ),
        ],
      };
    },
  };
}

/** Reporter trust moves the tier in one direction only, per level. */
function moderationEscalationScenario(): CodeQualityScenario {
  return {
    scenarioId: "moderation-escalation",
    taskId: "task:code-quality:moderation-escalation",
    intent: "Choose a moderation escalation tier under the approved trust policy.",
    publicPrompt:
      "Implement resolveEscalation in src/policy.js according to the current moderation escalation policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function resolveEscalation(_severity, _reporterTrust) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { resolveEscalation } from "../src/policy.js";

test("a high-severity report from a standard reporter is urgent", () => {
  assert.equal(resolveEscalation("high", "standard"), "urgent");
});
`,
    workItemId: "work-item:moderation-escalation",
    plannedSymbolId: "planned-symbol:resolve-escalation",
    qualifiedName: "resolveEscalation",
    capabilityId: "capability:moderation-escalation",
    canonicalTerms: ["ESCALATION_OVERRIDES"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:moderation-escalation-policy",
        revisionId: "revision:moderation-escalation-policy:1",
        statement:
          "resolveEscalation maps severity low to 'monitor', medium to 'review', and high to 'urgent'. A 'verified' reporterTrust raises medium to 'urgent' but leaves low unchanged. A 'flagged' reporterTrust lowers high to 'review' but leaves medium unchanged. An unknown severity or reporterTrust throws RangeError.",
        evidenceId: "evidence:trust:escalation-policy",
        evidenceText:
          "The trust and safety owner approved raising only medium reports from verified reporters, since verified low-severity reports still do not warrant human urgency, and lowering only high reports from flagged reporters.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:escalation-override",
        revisionId: "revision:escalation-override:1",
        term: "Escalation Override",
        definition:
          "A reporter-trust adjustment applied to a severity's base tier. Implementation code names this table ESCALATION_OVERRIDES.",
        avoid: ["trust boost"],
        evidenceId: "evidence:domain:escalation-override",
        evidenceText:
          "The trust and safety domain owner standardized Escalation Override and the code term ESCALATION_OVERRIDES.",
      },
      {
        kind: "invariant",
        recordId: "invariant:escalation-tier-known",
        revisionId: "revision:escalation-tier-known:1",
        statement: "Every resolved escalation is monitor, review, or urgent.",
        evidenceId: "evidence:trust:tier-closed",
        evidenceText: "The moderation queue accepts only these three escalation tiers.",
      },
    ],
    evaluateHidden: async (module) => {
      const resolve = requiredFunction(module, "resolveEscalation");
      return {
        assertions: [
          callAssertion("verified-raises-medium", resolve, ["medium", "verified"], "urgent"),
          callAssertion("verified-leaves-low-alone", resolve, ["low", "verified"], "monitor"),
          callAssertion("flagged-lowers-high", resolve, ["high", "flagged"], "review"),
          callAssertion("flagged-leaves-medium-alone", resolve, ["medium", "flagged"], "review"),
          callAssertion("standard-uses-the-base-tier", resolve, ["low", "standard"], "monitor"),
          throwAssertion("rejects-unknown-severity", resolve, ["critical", "standard"], RangeError),
        ],
      };
    },
  };
}

/** Refill is floored per whole minute, with no fractional carry. */
function rateQuotaScenario(): CodeQualityScenario {
  return {
    scenarioId: "api-rate-quota",
    taskId: "task:code-quality:api-rate-quota",
    intent: "Compute available API quota under the approved refill policy.",
    publicPrompt:
      "Implement availableTokens in src/policy.js according to the current API quota policy. Keep the public API stable and run the tests.",
    sourceFile: "src/policy.js",
    initialSource: `export function availableTokens(_state, _nowMs) {
  throw new Error("Not implemented");
}
`,
    publicTestSource: `import assert from "node:assert/strict";
import test from "node:test";
import { availableTokens } from "../src/policy.js";

test("no elapsed time leaves the balance untouched", () => {
  assert.equal(
    availableTokens(
      { tokens: 5, lastRefillMs: 1000, sustainedPerMinute: 60, burstCapacity: 100 },
      1000,
    ),
    5,
  );
});
`,
    workItemId: "work-item:api-rate-quota",
    plannedSymbolId: "planned-symbol:available-tokens",
    qualifiedName: "availableTokens",
    capabilityId: "capability:api-rate-quota",
    canonicalTerms: ["burstCapacityTokens"],
    rules: [
      {
        kind: "decision",
        recordId: "decision:api-quota-refill",
        revisionId: "revision:api-quota-refill:1",
        statement:
          "availableTokens refills only for whole elapsed minutes: refill is floor((nowMs - lastRefillMs) / 60000) multiplied by sustainedPerMinute, so a partial minute grants nothing and no fractional remainder is carried. The result is capped at burstCapacity and never below zero. A nowMs earlier than lastRefillMs throws RangeError.",
        evidenceId: "evidence:platform:quota-refill",
        evidenceText:
          "The platform owner approved whole-minute refills without fractional carry so that quota is reproducible from the stored state alone and cannot be advanced by polling more often.",
      },
      {
        kind: "domain_term",
        recordId: "domain-term:burst-capacity",
        revisionId: "revision:burst-capacity:1",
        term: "Burst Capacity",
        definition:
          "The ceiling a refilled quota may reach. Implementation code names this bound burstCapacityTokens.",
        avoid: ["max tokens"],
        evidenceId: "evidence:domain:burst-capacity",
        evidenceText:
          "The platform domain owner standardized Burst Capacity and the code term burstCapacityTokens.",
      },
      {
        kind: "invariant",
        recordId: "invariant:quota-within-burst",
        revisionId: "revision:quota-within-burst:1",
        statement: "Available quota never exceeds Burst Capacity and is never negative.",
        evidenceId: "evidence:platform:quota-bound",
        evidenceText: "The gateway requires quota to stay within its burst ceiling at all times.",
      },
    ],
    evaluateHidden: async (module) => {
      const available = requiredFunction(module, "availableTokens");
      const state = {
        tokens: 10,
        lastRefillMs: 0,
        sustainedPerMinute: 60,
        burstCapacity: 100,
      };
      return {
        assertions: [
          // 59.999 seconds is not a whole minute, so nothing is granted.
          callAssertion("partial-minute-grants-nothing", available, [state, 59_999], 10),
          callAssertion("one-whole-minute-grants-the-rate", available, [state, 60_000], 70),
          // 90 seconds is one whole minute; the extra 30 seconds is not carried.
          callAssertion("fraction-is-not-carried", available, [state, 90_000], 70),
          callAssertion("refill-stops-at-burst-capacity", available, [state, 600_000], 100),
          callAssertion(
            "an-overdrawn-balance-never-goes-negative",
            available,
            [{ ...state, tokens: -5 }, 0],
            0,
          ),
          throwAssertion(
            "rejects-time-moving-backward",
            available,
            [{ ...state, lastRefillMs: 5000 }, 1000],
            RangeError,
          ),
        ],
      };
    },
  };
}
