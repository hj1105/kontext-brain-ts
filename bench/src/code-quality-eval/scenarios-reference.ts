/**
 * Two implementations per scenario, used to prove each scenario is a valid
 * benchmark item rather than trusting the fixtures by inspection.
 *
 * `referenceImplementations` satisfy the held-out policy exactly, so every
 * hidden assertion must pass. `naiveImplementations` are the choice a competent
 * engineer reaches from the public prompt and public test alone, so each must
 * pass the public test and fail at least one hidden assertion. A scenario where
 * the naive version already passes everything hands the baseline arm free
 * credit and measures nothing.
 */
export type ScenarioModule = Readonly<Record<string, unknown>>;

interface ReservationRequest {
  readonly orderId: string;
  readonly quantity: number;
  readonly priorityTier: number;
  readonly promisedAt: string;
}

interface QuotaState {
  readonly tokens: number;
  readonly lastRefillMs: number;
  readonly sustainedPerMinute: number;
  readonly burstCapacity: number;
}

export const referenceImplementations: Readonly<Record<string, ScenarioModule>> = {
  "retry-policy": {
    computeRetryDelay(failureIndex: number, baseMs: number): number {
      if (
        !Number.isInteger(failureIndex) ||
        failureIndex < 0 ||
        !Number.isInteger(baseMs) ||
        baseMs < 0
      ) {
        throw new RangeError("invalid retry input");
      }
      return Math.min(baseMs * 3 ** failureIndex, 4_500);
    },
  },
  "order-cancellation": {
    cancellationOutcome(order: {
      state: string;
      shipmentId: string | null;
      fraudHold: boolean;
    }): string {
      return order.state === "confirmed" && !order.shipmentId && order.fraudHold !== true
        ? "revocable"
        : "locked";
    },
  },
  "service-credit-allocation": {
    allocateServiceCredit(totalCents: number, accountIds: readonly string[]): object {
      if (!Number.isInteger(totalCents) || totalCents < 0) throw new RangeError("invalid total");
      if (
        accountIds.length === 0 ||
        accountIds.some((id) => !id) ||
        new Set(accountIds).size !== accountIds.length
      ) {
        throw new Error("invalid accounts");
      }
      const sorted = [...accountIds].sort();
      const base = Math.floor(totalCents / sorted.length);
      const values = new Map(sorted.map((id) => [id, base] as const));
      let remainder = totalCents % sorted.length;
      for (let index = sorted.length - 1; remainder > 0; index -= 1) {
        const id = sorted[index];
        if (!id) throw new Error("missing account");
        values.set(id, base + 1);
        remainder -= 1;
      }
      return Object.fromEntries(sorted.map((id) => [id, values.get(id)]));
    },
  },
  "invoice-tax-rounding": {
    computeInvoiceTaxCents(
      lineAmountsCents: readonly number[],
      taxRateBasisPoints: number,
    ): number {
      if (
        !Number.isInteger(taxRateBasisPoints) ||
        taxRateBasisPoints < 0 ||
        taxRateBasisPoints > 10_000
      ) {
        throw new RangeError("invalid tax rate");
      }
      let invoiceTaxCents = 0;
      for (const lineAmountCents of lineAmountsCents) {
        if (!Number.isInteger(lineAmountCents) || lineAmountCents < 0) {
          throw new RangeError("invalid line amount");
        }
        const exact = (lineAmountCents * taxRateBasisPoints) / 10_000;
        const lower = Math.floor(exact);
        const fraction = exact - lower;
        const roundedLineTaxCents =
          fraction > 0.5 ? lower + 1 : fraction < 0.5 ? lower : lower % 2 === 0 ? lower : lower + 1;
        invoiceTaxCents += roundedLineTaxCents;
      }
      return invoiceTaxCents;
    },
  },
  "account-lockout": {
    evaluateLockout(
      failureTimestamps: readonly string[],
      nowIso: string,
    ): { locked: boolean; unlockAtIso: string | null } {
      const LOCKOUT_LADDER_MINUTES = [
        { atLeast: 7, minutes: 240 },
        { atLeast: 5, minutes: 30 },
        { atLeast: 3, minutes: 5 },
      ];
      const now = Date.parse(nowIso);
      if (Number.isNaN(now)) throw new RangeError("invalid now");
      const windowStart = now - 15 * 60_000;
      const counted: number[] = [];
      for (const timestamp of failureTimestamps) {
        const at = Date.parse(timestamp);
        if (Number.isNaN(at)) throw new RangeError("invalid failure timestamp");
        if (at > now) throw new RangeError("failure after now");
        if (at > windowStart) counted.push(at);
      }
      const rung = LOCKOUT_LADDER_MINUTES.find((entry) => counted.length >= entry.atLeast);
      if (!rung) return { locked: false, unlockAtIso: null };
      const latest = Math.max(...counted);
      return {
        locked: true,
        unlockAtIso: new Date(latest + rung.minutes * 60_000).toISOString(),
      };
    },
  },
  "shipping-quote": {
    quoteShippingCents(weightGrams: number, zone: string): number {
      if (!Number.isInteger(weightGrams) || weightGrams < 0) throw new RangeError("invalid weight");
      const multiplier = ({ domestic: 1, regional: 2, international: 3 } as Record<string, number>)[
        zone
      ];
      if (multiplier === undefined) throw new RangeError("unknown zone");
      const baseCents = weightGrams < 1_000 ? 500 : weightGrams < 5_000 ? 1_200 : 2_500;
      const zoneSurchargeCents = (weightGrams >= 5_000 ? 800 : 0) * multiplier;
      return baseCents * multiplier + zoneSurchargeCents;
    },
  },
  "subscription-proration": {
    prorateUpgradeCents(monthlyCents: number, changeDayOfMonth: number): number {
      if (!Number.isInteger(monthlyCents) || monthlyCents < 0)
        throw new RangeError("invalid price");
      if (!Number.isInteger(changeDayOfMonth) || changeDayOfMonth < 1 || changeDayOfMonth > 31) {
        throw new RangeError("invalid change day");
      }
      const remainingDays = 30 - Math.min(changeDayOfMonth, 30);
      const proratedChargeCents = Math.floor((monthlyCents * remainingDays) / 30);
      return proratedChargeCents;
    },
  },
  "inventory-reservation": {
    reserveInventory(availableUnits: number, requests: readonly ReservationRequest[]): object {
      if (!Number.isInteger(availableUnits) || availableUnits < 0) {
        throw new RangeError("invalid stock");
      }
      for (const request of requests) {
        if (!Number.isInteger(request.quantity) || request.quantity <= 0) {
          throw new RangeError("invalid quantity");
        }
      }
      const ordered = [...requests].sort(
        (left, right) =>
          left.priorityTier - right.priorityTier ||
          left.promisedAt.localeCompare(right.promisedAt) ||
          left.orderId.localeCompare(right.orderId),
      );
      const reservationLedger: Record<string, number> = {};
      let remaining = availableUnits;
      for (const request of ordered) {
        if (request.quantity <= remaining) {
          reservationLedger[request.orderId] = request.quantity;
          remaining -= request.quantity;
        } else {
          reservationLedger[request.orderId] = 0;
        }
      }
      return reservationLedger;
    },
  },
  "moderation-escalation": {
    resolveEscalation(severity: string, reporterTrust: string): string {
      const baseTier: Readonly<Record<string, string>> = {
        low: "monitor",
        medium: "review",
        high: "urgent",
      };
      const ESCALATION_OVERRIDES: Readonly<Record<string, Readonly<Record<string, string>>>> = {
        verified: { medium: "urgent" },
        flagged: { high: "review" },
        standard: {},
      };
      const tier = baseTier[severity];
      if (tier === undefined) throw new RangeError("unknown severity");
      const overrides = ESCALATION_OVERRIDES[reporterTrust];
      if (overrides === undefined) throw new RangeError("unknown reporter trust");
      return overrides[severity] ?? tier;
    },
  },
  "api-rate-quota": {
    availableTokens(state: QuotaState, nowMs: number): number {
      if (nowMs < state.lastRefillMs) throw new RangeError("time moved backward");
      const wholeMinutes = Math.floor((nowMs - state.lastRefillMs) / 60_000);
      const burstCapacityTokens = state.burstCapacity;
      const refilled = state.tokens + wholeMinutes * state.sustainedPerMinute;
      return Math.max(0, Math.min(refilled, burstCapacityTokens));
    },
  },
  "ledger-posting": {
    classifyPosting(posting: {
      amountCents: number;
      clearedAt: string | null;
      disputed: boolean;
    }): string {
      const DISPUTE_HOLD_CENTS = 2500;
      if (posting.amountCents <= 0) return "rejected";
      if (posting.disputed && posting.amountCents > DISPUTE_HOLD_CENTS) return "pending";
      return posting.clearedAt ? "settled" : "pending";
    },
  },
  "membership-tier": {
    MembershipPolicy: class {
      resolve(monthsActive: number, lifetimeSpendCents: number): string {
        const TENURE_CREDIT_CENTS = 1_000;
        if (monthsActive < 0 || lifetimeSpendCents < 0) throw new RangeError("negative input");
        const credited = lifetimeSpendCents + Math.floor(monthsActive / 12) * TENURE_CREDIT_CENTS;
        if (credited >= 50_000) return "gold";
        if (credited >= 20_000) return "silver";
        return "standard";
      }
    },
  },
};

export const naiveImplementations: Readonly<Record<string, ScenarioModule>> = {
  // Doubling with no cap is the common backoff default.
  "retry-policy": {
    computeRetryDelay(failureIndex: number, baseMs: number): number {
      return baseMs * 2 ** failureIndex;
    },
  },
  // Treating a captured payment as blocking is the intuitive reading.
  "order-cancellation": {
    cancellationOutcome(order: {
      state: string;
      shipmentId: string | null;
      fraudHold: boolean;
      paymentCaptured?: boolean;
    }): string {
      return order.state === "confirmed" && !order.shipmentId && !order.paymentCaptured
        ? "revocable"
        : "locked";
    },
  },
  // Remainder to the first accounts is the natural loop direction.
  "service-credit-allocation": {
    allocateServiceCredit(totalCents: number, accountIds: readonly string[]): object {
      const sorted = [...accountIds].sort();
      const base = Math.floor(totalCents / sorted.length);
      const remainder = totalCents % sorted.length;
      return Object.fromEntries(
        sorted.map((id, index) => [id, index < remainder ? base + 1 : base]),
      );
    },
  },
  // Rounding the invoice total with half-up is the default reading.
  "invoice-tax-rounding": {
    computeInvoiceTaxCents(
      lineAmountsCents: readonly number[],
      taxRateBasisPoints: number,
    ): number {
      const subtotal = lineAmountsCents.reduce((sum, amount) => sum + amount, 0);
      return Math.round((subtotal * taxRateBasisPoints) / 10_000);
    },
  },
  // Counting every failure with one fixed duration is the obvious approach.
  "account-lockout": {
    evaluateLockout(
      failureTimestamps: readonly string[],
      nowIso: string,
    ): { locked: boolean; unlockAtIso: string | null } {
      if (failureTimestamps.length < 3) return { locked: false, unlockAtIso: null };
      const latest = Math.max(...failureTimestamps.map((value) => Date.parse(value)));
      void nowIso;
      return { locked: true, unlockAtIso: new Date(latest + 15 * 60_000).toISOString() };
    },
  },
  // Inclusive upper bounds and a flat surcharge added after the multiplier.
  "shipping-quote": {
    quoteShippingCents(weightGrams: number, zone: string): number {
      const multiplier =
        ({ domestic: 1, regional: 2, international: 3 } as Record<string, number>)[zone] ?? 1;
      const baseCents = weightGrams <= 1_000 ? 500 : weightGrams <= 5_000 ? 1_200 : 2_500;
      const oversizeCents = weightGrams > 5_000 ? 800 : 0;
      return baseCents * multiplier + oversizeCents;
    },
  },
  // Counting the change day itself is the intuitive inclusive reading.
  "subscription-proration": {
    prorateUpgradeCents(monthlyCents: number, changeDayOfMonth: number): number {
      const remainingDays = 30 - changeDayOfMonth + 1;
      return Math.round((monthlyCents * remainingDays) / 30);
    },
  },
  // First-promised-first-served with partial fills is the common queue default.
  "inventory-reservation": {
    reserveInventory(availableUnits: number, requests: readonly ReservationRequest[]): object {
      const ordered = [...requests].sort((left, right) =>
        left.promisedAt.localeCompare(right.promisedAt),
      );
      const ledger: Record<string, number> = {};
      let remaining = availableUnits;
      for (const request of ordered) {
        const granted = Math.min(request.quantity, remaining);
        ledger[request.orderId] = granted;
        remaining -= granted;
      }
      return ledger;
    },
  },
  // Letting trust move every severity uniformly is the symmetric guess.
  "moderation-escalation": {
    resolveEscalation(severity: string, reporterTrust: string): string {
      const ladder = ["monitor", "review", "urgent"];
      const baseIndex = { low: 0, medium: 1, high: 2 }[severity];
      if (baseIndex === undefined) throw new RangeError("unknown severity");
      const shift = reporterTrust === "verified" ? 1 : reporterTrust === "flagged" ? -1 : 0;
      const index = Math.min(ladder.length - 1, Math.max(0, baseIndex + shift));
      return ladder[index] as string;
    },
  },
  // Continuous fractional refill is the textbook token bucket.
  "api-rate-quota": {
    availableTokens(state: QuotaState, nowMs: number): number {
      const elapsedMinutes = (nowMs - state.lastRefillMs) / 60_000;
      const refilled = state.tokens + elapsedMinutes * state.sustainedPerMinute;
      return Math.min(Math.round(refilled), state.burstCapacity);
    },
  },
  // Treating any dispute as blocking is the intuitive reading.
  "ledger-posting": {
    classifyPosting(posting: {
      amountCents: number;
      clearedAt: string | null;
      disputed: boolean;
    }): string {
      if (posting.amountCents <= 0) return "rejected";
      if (posting.disputed) return "pending";
      return posting.clearedAt ? "settled" : "pending";
    },
  },
  // Comparing raw spend and treating tenure as its own rule is the obvious split.
  "membership-tier": {
    MembershipPolicy: class {
      resolve(monthsActive: number, lifetimeSpendCents: number): string {
        if (lifetimeSpendCents >= 50_000 || monthsActive >= 60) return "gold";
        if (lifetimeSpendCents >= 20_000 || monthsActive >= 24) return "silver";
        return "standard";
      }
    },
  },
};
