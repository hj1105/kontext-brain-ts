/**
 * Generates a repository where the answer is known by construction.
 *
 * The single-function scenarios could not measure the product's actual claim.
 * With one file and one symbol, "did it change the right symbol and only that
 * symbol" is trivially satisfied, so the only remaining question was whether a
 * model can apply a rule it was handed — which is saturated, and which is why
 * the retrieval arm scored 100%.
 *
 * Here one subsystem is governed by the Decision under test and three others
 * carry deliberately similar backoff code governed by different, unchanged
 * Decisions. Finding the governed set among lookalikes is the thing a curated
 * symbol-to-Decision graph can do and text similarity cannot, and it is
 * measurable because the generator emits the ground truth alongside the code.
 */

export interface GeneratedFunction {
  readonly file: string;
  readonly name: string;
  readonly subsystem: string;
  readonly governed: boolean;
}

export interface GeneratedRepository {
  readonly files: ReadonlyMap<string, string>;
  readonly functions: readonly GeneratedFunction[];
  readonly governedNames: readonly string[];
  readonly decoyNames: readonly string[];
}

interface SubsystemSpec {
  readonly name: string;
  readonly directory: string;
  /** Multiplier the subsystem's own approved Decision fixes. */
  readonly factor: number;
  readonly capMs: number;
  readonly operations: readonly string[];
}

/**
 * Every subsystem computes a capped exponential backoff, so the code reads the
 * same everywhere. Only `billing` is governed by the Decision under test.
 */
export const subsystems: readonly SubsystemSpec[] = [
  {
    name: "billing",
    directory: "src/billing",
    factor: 2,
    capMs: 30_000,
    operations: ["charge", "refund", "capture", "void", "settle", "dispute", "payout", "reconcile"],
  },
  {
    name: "notify",
    directory: "src/notify",
    factor: 2,
    capMs: 30_000,
    operations: ["push", "email", "sms", "webhook", "digest", "receipt", "alert", "invite"],
  },
  {
    name: "sync",
    directory: "src/sync",
    factor: 2,
    capMs: 30_000,
    operations: ["pull", "push", "merge", "reindex", "purge", "rebuild", "verify", "compact"],
  },
  {
    name: "media",
    directory: "src/media",
    factor: 2,
    capMs: 30_000,
    operations: ["upload", "transcode", "thumbnail", "probe", "publish", "archive", "restore"],
  },
];

export const governedSubsystem = "billing";

function functionName(subsystem: string, operation: string): string {
  return `${operation}${subsystem[0]?.toUpperCase()}${subsystem.slice(1)}RetryDelay`;
}

function functionSource(subsystem: SubsystemSpec, operation: string): string {
  const name = functionName(subsystem.name, operation);
  return `/**
 * Retry delay for the ${operation} step of the ${subsystem.name} subsystem.
 */
export function ${name}(failureIndex, baseMs) {
  if (!Number.isInteger(failureIndex) || failureIndex < 0) {
    throw new RangeError("failureIndex must be a non-negative integer");
  }
  if (!Number.isInteger(baseMs) || baseMs < 0) {
    throw new RangeError("baseMs must be a non-negative integer");
  }
  return Math.min(baseMs * ${subsystem.factor} ** failureIndex, ${subsystem.capMs});
}
`;
}

const barrelFile = "index";

export function generateRepository(): GeneratedRepository {
  const files = new Map<string, string>();
  const functions: GeneratedFunction[] = [];

  for (const subsystem of subsystems) {
    // An operation named like the barrel would overwrite it and make the
    // re-export cycle back on itself, which surfaces much later as an import
    // error inside the fixture rather than as a generation failure.
    if (subsystem.operations.includes(barrelFile)) {
      throw new Error(
        `Subsystem ${subsystem.name} has an operation named "${barrelFile}", which collides with its barrel file`,
      );
    }
    if (new Set(subsystem.operations).size !== subsystem.operations.length) {
      throw new Error(`Subsystem ${subsystem.name} repeats an operation`);
    }
  }

  for (const subsystem of subsystems) {
    for (const operation of subsystem.operations) {
      const file = `${subsystem.directory}/${operation}.js`;
      files.set(file, functionSource(subsystem, operation));
      functions.push({
        file,
        name: functionName(subsystem.name, operation),
        subsystem: subsystem.name,
        governed: subsystem.name === governedSubsystem,
      });
    }
  }

  // An index per subsystem, so the repository has realistic re-exports and the
  // agent cannot assume one file holds everything.
  for (const subsystem of subsystems) {
    const exports = subsystem.operations
      .map(
        (operation) =>
          `export { ${functionName(subsystem.name, operation)} } from "./${operation}.js";`,
      )
      .join("\n");
    files.set(`${subsystem.directory}/${barrelFile}.js`, `${exports}\n`);
  }

  return {
    files,
    functions,
    governedNames: functions.filter((item) => item.governed).map((item) => item.name),
    decoyNames: functions.filter((item) => !item.governed).map((item) => item.name),
  };
}

/**
 * The policy under test, held only in the sidecar. The public surface never
 * states the factor, the cap, or the constant's name.
 */
export const governedPolicy = {
  factor: 3,
  capMs: 45_000,
  constantName: "BILLING_RECOVERY_CEILING_MS",
  /**
   * The ceiling must live in one shared module that the governed files import,
   * rather than being copied into each of them. That turns the task into real
   * structural work — create a module, export the constant, rewire eight
   * imports — instead of the same edit repeated eight times, and a maintainer
   * would know it while a newcomer would not.
   */
  sharedModule: "src/billing/recovery-ceiling.js",
} as const;

export function expectedGovernedDelay(failureIndex: number, baseMs: number): number {
  return Math.min(baseMs * governedPolicy.factor ** failureIndex, governedPolicy.capMs);
}

/** The behaviour a decoy must keep. */
export function expectedDecoyDelay(failureIndex: number, baseMs: number): number {
  return Math.min(baseMs * 2 ** failureIndex, 30_000);
}
