import { isDeepStrictEqual } from "node:util";
import type { HiddenAssertionResult } from "./contracts.js";

export function requiredFunction(
  module: Readonly<Record<string, unknown>>,
  exportName: string,
): (...args: readonly unknown[]) => unknown {
  const value = module[exportName];
  if (typeof value !== "function") throw new Error(`Missing function export ${exportName}`);
  return value as (...args: readonly unknown[]) => unknown;
}

export function callAssertion(
  assertionId: string,
  operation: (...args: readonly unknown[]) => unknown,
  args: readonly unknown[],
  expected: unknown,
): HiddenAssertionResult {
  try {
    const actual = operation(...args);
    return isDeepStrictEqual(actual, expected)
      ? { assertionId, passed: true }
      : {
          assertionId,
          passed: false,
          diagnostic: `Expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`,
        };
  } catch (error) {
    return { assertionId, passed: false, diagnostic: errorMessage(error) };
  }
}

export function throwAssertion(
  assertionId: string,
  operation: (...args: readonly unknown[]) => unknown,
  args: readonly unknown[],
  expected: new (...args: never[]) => Error,
): HiddenAssertionResult {
  try {
    operation(...args);
    return { assertionId, passed: false, diagnostic: `Expected ${expected.name}` };
  } catch (error) {
    return error instanceof expected
      ? { assertionId, passed: true }
      : {
          assertionId,
          passed: false,
          diagnostic: `Expected ${expected.name}, received ${errorMessage(error)}`,
        };
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}
