import { describe, expect, it } from "vitest";
import { PythonCodeProvider } from "../src/python-code-provider.js";

const provider = new PythonCodeProvider();

function analyze(content: string, targetPath = "src/policy.py") {
  return provider.analyze({
    codebaseId: "codebase:test",
    targetPath,
    files: [{ path: targetPath, content }],
  });
}

function symbol(content: string, qualifiedName: string) {
  const found = analyze(content).symbols.find(
    (candidate) => candidate.identity.qualifiedName === qualifiedName,
  );
  if (!found) throw new Error(`Missing symbol ${qualifiedName}`);
  return found;
}

const implementation = `RECOVERY_WINDOW_MS = 4500


def compute_retry_delay(failure_index: int, base_ms: int) -> int:
    """Return the approved delay."""
    if failure_index < 0:
        raise ValueError("negative")
    return min(base_ms * 3 ** failure_index, RECOVERY_WINDOW_MS)


class RetryPolicy:
    def describe(self) -> str:
        return "policy"

    def _internal(self) -> None:
        return None
`;

describe("PythonCodeProvider", () => {
  it("reports syntactic support because Python has no type checker here", () => {
    expect(provider.language).toBe("python");
    expect(provider.semanticSupport).toBe("syntactic");
    expect(symbol(implementation, "compute_retry_delay").semanticSupport).toBe("syntactic");
  });

  it("extracts module, function, class, method, and constant symbols", () => {
    const analysis = analyze(implementation);
    const byName = new Map(
      analysis.symbols.map((record) => [record.identity.qualifiedName, record] as const),
    );
    expect(byName.get("<module>")?.identity.kind).toBe("module");
    expect(byName.get("compute_retry_delay")?.identity.kind).toBe("function");
    expect(byName.get("RetryPolicy")?.identity.kind).toBe("class");
    expect(byName.get("RetryPolicy.describe")?.identity.kind).toBe("method");
    expect(byName.get("RECOVERY_WINDOW_MS")?.identity.kind).toBe("constant");
    expect(analysis.language).toBe("python");
  });

  it("treats functions and methods as behaviour bearing but not classes or constants", () => {
    expect(symbol(implementation, "compute_retry_delay").behaviorBearing).toBe(true);
    expect(symbol(implementation, "RetryPolicy.describe").behaviorBearing).toBe(true);
    expect(symbol(implementation, "RetryPolicy").behaviorBearing).toBe(false);
    expect(symbol(implementation, "RECOVERY_WINDOW_MS").behaviorBearing).toBe(false);
  });

  it("marks an underscore-prefixed name as not exported", () => {
    expect(symbol(implementation, "RetryPolicy.describe").exported).toBe(true);
    expect(symbol(implementation, "RetryPolicy._internal").exported).toBe(false);
  });

  it("keeps behaviour identity stable across a format-only edit", () => {
    // Indentation is semantic in Python, so a format-only edit re-indents the
    // whole block uniformly and changes only spacing inside lines.
    const reformatted = `RECOVERY_WINDOW_MS = 4500


def compute_retry_delay(failure_index: int, base_ms: int) -> int:
        """Return the approved delay."""

        if failure_index < 0:
                raise ValueError("negative")

        return min(base_ms*3**failure_index,   RECOVERY_WINDOW_MS)


class RetryPolicy:
    def describe(self) -> str:
        return "policy"

    def _internal(self) -> None:
        return None
`;
    expect(symbol(reformatted, "compute_retry_delay").contentHash).toBe(
      symbol(implementation, "compute_retry_delay").contentHash,
    );
  });

  it("changes behaviour identity when real nesting changes", () => {
    const nested = implementation.replace(
      "    return min(base_ms * 3 ** failure_index, RECOVERY_WINDOW_MS)",
      "    if base_ms > 0:\n        return min(base_ms * 3 ** failure_index, RECOVERY_WINDOW_MS)",
    );
    expect(symbol(nested, "compute_retry_delay").contentHash).not.toBe(
      symbol(implementation, "compute_retry_delay").contentHash,
    );
  });

  it("ignores comments and docstrings when deriving behaviour identity", () => {
    const annotated = implementation
      .replace('    """Return the approved delay."""', '    """Completely different prose."""')
      .replace("    return min(", "    # explain the cap\n    return min(");
    expect(symbol(annotated, "compute_retry_delay").contentHash).toBe(
      symbol(implementation, "compute_retry_delay").contentHash,
    );
  });

  it("changes behaviour identity when the body changes", () => {
    const changed = implementation.replace(
      "base_ms * 3 ** failure_index",
      "base_ms * 2 ** failure_index",
    );
    expect(symbol(changed, "compute_retry_delay").contentHash).not.toBe(
      symbol(implementation, "compute_retry_delay").contentHash,
    );
  });

  it("changes the symbol ID when the signature changes", () => {
    const changed = implementation.replace(
      "def compute_retry_delay(failure_index: int, base_ms: int) -> int:",
      "def compute_retry_delay(failure_index: int, base_ms: float) -> float:",
    );
    expect(symbol(changed, "compute_retry_delay").symbolId).not.toBe(
      symbol(implementation, "compute_retry_delay").symbolId,
    );
  });

  it("keeps the symbol ID stable when only the body changes", () => {
    const changed = implementation.replace(
      "base_ms * 3 ** failure_index",
      "base_ms * 2 ** failure_index",
    );
    expect(symbol(changed, "compute_retry_delay").symbolId).toBe(
      symbol(implementation, "compute_retry_delay").symbolId,
    );
  });

  it("reads a signature that wraps across lines", () => {
    const wrapped = `def allocate(
    total_cents: int,
    account_ids: list[str],
) -> dict[str, int]:
    return {}
`;
    expect(symbol(wrapped, "allocate").signature).toBe(
      "( total_cents: int, account_ids: list[str], ) -> dict[str, int]",
    );
  });

  it("does not expose a closure as its own symbol", () => {
    const withClosure = `def outer(value: int) -> int:
    def inner(x: int) -> int:
        return x + 1

    return inner(value)
`;
    const names = analyze(withClosure).symbols.map((record) => record.identity.qualifiedName);
    expect(names).toContain("outer");
    expect(names).not.toContain("outer.inner");
  });

  it("handles async definitions", () => {
    const asyncSource = `async def fetch(url: str) -> str:
    return url
`;
    expect(symbol(asyncSource, "fetch").identity.kind).toBe("function");
    expect(symbol(asyncSource, "fetch").behaviorBearing).toBe(true);
  });
});
