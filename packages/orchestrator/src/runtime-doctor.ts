import type { AgentRuntimePort, RuntimeCapabilitySnapshot, RuntimeProvider } from "./runtime.js";

export type RuntimeDoctorIssueCode =
  | "not_installed"
  | "not_authenticated"
  | "api_billing_requires_consent"
  | "unknown_billing_path"
  | "missing_structured_output"
  | "missing_workspace_sandbox";

export interface RuntimeDoctorIssue {
  readonly provider: RuntimeProvider;
  readonly code: RuntimeDoctorIssueCode;
  readonly message: string;
}

export interface RuntimeDoctorReport {
  readonly capabilities: readonly RuntimeCapabilitySnapshot[];
  readonly issues: readonly RuntimeDoctorIssue[];
  readonly eligibleProviders: readonly RuntimeProvider[];
}

export class RuntimeDoctor {
  async inspect(runtimes: readonly AgentRuntimePort[]): Promise<RuntimeDoctorReport> {
    const capabilities = await Promise.all(
      [...runtimes]
        .sort((left, right) => left.provider.localeCompare(right.provider))
        .map((runtime) => runtime.inspectCapabilities()),
    );
    const issues = capabilities.flatMap(runtimeIssues);
    const eligibleProviders = capabilities
      .filter(
        (capability) =>
          capability.installed &&
          capability.authenticated &&
          capability.billingPath === "subscription" &&
          capability.supports.structuredOutput &&
          capability.supports.workspaceSandbox,
      )
      .map((capability) => capability.provider);
    return { capabilities, issues, eligibleProviders };
  }
}

function runtimeIssues(capability: RuntimeCapabilitySnapshot): readonly RuntimeDoctorIssue[] {
  const issues: RuntimeDoctorIssue[] = [];
  const add = (code: RuntimeDoctorIssueCode, message: string): void => {
    issues.push({ provider: capability.provider, code, message });
  };
  if (!capability.installed) add("not_installed", capability.diagnostic ?? "CLI is not installed");
  else if (!capability.authenticated) {
    add("not_authenticated", capability.diagnostic ?? "CLI is not authenticated");
  }
  if (capability.billingPath === "api") {
    add(
      "api_billing_requires_consent",
      "Runtime selected usage-billed API credentials instead of subscription auth",
    );
  } else if (capability.billingPath === "unknown") {
    add("unknown_billing_path", "Runtime billing path could not be proven");
  }
  if (!capability.supports.structuredOutput) {
    add("missing_structured_output", "Runtime cannot provide structured events");
  }
  if (!capability.supports.workspaceSandbox) {
    add("missing_workspace_sandbox", "Runtime cannot prove workspace write isolation");
  }
  return issues;
}
