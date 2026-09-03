import type { DeepSwePilotSpec } from "./pilot-corpus.js";

const resolverBuilder = "domain:awilix:resolver-builder";
const dependencyGraph = "domain:awilix:dependency-graph";
const containerLifecycle = "domain:awilix:container-lifecycle";
const scopeOwnership = "domain:awilix:scope-ownership";
const errorSemantics = "domain:awilix:error-semantics";
const publicApi = "domain:awilix:public-api";

/**
 * Preregistered only from Awilix's public base revision. DeepSWE task tests,
 * verifier files, solution patches, and prior trajectories are not sources.
 */
export const awilixAsyncInitializationPilot: DeepSwePilotSpec = {
  taskId: "awilix-async-container-initialization",
  organizationId: "organization:deepswe-pilot",
  codebaseId: "codebase:github:jeffijoe/awilix@82ac179c",
  repository: "jeffijoe/awilix",
  repositoryUrl: "https://github.com/jeffijoe/awilix.git",
  baseCommit: "82ac179c1de4c216c4e333093044fac643303f0c",
  observedAt: "2026-09-03T06:08:28.000Z",
  snapshotAt: "2026-09-03T06:08:29.000Z",
  sourceIntegrity: [
    {
      relativePath: "README.md",
      sha256: "7f0147b27be8e6cb7457dee875ada6a33a5d73703c237516777d2bc9862e7dba",
    },
    {
      relativePath: "src/awilix.ts",
      sha256: "ffacb00c764dc87a05ab26920e9a98afae4e73071203042c1c1f7d02c6242c81",
    },
    {
      relativePath: "src/container.ts",
      sha256: "808fed5b2736cf87e09676020026888e054b1acd6a8eee3641db85fed8ba9731",
    },
    {
      relativePath: "src/errors.ts",
      sha256: "31d11db71364ea353efa7022860152940813ccaf048561972e86b57c038c1c78",
    },
    {
      relativePath: "src/resolvers.ts",
      sha256: "f76d382a9657dbd60cd57e97465dd9e59f645dccc74c4893ded260e1f9827b85",
    },
  ],
  sources: [
    {
      evidenceAlias: "readme-lifetimes",
      relativePath: "README.md",
      title: "Awilix README",
      startLine: 195,
      endLine: 210,
      ontologyNodeIds: [scopeOwnership],
    },
    {
      evidenceAlias: "readme-scopes",
      relativePath: "README.md",
      title: "Awilix README",
      startLine: 1189,
      endLine: 1215,
      ontologyNodeIds: [scopeOwnership],
    },
    {
      evidenceAlias: "readme-dispose",
      relativePath: "README.md",
      title: "Awilix README",
      startLine: 1350,
      endLine: 1357,
      ontologyNodeIds: [containerLifecycle],
    },
    {
      evidenceAlias: "container-resolution",
      relativePath: "src/container.ts",
      title: "Awilix container implementation",
      startLine: 479,
      endLine: 607,
      ontologyNodeIds: [dependencyGraph, errorSemantics, scopeOwnership],
    },
    {
      evidenceAlias: "container-dispose",
      relativePath: "src/container.ts",
      title: "Awilix container implementation",
      startLine: 705,
      endLine: 721,
      ontologyNodeIds: [containerLifecycle],
    },
    {
      evidenceAlias: "resolver-contracts",
      relativePath: "src/resolvers.ts",
      title: "Awilix resolver implementation",
      startLine: 28,
      endLine: 70,
      ontologyNodeIds: [resolverBuilder, containerLifecycle],
    },
    {
      evidenceAlias: "resolver-fluent-builders",
      relativePath: "src/resolvers.ts",
      title: "Awilix resolver implementation",
      startLine: 245,
      endLine: 301,
      ontologyNodeIds: [resolverBuilder],
    },
    {
      evidenceAlias: "resolver-dependency-parsing",
      relativePath: "src/resolvers.ts",
      title: "Awilix resolver implementation",
      startLine: 451,
      endLine: 499,
      ontologyNodeIds: [dependencyGraph, resolverBuilder],
    },
    {
      evidenceAlias: "resolution-error-contract",
      relativePath: "src/errors.ts",
      title: "Awilix error implementation",
      startLine: 121,
      endLine: 153,
      ontologyNodeIds: [errorSemantics, publicApi],
    },
    {
      evidenceAlias: "public-api-barrel",
      relativePath: "src/awilix.ts",
      title: "Awilix public API barrel",
      startLine: 1,
      endLine: 43,
      ontologyNodeIds: [publicApi],
    },
  ],
  normativeRecords: [
    {
      kind: "domain_term",
      recordId: "domain-term:awilix-scope",
      revisionId: "domain-term:awilix-scope@82ac179c",
      term: "Scope",
      definition:
        "A child container with its own cache for SCOPED registrations; SINGLETON registrations remain cached in the root container.",
      avoid: ["child singleton cache", "global scoped cache"],
      evidenceAliases: ["readme-lifetimes", "readme-scopes", "container-resolution"],
      ontologyNodeIds: [scopeOwnership],
    },
    {
      kind: "invariant",
      recordId: "invariant:awilix-cache-ownership",
      revisionId: "invariant:awilix-cache-ownership@82ac179c",
      statement:
        "SCOPED values are resolved and cached by the current container, while SINGLETON values are resolved and cached by the root container.",
      evidenceAliases: ["readme-lifetimes", "readme-scopes", "container-resolution"],
      ontologyNodeIds: [scopeOwnership],
      verifiers: [
        { kind: "test", ref: "src/__tests__/container.test.ts" },
        { kind: "typecheck", ref: "npm run check" },
      ],
    },
    {
      kind: "invariant",
      recordId: "invariant:awilix-disposal-cache",
      revisionId: "invariant:awilix-disposal-cache@82ac179c",
      statement:
        "Container disposal clears its own cache and awaits disposer functions only for cached SCOPED or SINGLETON resolutions.",
      evidenceAliases: ["readme-dispose", "container-dispose", "resolver-contracts"],
      ontologyNodeIds: [containerLifecycle],
      verifiers: [
        { kind: "test", ref: "src/__tests__/container.disposing.test.ts" },
        { kind: "typecheck", ref: "npm run check" },
      ],
    },
    {
      kind: "decision",
      recordId: "decision:awilix-resolver-fluent-extension",
      revisionId: "decision:awilix-resolver-fluent-extension@82ac179c",
      statement:
        "Extend resolver capabilities through the existing chainable builder seam: each fluent method returns a copied resolver object while preserving the configured options.",
      evidenceAliases: [
        "resolver-contracts",
        "resolver-fluent-builders",
        "resolver-dependency-parsing",
      ],
      ontologyNodeIds: [resolverBuilder],
    },
    {
      kind: "invariant",
      recordId: "invariant:awilix-dependency-metadata",
      revisionId: "invariant:awilix-dependency-metadata@82ac179c",
      statement:
        "A function or class resolver parses its dependency parameter list once when its resolve function is generated, including in PROXY mode; CLASSIC mode resolves those stored names with optionality preserved.",
      evidenceAliases: ["resolver-dependency-parsing", "container-resolution"],
      ontologyNodeIds: [dependencyGraph, resolverBuilder],
      verifiers: [
        { kind: "test", ref: "src/__tests__/resolvers.test.ts" },
        { kind: "typecheck", ref: "npm run check" },
      ],
    },
    {
      kind: "invariant",
      recordId: "invariant:awilix-circular-resolution-error",
      revisionId: "invariant:awilix-circular-resolution-error@82ac179c",
      statement:
        "A circular dependency is reported as AwilixResolutionError with a resolution path, and a failed resolution resets the shared resolution stack.",
      evidenceAliases: ["container-resolution", "resolution-error-contract"],
      ontologyNodeIds: [dependencyGraph, errorSemantics],
      verifiers: [
        { kind: "test", ref: "src/__tests__/container.test.ts" },
        { kind: "typecheck", ref: "npm run check" },
      ],
    },
    {
      kind: "decision",
      recordId: "decision:awilix-public-api-barrel",
      revisionId: "decision:awilix-public-api-barrel@82ac179c",
      statement:
        "Public container methods, Awilix error classes, and resolver types/builders are exposed through the src/awilix.ts package barrel.",
      evidenceAliases: ["public-api-barrel", "resolution-error-contract", "resolver-contracts"],
      ontologyNodeIds: [publicApi],
    },
  ],
  targets: [
    {
      workItemId: "work-item:awilix-resolver-initializer",
      plannedSymbolId: "planned-symbol:awilix:createBuildResolver",
      relativePath: "src/resolvers.ts",
      qualifiedName: "createBuildResolver",
      kind: "function",
      responsibility:
        "Add the initializer fluent API and retain dependency metadata needed for initialization planning.",
      ontologyNodeIds: [resolverBuilder, dependencyGraph],
      allowedPaths: ["src/resolvers.ts"],
    },
    {
      workItemId: "work-item:awilix-container-initialize",
      plannedSymbolId: "planned-symbol:awilix:createContainerInternal",
      relativePath: "src/container.ts",
      qualifiedName: "createContainerInternal",
      kind: "function",
      responsibility:
        "Implement dependency-level initialization, scope-aware state, concurrency, idempotence, and rollback.",
      ontologyNodeIds: [containerLifecycle, scopeOwnership, dependencyGraph, errorSemantics],
      allowedPaths: ["src/container.ts"],
    },
    {
      workItemId: "work-item:awilix-initialization-errors",
      plannedSymbolId: "planned-symbol:awilix:AwilixResolutionError.constructor",
      relativePath: "src/errors.ts",
      qualifiedName: "AwilixResolutionError.constructor",
      kind: "method",
      responsibility:
        "Preserve Awilix error identity and cause semantics for initialization failures.",
      ontologyNodeIds: [errorSemantics, publicApi],
      allowedPaths: ["src/errors.ts"],
    },
    {
      workItemId: "work-item:awilix-public-exports",
      plannedSymbolId: "planned-symbol:awilix:public-api",
      relativePath: "src/awilix.ts",
      qualifiedName: "createContainer",
      kind: "function",
      responsibility:
        "Expose the new public initialization API and error types from the package barrel.",
      ontologyNodeIds: [publicApi],
      allowedPaths: ["src/awilix.ts"],
    },
  ],
  contract: {
    intent: "Implement dependency-aware asynchronous initialization in Awilix.",
    acceptance: [
      {
        criterionId: "acceptance:deepswe-verifier",
        statement:
          "The DeepSWE separate verifier accepts initialization ordering, concurrency, scope, rollback, error, and retry behavior without regressions.",
        verifier: { kind: "test", ref: "DeepSWE separate verifier" },
      },
    ],
    nonGoals: [
      "Reading or changing DeepSWE solution artifacts",
      "Reading or changing DeepSWE verifier artifacts",
      "Changing unrelated Awilix behavior",
    ],
    risk: "high",
  },
};
