import { randomUUID } from "node:crypto";
import { chmod, mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { CodeResourceSnapshotAdapter, TypeScriptCodeProvider } from "@kontext-brain/code";
import { TaskContextWorkflow } from "@kontext-brain/context";
import {
  InMemoryKnowledgeGraphRepository,
  InMemoryResourceContentStore,
  SyncResourceUseCase,
} from "@kontext-brain/core";
import { ExternalIdNormativeResourceReader, SymbolGovernanceResolver } from "@kontext-brain/loader";
import { FileTaskContextRepository, assembleCurrentTaskContextState } from "@kontext-brain/local";
import type { GovernanceScope, NormativeRevision, TaskContract } from "@kontext-brain/spec";
import { runWorkspaceCommand } from "../workspace.js";
import type { DeepSweContextCorpus } from "./contracts.js";
import { sha256 } from "./corpus.js";
import { exportDeepSweCorpus } from "./export-corpus.js";

export interface DeepSwePilotSource {
  readonly evidenceAlias: string;
  readonly relativePath: string;
  readonly title: string;
  readonly startLine: number;
  readonly endLine: number;
  readonly ontologyNodeIds: readonly string[];
}

interface DeepSwePilotNormativeBase {
  readonly recordId: string;
  readonly revisionId: string;
  readonly evidenceAliases: readonly string[];
  readonly ontologyNodeIds: readonly string[];
}

export type DeepSwePilotNormativeRecord = DeepSwePilotNormativeBase &
  (
    | { readonly kind: "decision"; readonly statement: string }
    | {
        readonly kind: "domain_term";
        readonly term: string;
        readonly definition: string;
        readonly avoid?: readonly string[];
      }
    | {
        readonly kind: "invariant";
        readonly statement: string;
        readonly verifiers: readonly {
          readonly kind: "test" | "typecheck";
          readonly ref: string;
        }[];
      }
  );

export interface DeepSwePilotTarget {
  readonly workItemId: string;
  readonly plannedSymbolId: string;
  readonly relativePath: string;
  readonly qualifiedName: string;
  readonly kind: "class" | "function" | "method";
  readonly responsibility: string;
  readonly ontologyNodeIds: readonly string[];
  readonly allowedPaths: readonly string[];
}

export interface DeepSwePilotSpec {
  readonly taskId: string;
  readonly organizationId: string;
  readonly codebaseId: string;
  readonly repository: string;
  readonly repositoryUrl: string;
  readonly baseCommit: string;
  readonly observedAt: string;
  readonly snapshotAt: string;
  readonly sourceIntegrity: readonly {
    readonly relativePath: string;
    readonly sha256: string;
  }[];
  readonly sources: readonly DeepSwePilotSource[];
  readonly normativeRecords: readonly DeepSwePilotNormativeRecord[];
  readonly targets: readonly DeepSwePilotTarget[];
  readonly contract: Omit<TaskContract, "taskId" | "targets">;
}

export interface DeepSwePilotCorpusResult {
  readonly corpus: DeepSweContextCorpus;
  readonly codeResources: number;
  readonly codeSymbols: number;
  readonly behaviorBearingSymbols: number;
  readonly evidenceResources: number;
  readonly governingRecordIds: readonly string[];
}

interface MaterializedEvidence {
  readonly alias: string;
  readonly evidenceId: string;
  readonly text: string;
  readonly sourceSpan: string;
  readonly resourceId: string;
  readonly chunkId: string;
  readonly resourceTitle: string;
  readonly source: {
    readonly connectorId: string;
    readonly externalId: string;
    readonly type: string;
  };
  readonly observedAt: string;
  readonly contentHash: string;
  readonly ontologyNodeIds: readonly string[];
}

/**
 * Builds a scored corpus only from a pinned public base checkout. The source
 * tree is indexed through the production Resource/Chunk/Evidence graph, then
 * ontology hops derive the Planned Symbol governance links used by Kontext.
 */
export async function buildDeepSwePilotCorpus(input: {
  readonly spec: DeepSwePilotSpec;
  readonly checkoutPath: string;
  readonly dataDirectory: string;
  readonly outputPath: string;
  readonly runtimeProvider: string;
  readonly generatorRevision: string;
}): Promise<DeepSwePilotCorpusResult> {
  validateSpec(input.spec);
  await verifyCheckout(input.checkoutPath, input.spec);
  const repository = new InMemoryKnowledgeGraphRepository();
  const contentStore = new InMemoryResourceContentStore();
  const clock = { now: () => new Date(input.spec.observedAt) };
  const sync = new SyncResourceUseCase(repository, contentStore, clock);

  const sourceTexts = await verifiedSourceTexts(input.checkoutPath, input.spec);
  const fileNodeIds = targetNodesByFile(input.spec.targets);
  const code = await ingestTypeScriptCode({
    spec: input.spec,
    checkoutPath: input.checkoutPath,
    sourceTexts,
    fileNodeIds,
    sync,
  });
  const evidence = await ingestEvidenceSources({
    spec: input.spec,
    sourceTexts,
    repository,
    sync,
  });
  const evidenceByAlias = new Map(evidence.map((item) => [item.alias, item]));
  const revisions = input.spec.normativeRecords.map((record) =>
    materializeRevision(record, input.spec, input.runtimeProvider, evidenceByAlias),
  );
  await ingestNormativeResources(input.spec, revisions, sync);

  const revisionByRecordId = new Map(
    revisions.map((revision) => [revision.recordId, revision.revisionId]),
  );
  const resolver = new SymbolGovernanceResolver(
    repository,
    new ExternalIdNormativeResourceReader((recordId) => revisionByRecordId.get(recordId)),
  );
  const resolutions = await Promise.all(
    input.spec.targets.map((target) =>
      resolver.resolve({
        organizationId: input.spec.organizationId,
        codebaseId: input.spec.codebaseId,
        relativePath: target.relativePath,
        plannedSymbolId: target.plannedSymbolId,
      }),
    ),
  );
  assertOntologyResolution(input.spec, resolutions);

  const scope: GovernanceScope = { kind: "codebase", codebaseId: input.spec.codebaseId };
  const current = assembleCurrentTaskContextState({
    taskId: input.spec.taskId,
    organizationId: input.spec.organizationId,
    codeRevision: input.spec.baseCommit,
    baseScopes: [scope],
    localManifest: {
      schemaVersion: 1,
      organizationId: input.spec.organizationId,
      revisions,
      activations: revisions.map((revision) => ({
        organizationId: input.spec.organizationId,
        kind: revision.kind,
        recordId: revision.recordId,
        revisionId: revision.revisionId,
        scope,
        state: "accepted_local" as const,
        acceptedBy: "benchmark-curator:base-source",
        acceptedAt: input.spec.snapshotAt,
      })),
    },
    evidence: evidence.map((item) => ({
      evidenceId: item.evidenceId,
      text: item.text,
      sourceSpan: item.sourceSpan,
      availability: "current" as const,
      allowedRuntimeProviders: [input.runtimeProvider],
      provenance: {
        resourceId: item.resourceId,
        chunkId: item.chunkId,
        resourceTitle: item.resourceTitle,
        source: item.source,
        observedAt: item.observedAt,
        contentHash: item.contentHash,
        ontologyNodeIds: item.ontologyNodeIds,
      },
    })),
    logicPlans: input.spec.targets.map((target) => ({
      workItemId: target.workItemId,
      plannedSymbolIds: [target.plannedSymbolId],
      plannedSymbols: [
        {
          plannedSymbolId: target.plannedSymbolId,
          taskId: input.spec.taskId,
          intendedIdentity: {
            codebaseId: input.spec.codebaseId,
            relativePath: target.relativePath,
            language: "typescript" as const,
            kind: target.kind,
            qualifiedName: target.qualifiedName,
          },
          responsibility: target.responsibility,
        },
      ],
      allowedPaths: target.allowedPaths,
      requiredVerifiers: [
        { kind: "typecheck" as const, ref: "npm test -- --runInBand" },
        { kind: "test" as const, ref: "DeepSWE separate verifier" },
      ],
    })),
    governanceLinks: resolutions.flatMap((resolution) =>
      resolution.records.map((record) => ({
        plannedSymbolId: resolution.plannedSymbolId,
        recordId: record.recordId,
        revisionId: record.revisionId,
        origin: record.origin,
      })),
    ),
  });

  const sidecar = new FileTaskContextRepository(input.dataDirectory);
  await sidecar.publishCurrent(input.spec.taskId, current);
  const workflow = new TaskContextWorkflow(sidecar, sidecar);
  const prepared = await workflow.prepareTask({
    contract: {
      ...input.spec.contract,
      taskId: input.spec.taskId,
      targets: input.spec.targets.map((target) => target.plannedSymbolId),
    },
    createdAt: input.spec.snapshotAt,
  });
  const corpus = exportDeepSweCorpus({
    taskId: input.spec.taskId,
    organizationId: input.spec.organizationId,
    runtimeProvider: input.runtimeProvider,
    generatorRevision: input.generatorRevision,
    prepared,
    current,
  });
  await atomicPrivateWrite(input.outputPath, `${JSON.stringify(corpus, null, 2)}\n`);

  return {
    corpus,
    ...code,
    evidenceResources: new Set(evidence.map((item) => item.resourceId)).size,
    governingRecordIds: [
      ...new Set(resolutions.flatMap((item) => item.records.map((record) => record.recordId))),
    ].sort(),
  };
}

async function ingestTypeScriptCode(input: {
  readonly spec: DeepSwePilotSpec;
  readonly checkoutPath: string;
  readonly sourceTexts: ReadonlyMap<string, string>;
  readonly fileNodeIds: ReadonlyMap<string, readonly string[]>;
  readonly sync: SyncResourceUseCase;
}): Promise<{
  readonly codeResources: number;
  readonly codeSymbols: number;
  readonly behaviorBearingSymbols: number;
}> {
  const provider = new TypeScriptCodeProvider();
  const adapter = new CodeResourceSnapshotAdapter();
  let codeSymbols = 0;
  let behaviorBearingSymbols = 0;
  const files = [...input.sourceTexts]
    .filter(([relativePath]) => relativePath.endsWith(".ts"))
    .map(([relativePath, content]) => ({ path: relativePath, content }))
    .sort((left, right) => left.path.localeCompare(right.path));
  for (const file of files) {
    const analysis = provider.analyze({
      codebaseId: input.spec.codebaseId,
      targetPath: file.path,
      files: [file],
    });
    codeSymbols += analysis.symbols.length;
    behaviorBearingSymbols += analysis.symbols.filter((symbol) => symbol.behaviorBearing).length;
    const normalized = adapter.normalize({
      analysis,
      organizationId: input.spec.organizationId,
      acl: { organizationWide: true },
      ontologyNodeIds: input.fileNodeIds.get(file.path) ?? [`code-module:${file.path}`],
    });
    await input.sync.execute({
      ...normalized,
      source: { ...normalized.source, type: "code-module" },
    });
  }
  return { codeResources: files.length, codeSymbols, behaviorBearingSymbols };
}

async function ingestEvidenceSources(input: {
  readonly spec: DeepSwePilotSpec;
  readonly sourceTexts: ReadonlyMap<string, string>;
  readonly repository: InMemoryKnowledgeGraphRepository;
  readonly sync: SyncResourceUseCase;
}): Promise<readonly MaterializedEvidence[]> {
  const byFile = groupSourcesByFile(input.spec.sources);
  const output: MaterializedEvidence[] = [];
  for (const [relativePath, sources] of [...byFile].sort(([left], [right]) =>
    left.localeCompare(right),
  )) {
    const body = required(input.sourceTexts, relativePath);
    const sourceIdentity = {
      connectorId: "github-repository",
      externalId: `${input.spec.repositoryUrl.replace(/\.git$/, "")}/blob/${input.spec.baseCommit}/${relativePath}`,
      type: relativePath.toLowerCase().endsWith(".md") ? "markdown" : "code",
    };
    const chunks = sources.map((source, position) => {
      const text = lineRange(body, source.startLine, source.endLine, relativePath);
      return {
        id: source.evidenceAlias,
        contentHash: sha256(text),
        text,
        position,
        ontologyNodeIds: uniqueSorted(source.ontologyNodeIds),
      };
    });
    const result = await input.sync.execute({
      organizationId: input.spec.organizationId,
      source: sourceIdentity,
      title: sources[0]?.title ?? relativePath,
      contentHash: sha256(body),
      body,
      acl: { organizationWide: true },
      ontologyNodeIds: uniqueSorted(sources.flatMap((source) => source.ontologyNodeIds)),
      chunks,
    });
    const [resource, storedChunks, evidenceRecords] = await Promise.all([
      input.repository.getResource(input.spec.organizationId, result.resourceId),
      input.repository.listChunks(input.spec.organizationId, result.resourceId),
      input.repository.listEvidenceForResource(input.spec.organizationId, result.resourceId),
    ]);
    if (!resource) throw new Error(`Missing ingested Resource for ${relativePath}`);
    for (const source of sources) {
      const chunk = storedChunks.find((item) => item.sourceChunkId === source.evidenceAlias);
      if (!chunk) throw new Error(`Missing ingested Chunk for ${source.evidenceAlias}`);
      const evidence = evidenceRecords.find(
        (item) => item.chunkId === chunk.chunkId && item.factKey === undefined,
      );
      if (!evidence?.observedAt) {
        throw new Error(`Missing provenance Evidence for ${source.evidenceAlias}`);
      }
      output.push({
        alias: source.evidenceAlias,
        evidenceId: evidence.evidenceId,
        text: lineRange(body, source.startLine, source.endLine, relativePath),
        sourceSpan: `${relativePath}:${source.startLine}-${source.endLine}`,
        resourceId: resource.resourceId,
        chunkId: chunk.chunkId,
        resourceTitle: source.title,
        source: resource.source,
        observedAt: evidence.observedAt,
        contentHash: chunk.contentHash,
        ontologyNodeIds: uniqueSorted(chunk.ontologyNodeIds),
      });
    }
  }
  return output.sort((left, right) => left.evidenceId.localeCompare(right.evidenceId));
}

async function ingestNormativeResources(
  spec: DeepSwePilotSpec,
  revisions: readonly NormativeRevision[],
  sync: SyncResourceUseCase,
): Promise<void> {
  const nodesByRecord = new Map(
    spec.normativeRecords.map((record) => [record.recordId, record.ontologyNodeIds]),
  );
  for (const revision of revisions) {
    const body = normativeText(revision);
    await sync.execute({
      organizationId: spec.organizationId,
      source: { connectorId: "normative", externalId: revision.recordId, type: revision.kind },
      title: revision.recordId,
      contentHash: sha256(`${revision.revisionId}\n${body}`),
      body,
      acl: { organizationWide: true },
      ontologyNodeIds: uniqueSorted(nodesByRecord.get(revision.recordId) ?? []),
      chunks: [
        {
          id: revision.revisionId,
          contentHash: sha256(body),
          text: body,
          position: 0,
        },
      ],
    });
  }
}

function materializeRevision(
  record: DeepSwePilotNormativeRecord,
  spec: DeepSwePilotSpec,
  runtimeProvider: string,
  evidenceByAlias: ReadonlyMap<string, MaterializedEvidence>,
): NormativeRevision {
  const common = {
    organizationId: spec.organizationId,
    recordId: record.recordId,
    revisionId: record.revisionId,
    scope: { kind: "codebase" as const, codebaseId: spec.codebaseId },
    evidence: record.evidenceAliases.map((alias) => {
      const evidence = required(evidenceByAlias, alias);
      return { evidenceId: evidence.evidenceId, sourceSpan: evidence.sourceSpan };
    }),
    egress: { dataClassification: "public" as const, allowedRuntimeProviders: [runtimeProvider] },
    authoredBy: `benchmark-curator:source-derived:${spec.repository}`,
    authoredAt: spec.observedAt,
  };
  switch (record.kind) {
    case "decision":
      return { ...common, kind: record.kind, statement: record.statement };
    case "domain_term":
      return {
        ...common,
        kind: record.kind,
        term: record.term,
        definition: record.definition,
        ...(record.avoid ? { avoid: record.avoid } : {}),
      };
    case "invariant":
      return {
        ...common,
        kind: record.kind,
        statement: record.statement,
        verifiers: record.verifiers,
      };
  }
}

async function verifiedSourceTexts(
  checkoutPath: string,
  spec: DeepSwePilotSpec,
): Promise<ReadonlyMap<string, string>> {
  const output = new Map<string, string>();
  for (const expected of spec.sourceIntegrity) {
    const filePath = safeCheckoutPath(checkoutPath, expected.relativePath);
    const value = await readFile(filePath, "utf8");
    if (sha256(value) !== expected.sha256) {
      throw new Error(`Source integrity mismatch: ${expected.relativePath}`);
    }
    output.set(expected.relativePath, value);
  }
  for (const source of spec.sources) required(output, source.relativePath);
  for (const target of spec.targets) required(output, target.relativePath);
  return output;
}

async function verifyCheckout(checkoutPath: string, spec: DeepSwePilotSpec): Promise<void> {
  const checkout = path.resolve(checkoutPath);
  if (!(await stat(checkout)).isDirectory()) throw new Error("Pilot checkout must be a directory");
  const [revision, status] = await Promise.all([
    runWorkspaceCommand(checkout, "git", ["rev-parse", "HEAD"]),
    runWorkspaceCommand(checkout, "git", ["status", "--porcelain", "--untracked-files=no"]),
  ]);
  if (revision.exitCode !== 0 || revision.stdout.trim() !== spec.baseCommit) {
    throw new Error(`Pilot checkout revision must be ${spec.baseCommit}`);
  }
  if (status.exitCode !== 0 || status.stdout.trim()) {
    throw new Error("Pilot checkout must have no tracked modifications");
  }
}

function assertOntologyResolution(
  spec: DeepSwePilotSpec,
  resolutions: readonly {
    readonly plannedSymbolId: string;
    readonly records: readonly { readonly recordId: string }[];
  }[],
): void {
  for (const target of spec.targets) {
    const actual = resolutions
      .find((resolution) => resolution.plannedSymbolId === target.plannedSymbolId)
      ?.records.map((record) => record.recordId)
      .sort();
    const targetNodes = new Set(target.ontologyNodeIds);
    const expected = spec.normativeRecords
      .filter((record) => record.ontologyNodeIds.some((nodeId) => targetNodes.has(nodeId)))
      .map((record) => record.recordId)
      .sort();
    if (JSON.stringify(actual ?? []) !== JSON.stringify(expected)) {
      throw new Error(
        `Ontology resolution mismatch for ${target.plannedSymbolId}: expected ${expected.join(", ")}; received ${(actual ?? []).join(", ")}`,
      );
    }
  }
}

function targetNodesByFile(
  targets: readonly DeepSwePilotTarget[],
): ReadonlyMap<string, readonly string[]> {
  const output = new Map<string, string[]>();
  for (const target of targets) {
    output.set(
      target.relativePath,
      uniqueSorted([...(output.get(target.relativePath) ?? []), ...target.ontologyNodeIds]),
    );
  }
  return output;
}

function groupSourcesByFile(
  sources: readonly DeepSwePilotSource[],
): ReadonlyMap<string, readonly DeepSwePilotSource[]> {
  const output = new Map<string, DeepSwePilotSource[]>();
  for (const source of sources) {
    const values = output.get(source.relativePath) ?? [];
    values.push(source);
    output.set(source.relativePath, values);
  }
  return output;
}

function lineRange(source: string, startLine: number, endLine: number, label: string): string {
  const lines = source.split(/\r?\n/);
  if (startLine < 1 || endLine < startLine || endLine > lines.length) {
    throw new Error(`Invalid source line range for ${label}: ${startLine}-${endLine}`);
  }
  return lines.slice(startLine - 1, endLine).join("\n");
}

function safeCheckoutPath(checkoutPath: string, relativePath: string): string {
  if (path.isAbsolute(relativePath) || relativePath.split(/[\\/]+/).includes("..")) {
    throw new Error(`Source path escapes checkout: ${relativePath}`);
  }
  const root = path.resolve(checkoutPath);
  const resolved = path.resolve(root, relativePath);
  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) {
    throw new Error(`Source path escapes checkout: ${relativePath}`);
  }
  return resolved;
}

function normativeText(revision: NormativeRevision): string {
  if (revision.kind === "domain_term") {
    return `${revision.term}: ${revision.definition}${revision.avoid?.length ? ` Avoid: ${revision.avoid.join(", ")}.` : ""}`;
  }
  return revision.statement;
}

function validateSpec(spec: DeepSwePilotSpec): void {
  const aliases = new Set<string>();
  for (const source of spec.sources) {
    if (aliases.has(source.evidenceAlias)) {
      throw new Error(`Duplicate pilot Evidence alias: ${source.evidenceAlias}`);
    }
    aliases.add(source.evidenceAlias);
  }
  for (const record of spec.normativeRecords) {
    if (record.evidenceAliases.length === 0) {
      throw new Error(`Pilot normative record has no Evidence: ${record.recordId}`);
    }
    for (const alias of record.evidenceAliases) {
      if (!aliases.has(alias)) throw new Error(`Unknown pilot Evidence alias: ${alias}`);
    }
  }
  if (Date.parse(spec.observedAt) >= Date.parse(spec.snapshotAt)) {
    throw new Error("Pilot snapshot must be created after its base-source observation");
  }
}

async function atomicPrivateWrite(filePath: string, contents: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true, mode: 0o700 });
  const temporaryPath = `${filePath}.${process.pid}.${randomUUID()}.tmp`;
  await writeFile(temporaryPath, contents, { encoding: "utf8", mode: 0o600 });
  await chmod(temporaryPath, 0o600);
  await rename(temporaryPath, filePath);
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) => left.localeCompare(right));
}

function required<K, V>(map: ReadonlyMap<K, V>, key: K): V {
  const value = map.get(key);
  if (value === undefined) throw new Error(`Missing pilot value: ${String(key)}`);
  return value;
}
