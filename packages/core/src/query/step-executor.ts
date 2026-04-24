import type {
  DocumentContent,
  MetaDocument,
  PipelineStep,
} from "../graph/layered-models.js";
import { DepthType } from "../graph/layered-models.js";
import type { OntologyNode } from "../graph/ontology-node.js";
import { BM25BodyExtractor, type ContentFetcherRegistry } from "./content-fetcher.js";
import type { MetaDocumentSelector } from "./content-fetcher.js";
import type { MetaIndexStore } from "./meta-index-store.js";
import type { VectorStore } from "./vector-store.js";

export interface StepContext {
  readonly node: OntologyNode;
  readonly query: string;
  readonly accumulatedDocs: readonly MetaDocument[];
  readonly metaIndexStore: MetaIndexStore;
  readonly metaSelector: MetaDocumentSelector;
  readonly fetcherRegistry: ContentFetcherRegistry;
  readonly vectorStore?: VectorStore | null;
}

export interface StepResult {
  readonly contextSection: string;
  readonly selectedDocs: readonly MetaDocument[];
  readonly fetchedContents: readonly DocumentContent[];
}

const EMPTY_RESULT: StepResult = {
  contextSection: "",
  selectedDocs: [],
  fetchedContents: [],
};

export interface StepExecutor {
  readonly supportedType: DepthType;
  execute(ctx: StepContext, step: PipelineStep): Promise<StepResult>;
}

// ── ONTOLOGY ──────────────────────────────────────────────────

export class OntologyStepExecutor implements StepExecutor {
  readonly supportedType = DepthType.ONTOLOGY;

  async execute(ctx: StepContext): Promise<StepResult> {
    const refData = ctx.node.refBlock ? await ctx.node.refBlock() : null;
    const dataStr = refData ? String(refData) : "";
    const section = `## [Ontology: ${ctx.node.id}] — ${ctx.node.description}${dataStr ? `\n${dataStr}` : ""}`;
    return { contextSection: section, selectedDocs: [], fetchedContents: [] };
  }
}

// ── META ──────────────────────────────────────────────────────

export class MetaStepExecutor implements StepExecutor {
  readonly supportedType = DepthType.META;

  async execute(ctx: StepContext, step: PipelineStep): Promise<StepResult> {
    const candidates = await ctx.metaIndexStore.search(
      ctx.node.id,
      ctx.query,
      step.maxSelect * 3,
    );
    if (candidates.length === 0) return EMPTY_RESULT;

    const selected = step.fetchFull
      ? candidates.slice(0, step.maxSelect)
      : await ctx.metaSelector.select(ctx.query, candidates, step.maxSelect);

    const section = `## [${ctx.node.id}] Candidate documents\n${selected
      .map((d) => `- [${d.source}] ${d.title}`)
      .join("\n")}`;
    return { contextSection: section, selectedDocs: selected, fetchedContents: [] };
  }
}

// ── VECTOR ────────────────────────────────────────────────────

export class VectorStepExecutor implements StepExecutor {
  readonly supportedType = DepthType.VECTOR;

  async execute(ctx: StepContext, step: PipelineStep): Promise<StepResult> {
    const vs = ctx.vectorStore;
    if (!vs) return EMPTY_RESULT;

    const keys = await vs.similaritySearchWithPrefix(
      ctx.query,
      `content:${ctx.node.id}:`,
      step.maxSelect,
      step.threshold,
    );
    if (keys.length === 0) return EMPTY_RESULT;

    // Resolve keys -> MetaDocuments from meta index
    const docs = await ctx.metaIndexStore.search(ctx.node.id, ctx.query, step.maxSelect * 3);
    const byId = new Map(docs.map((d) => [d.id, d]));
    const selected: MetaDocument[] = [];
    for (const k of keys) {
      const doc = byId.get(k);
      if (doc) selected.push(doc);
    }
    const section = `## [${ctx.node.id}] Vector-selected\n${selected
      .map((d) => `- [${d.source}] ${d.title}`)
      .join("\n")}`;
    return { contextSection: section, selectedDocs: selected, fetchedContents: [] };
  }
}

// ── CONTENT ───────────────────────────────────────────────────

export class ContentStepExecutor implements StepExecutor {
  readonly supportedType = DepthType.CONTENT;

  async execute(ctx: StepContext, step: PipelineStep): Promise<StepResult> {
    // Prefer docs that came from the current node (avoids re-fetching across
    // depths and keeps context topically aligned). Fall back to all
    // accumulated docs if none were tagged for this node.
    const forThisNode = ctx.accumulatedDocs.filter(
      (d) => d.ontologyNodeId === ctx.node.id,
    );
    const toFetch = (forThisNode.length > 0 ? forThisNode : ctx.accumulatedDocs).slice(
      0,
      step.maxSelect,
    );
    if (toFetch.length === 0) return EMPTY_RESULT;

    const contents: DocumentContent[] = [];
    const parts: string[] = [];
    for (const doc of toFetch) {
      const content = await ctx.fetcherRegistry.fetch(doc);
      contents.push(content);
      const body = step.fetchFull
        ? content.body.slice(0, 2000)
        : BM25BodyExtractor.extract(content.body, ctx.query, 3);
      parts.push(`### ${content.title}\n${body}`);
    }
    return {
      contextSection: parts.join("\n\n"),
      selectedDocs: [],
      fetchedContents: contents,
    };
  }
}

// ── SECTION ───────────────────────────────────────────────────

export class SectionStepExecutor implements StepExecutor {
  readonly supportedType = DepthType.SECTION;

  async execute(ctx: StepContext, step: PipelineStep): Promise<StepResult> {
    const sectionKey = step.sectionKey;
    if (!sectionKey) return EMPTY_RESULT;

    const toFetch = ctx.accumulatedDocs.slice(0, step.maxSelect);
    if (toFetch.length === 0) return EMPTY_RESULT;

    const contents: DocumentContent[] = [];
    const parts: string[] = [];
    for (const doc of toFetch) {
      const content = await ctx.fetcherRegistry.fetch(doc);
      const idx = content.body.indexOf(sectionKey);
      const sectionText = idx >= 0 ? content.body.slice(idx, idx + 1500) : "";
      if (!sectionText) continue;
      contents.push({ ...content, sectionContent: sectionText });
      parts.push(`### ${content.title} (section)\n${sectionText}`);
    }
    return {
      contextSection: parts.join("\n\n"),
      selectedDocs: toFetch,
      fetchedContents: contents,
    };
  }
}

// ── Registry ──────────────────────────────────────────────────

export class StepExecutorRegistry {
  private readonly executors = new Map<DepthType, StepExecutor>();

  constructor() {
    this.register(new OntologyStepExecutor());
    this.register(new MetaStepExecutor());
    this.register(new VectorStepExecutor());
    this.register(new ContentStepExecutor());
    this.register(new SectionStepExecutor());
  }

  register(executor: StepExecutor): void {
    this.executors.set(executor.supportedType, executor);
  }

  resolve(type: DepthType): StepExecutor {
    const exec = this.executors.get(type);
    if (!exec) throw new Error(`No StepExecutor registered for: ${type}`);
    return exec;
  }
}
