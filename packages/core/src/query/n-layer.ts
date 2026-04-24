/**
 * N-layer pipeline abstraction (additive — does not replace 3-layer).
 *
 * A layer is a single index that narrows the search space. Layers chain by
 * passing `Candidate` lists between executors. The legacy 3-layer
 * (ontology → meta → content) is one preset of this; richer domains can
 * define more (e.g. law: domain → article → section → body, code: repo →
 * class → member → source).
 *
 * Candidate carries a discriminated `kind` so different input/output
 * granularities mix through the pipeline. LayerExecutor declares what kinds
 * it accepts and produces, so the pipeline loader validates chains before
 * running them.
 */

import type { Entity } from "../graph/entity.js";
import type { MetaDocument } from "../graph/layered-models.js";

export type CandidateKind = "node" | "doc" | "chunk" | "entity" | "member";

export interface NodeCandidate {
  readonly kind: "node";
  readonly nodeId: string;
  readonly score: number;
}

export interface DocCandidate {
  readonly kind: "doc";
  readonly docId: string;
  readonly meta: MetaDocument;
  readonly score: number;
}

export interface ChunkCandidate {
  readonly kind: "chunk";
  readonly docId: string;
  readonly chunkId: string;
  readonly text: string;
  readonly score: number;
}

export interface EntityCandidate {
  readonly kind: "entity";
  readonly entity: Entity;
  readonly score: number;
}

export interface MemberCandidate {
  readonly kind: "member";
  readonly memberId: string;
  readonly parentId: string;
  readonly score: number;
  readonly metadata?: Record<string, unknown>;
}

export type Candidate =
  | NodeCandidate
  | DocCandidate
  | ChunkCandidate
  | EntityCandidate
  | MemberCandidate;

export interface LayerInput {
  readonly question: string;
  readonly candidates: readonly Candidate[];
}

export interface LayerTrace {
  readonly layer: string;
  readonly receivedCount: number;
  readonly producedCount: number;
  readonly elapsedMs: number;
}

export interface LayerOutput {
  readonly question: string;
  readonly candidates: readonly Candidate[];
  readonly trace: LayerTrace;
}

export interface LayerExecutor {
  readonly name: string;
  readonly inputKinds: readonly CandidateKind[] | "any";
  readonly outputKind: CandidateKind;
  execute(input: LayerInput): Promise<LayerOutput>;
}

export interface PipelineSpec {
  readonly layers: readonly LayerExecutor[];
  readonly topK?: number;
}

/** Validate a chain by checking adjacent input/output kinds. */
export function validatePipeline(spec: PipelineSpec): void {
  for (let i = 1; i < spec.layers.length; i++) {
    const prev = spec.layers[i - 1]!;
    const curr = spec.layers[i]!;
    if (curr.inputKinds === "any") continue;
    if (!curr.inputKinds.includes(prev.outputKind)) {
      throw new Error(
        `Pipeline mismatch: layer "${curr.name}" expects ${JSON.stringify(curr.inputKinds)} but layer "${prev.name}" produces "${prev.outputKind}"`,
      );
    }
  }
}

/**
 * Run an N-layer pipeline. Returns final candidates + per-layer traces so
 * callers can explain why the final answer came from the chosen documents.
 */
export class NLayerRunner {
  constructor(private readonly spec: PipelineSpec) {
    validatePipeline(spec);
  }

  async run(question: string): Promise<{
    candidates: readonly Candidate[];
    traces: readonly LayerTrace[];
  }> {
    let candidates: readonly Candidate[] = [];
    const traces: LayerTrace[] = [];
    for (const layer of this.spec.layers) {
      const out = await layer.execute({ question, candidates });
      candidates = out.candidates;
      traces.push(out.trace);
    }
    if (this.spec.topK !== undefined) {
      candidates = [...candidates]
        .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
        .slice(0, this.spec.topK);
    }
    return { candidates, traces };
  }
}

/**
 * Helper to implement a LayerExecutor with less boilerplate. Wraps a pure
 * function and handles timing + trace.
 */
export function makeLayer(
  name: string,
  inputKinds: readonly CandidateKind[] | "any",
  outputKind: CandidateKind,
  fn: (input: LayerInput) => Promise<readonly Candidate[]>,
): LayerExecutor {
  return {
    name,
    inputKinds,
    outputKind,
    async execute(input: LayerInput): Promise<LayerOutput> {
      const t0 = performance.now();
      const produced = await fn(input);
      return {
        question: input.question,
        candidates: produced,
        trace: {
          layer: name,
          receivedCount: input.candidates.length,
          producedCount: produced.length,
          elapsedMs: performance.now() - t0,
        },
      };
    },
  };
}
