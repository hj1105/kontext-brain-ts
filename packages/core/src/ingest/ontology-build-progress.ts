/**
 * What an ontology build is doing right now, for a host to show while a run
 * that takes minutes would otherwise look hung. Phases follow the pipeline:
 * collect documents, discover topics (sampled), design nodes, classify every
 * document in batches, sync documents into the knowledge graph, project code.
 */
export type OntologyBuildPhase = "collect" | "discover" | "design" | "classify" | "sync" | "code";

export interface OntologyBuildProgressEvent {
  readonly phase: OntologyBuildPhase;
  /** Units completed so far in this phase; batches for model phases, items otherwise. */
  readonly done: number;
  /** Units expected in this phase; 0 while unknown. */
  readonly total: number;
  readonly message?: string;
}

export type OntologyBuildProgressSink = (event: OntologyBuildProgressEvent) => void;
