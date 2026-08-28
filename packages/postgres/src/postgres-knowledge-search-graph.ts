import type {
  EvidenceHit,
  Principal,
  ResourceContentStore,
  SearchEdge,
  SearchEdgeObservations,
  SearchGraphPort,
  SearchGraphSession,
  SearchNode,
  SearchSeed,
} from "@kontext-brain/core";
import { fuseSearchSeeds } from "@kontext-brain/core";
import type { Pool, PoolClient, QueryResultRow } from "pg";
import { PostgresSearchSession, runPostgresSearchRead } from "./postgres-search-session.js";
import { aclPredicate } from "./postgres-value-utils.js";

const FALLBACK_ONTOLOGY_SEED_LIMIT = 12;

interface StoredOntologyNode {
  readonly id: string;
  readonly description?: string;
  readonly parentId?: string | null;
}

interface StoredOntologyEdge {
  readonly from: string;
  readonly to: string;
  readonly weight?: number;
  readonly type?: string;
}

export interface StoredOntologyGraph {
  readonly nodes: readonly StoredOntologyNode[] | Readonly<Record<string, StoredOntologyNode>>;
  readonly edges: readonly StoredOntologyEdge[];
}

export interface SearchSeedProvider {
  /** Database-backed providers must reuse `session` via `runPostgresSearchRead`. */
  seed(
    question: string,
    principal: Principal,
    session?: SearchGraphSession,
  ): Promise<readonly SearchSeed[]>;
}

export class PostgresKnowledgeSearchGraph implements SearchGraphPort {
  constructor(
    private readonly pool: Pool,
    private readonly contentStore: ResourceContentStore,
    private readonly additionalSeedProviders: readonly SearchSeedProvider[] = [],
  ) {}

  async openSession(principal: Principal): Promise<SearchGraphSession> {
    return PostgresSearchSession.open(this.pool, principal.organizationId);
  }

  /**
   * Runs read-only work on the traversal's shared connection when a session is
   * active, and otherwise falls back to its own short read-only transaction.
   */
  private async runRead<T>(
    principal: Principal,
    session: SearchGraphSession | undefined,
    work: (client: PoolClient) => Promise<T>,
  ): Promise<T> {
    return runPostgresSearchRead(this.pool, principal.organizationId, session, work);
  }

  async seed(
    question: string,
    principal: Principal,
    session?: SearchGraphSession,
  ): Promise<readonly SearchSeed[]> {
    // A session pins one connection, so all database-backed seed providers must
    // receive it and run sequentially. Opening a second connection here can
    // deadlock a saturated pool because the traversal already owns the first one.
    const additional: Array<readonly SearchSeed[]> = [];
    for (const provider of this.additionalSeedProviders) {
      additional.push(await provider.seed(question, principal, session));
    }
    const databaseSeeds = await this.databaseSeeds(question, principal, session);
    return fuseSearchSeeds([databaseSeeds, ...additional].flat());
  }

  async neighbors(
    node: SearchNode,
    question: string,
    principal: Principal,
    session?: SearchGraphSession,
  ): Promise<readonly SearchEdge[]> {
    const edges = await this.runRead(principal, session, async (client) => {
      switch (node.kind) {
        case "ontology":
          return this.ontologyNeighbors(client, node, question, principal);
        case "resource":
          return this.resourceNeighbors(client, node, question, principal);
        case "chunk":
          return this.chunkNeighbors(client, node, question, principal);
        case "entity":
          return this.entityNeighbors(client, node, question, principal);
        case "fact":
          return this.factNeighbors(client, node, question, principal);
      }
    });
    return deduplicateEdges(edges);
  }

  async evidence(
    node: SearchNode,
    principal: Principal,
    session?: SearchGraphSession,
  ): Promise<readonly EvidenceHit[]> {
    if (node.kind !== "chunk") return [];
    const rows = await this.runRead(principal, session, async (client) => {
      const result = await client.query(
        `SELECT e.evidence_id, e.fact_key, e.resource_id, e.chunk_id, e.origin,
                  e.confidence, e.observed_at, e.verified_at,
                  f.status AS fact_status, c.source_chunk_id, c.content_object_key,
                  r.updated_at AS resource_updated_at
           FROM kontext_evidence e
           LEFT JOIN kontext_facts f
             ON f.organization_id = e.organization_id AND f.fact_key = e.fact_key
           JOIN kontext_resources r
             ON r.organization_id = e.organization_id AND r.resource_id = e.resource_id
           JOIN kontext_chunks c
             ON c.organization_id = e.organization_id AND c.chunk_id = e.chunk_id
           WHERE e.organization_id = $1 AND e.chunk_id = $4
             AND e.status = 'active' AND (e.fact_key IS NULL OR f.status IN ('active', 'conflict'))
             AND r.status = 'active' AND c.status = 'active'
             AND ${aclPredicate("e")}
             AND ${aclPredicate("r")}
             AND ${aclPredicate("c")}`,
        [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
      );
      return result.rows;
    });
    const hits: EvidenceHit[] = [];
    for (const row of rows) {
      const content = await this.contentStore.get(String(row.content_object_key));
      const text = content?.chunks[String(row.source_chunk_id)];
      if (text === undefined) continue;
      const freshnessDate = row.verified_at ?? row.observed_at ?? row.resource_updated_at;
      const freshnessDays =
        freshnessDate === null || freshnessDate === undefined
          ? undefined
          : ageInDays(freshnessDate);
      hits.push({
        evidenceId: String(row.evidence_id),
        factKey: row.fact_key === null ? undefined : String(row.fact_key),
        factStatus:
          row.fact_status === null ? undefined : (row.fact_status as EvidenceHit["factStatus"]),
        resourceId: String(row.resource_id),
        chunkId: String(row.chunk_id),
        text,
        observations: {
          origin: row.origin as "curated" | "derived",
          ...(row.confidence === null || row.confidence === undefined
            ? {}
            : { confidence: Number(row.confidence) }),
          ...(freshnessDays === undefined ? {} : { freshnessDays }),
          support: {
            activeEvidenceCount: 1,
            curatedEvidenceCount: row.origin === "curated" ? 1 : 0,
            derivedEvidenceCount: row.origin === "derived" ? 1 : 0,
            distinctResourceCount: 1,
            conflictCount: row.fact_status === "conflict" ? 1 : 0,
            staleEvidenceCount: 0,
          },
        },
      });
    }
    return hits;
  }

  private async databaseSeeds(
    question: string,
    principal: Principal,
    session?: SearchGraphSession,
  ): Promise<SearchSeed[]> {
    return this.runRead(principal, session, async (client) => {
      const seeds: SearchSeed[] = [];
      const resources = await client.query(
        `WITH matches AS (
           SELECT r.resource_id,
                  ts_rank(to_tsvector('simple', r.title), plainto_tsquery('simple', $4)) AS native_rank
           FROM kontext_resources r
           WHERE r.organization_id = $1 AND r.status = 'active'
             AND to_tsvector('simple', r.title) @@ plainto_tsquery('simple', $4)
             AND ${aclPredicate("r")}
         )
         SELECT resource_id, native_rank,
                row_number() OVER (ORDER BY native_rank DESC, resource_id) AS retrieval_rank,
                count(*) OVER () AS candidate_count
         FROM matches
         ORDER BY native_rank DESC, resource_id
         LIMIT 10`,
        [principal.organizationId, principal.subjectId, [...principal.groupIds], question],
      );
      for (const row of resources.rows) {
        seeds.push({
          node: { kind: "resource", id: String(row.resource_id) },
          observations: {
            providers: ["postgres-resource-lexical"],
            query: { lexical: rankedObservation(row) },
          },
        });
      }

      const entities = await client.query(
        `WITH matches AS (
         SELECT entity.entity_id,
                max(ts_rank(to_tsvector('simple', entity.name), plainto_tsquery('simple', $4))) AS native_rank
         FROM kontext_entities entity
         JOIN kontext_entity_mentions mention
           ON mention.organization_id = entity.organization_id AND mention.entity_id = entity.entity_id
         JOIN kontext_chunks c
           ON c.organization_id = mention.organization_id AND c.chunk_id = mention.chunk_id
         JOIN kontext_resources r
           ON r.organization_id = mention.organization_id AND r.resource_id = mention.resource_id
         WHERE entity.organization_id = $1 AND entity.status = 'active'
           AND mention.status = 'active' AND c.status = 'active' AND r.status = 'active'
           AND to_tsvector('simple', entity.name) @@ plainto_tsquery('simple', $4)
           AND ${aclPredicate("c")} AND ${aclPredicate("r")}
         GROUP BY entity.entity_id
         )
         SELECT entity_id, native_rank,
                row_number() OVER (ORDER BY native_rank DESC, entity_id) AS retrieval_rank,
                count(*) OVER () AS candidate_count
         FROM matches
         ORDER BY native_rank DESC, entity_id
         LIMIT 10`,
        [principal.organizationId, principal.subjectId, [...principal.groupIds], question],
      );
      for (const row of entities.rows) {
        seeds.push({
          node: { kind: "entity", id: String(row.entity_id) },
          observations: {
            providers: ["postgres-entity-lexical"],
            query: { lexical: rankedObservation(row) },
          },
        });
      }

      const graph = await loadActiveOntologyGraph(client, principal.organizationId);
      const queryTerms = terms(question);
      const ontologyNodes = normalizeOntologyNodes(graph?.nodes ?? []);
      const ontologyCandidates: Array<{
        readonly node: StoredOntologyNode;
        readonly overlap: number;
        readonly exactMatch: boolean;
      }> = [];
      for (const node of ontologyNodes) {
        const nodeTerms = terms(`${node.id} ${node.description ?? ""}`);
        const overlap = Array.from(nodeTerms).filter((term) => queryTerms.has(term)).length;
        // Only seed ontology nodes that actually overlap the query. Non-overlapping
        // nodes are still reachable via resource/entity "lift" edges, so seeding all
        // of them just floods the frontier and wastes the traversal budget.
        if (overlap === 0) continue;
        ontologyCandidates.push({
          node,
          overlap,
          exactMatch: question.trim().toLocaleLowerCase() === node.id.toLocaleLowerCase(),
        });
      }
      ontologyCandidates.sort(
        (left, right) => right.overlap - left.overlap || left.node.id.localeCompare(right.node.id),
      );
      for (const [index, candidate] of ontologyCandidates.entries()) {
        seeds.push({
          node: { kind: "ontology", id: candidate.node.id },
          observations: {
            providers: ["ontology-lexical"],
            query: {
              exactMatch: candidate.exactMatch,
              lexical: {
                rank: index + 1,
                candidateCount: ontologyCandidates.length,
                normalizedScore: queryTerms.size === 0 ? 0 : candidate.overlap / queryTerms.size,
              },
            },
          },
        });
      }
      if (seeds.length === 0 && ontologyCandidates.length === 0) {
        // The default runtime has no vector provider. Keep a bounded, deterministic
        // low-score backstop so a query with no exact lexical match does not produce
        // an empty frontier, while avoiding the old all-node candidate flood.
        const fallbackNodes = [...ontologyNodes]
          .sort((left, right) => {
            const rootOrder = Number(Boolean(left.parentId)) - Number(Boolean(right.parentId));
            if (rootOrder !== 0) return rootOrder;
            return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
          })
          .slice(0, FALLBACK_ONTOLOGY_SEED_LIMIT);
        for (const node of fallbackNodes) {
          seeds.push({
            node: { kind: "ontology", id: node.id },
            observations: { fallback: true, providers: ["ontology-fallback"] },
          });
        }
      }
      return seeds;
    });
  }

  private async ontologyNeighbors(
    client: PoolClient,
    node: SearchNode,
    question: string,
    principal: Principal,
  ): Promise<SearchEdge[]> {
    const output: SearchEdge[] = [];
    const graph = await loadActiveOntologyGraph(client, principal.organizationId);
    const ontologyNodes = normalizeOntologyNodes(graph?.nodes ?? []);
    for (const relation of graph?.edges ?? []) {
      if (relation.from === node.id) {
        const target = ontologyNodes.find((candidate) => candidate.id === relation.to);
        output.push(
          makeEdge(
            node,
            { kind: "ontology", id: relation.to },
            "expand",
            { kind: "declared", weight: relation.weight },
            queryMatchObservation(question, `${relation.to} ${target?.description ?? ""}`),
          ),
        );
      } else if (relation.to === node.id) {
        const target = ontologyNodes.find((candidate) => candidate.id === relation.from);
        output.push(
          makeEdge(
            node,
            { kind: "ontology", id: relation.from },
            "expand",
            { kind: "declared", weight: relation.weight },
            queryMatchObservation(question, `${relation.from} ${target?.description ?? ""}`),
          ),
        );
      }
    }
    const current = ontologyNodes.find((candidate) => candidate.id === node.id);
    if (current?.parentId) {
      const parent = ontologyNodes.find((candidate) => candidate.id === current.parentId);
      output.push(
        makeEdge(
          node,
          { kind: "ontology", id: current.parentId },
          "lift",
          { kind: "declared" },
          queryMatchObservation(question, `${current.parentId} ${parent?.description ?? ""}`),
        ),
      );
    }
    for (const child of ontologyNodes.filter((candidate) => candidate.parentId === node.id)) {
      output.push(
        makeEdge(
          node,
          { kind: "ontology", id: child.id },
          "ground",
          { kind: "declared" },
          queryMatchObservation(question, `${child.id} ${child.description ?? ""}`),
        ),
      );
    }
    const resources = await client.query(
      `SELECT r.resource_id, r.title, link.origin, link.confidence
       FROM kontext_resource_ontology_links link
       JOIN kontext_resources r
         ON r.organization_id = link.organization_id AND r.resource_id = link.resource_id
       WHERE link.organization_id = $1 AND link.ontology_node_id = $4
         AND r.status = 'active' AND ${aclPredicate("r")}
       LIMIT 30`,
      [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
    );
    for (const row of resources.rows) {
      output.push(
        makeEdge(
          node,
          { kind: "resource", id: String(row.resource_id) },
          "ground",
          ontologyLinkObservation(row),
          queryMatchObservation(question, String(row.title)),
        ),
      );
    }
    return output;
  }

  private async resourceNeighbors(
    client: PoolClient,
    node: SearchNode,
    question: string,
    principal: Principal,
  ): Promise<SearchEdge[]> {
    if (!(await visibleResource(client, node.id, principal))) return [];
    const output: SearchEdge[] = [];
    const links = await client.query(
      `SELECT ontology_node_id, origin, confidence FROM kontext_resource_ontology_links
       WHERE organization_id = $1 AND resource_id = $2`,
      [principal.organizationId, node.id],
    );
    for (const row of links.rows) {
      output.push(
        makeEdge(
          node,
          { kind: "ontology", id: String(row.ontology_node_id) },
          "lift",
          ontologyLinkObservation(row),
          queryMatchObservation(question, String(row.ontology_node_id)),
        ),
      );
    }
    const chunks = await client.query(
      `SELECT c.chunk_id, count(*) OVER () AS candidate_count
       FROM kontext_chunks c
       WHERE c.organization_id = $1 AND c.resource_id = $4 AND c.status = 'active'
         AND ${aclPredicate("c")}
       ORDER BY c.position
       LIMIT 50`,
      [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
    );
    for (const row of chunks.rows) {
      output.push(
        makeEdge(
          node,
          { kind: "chunk", id: String(row.chunk_id) },
          "ground",
          { kind: "deterministic" },
          undefined,
          undefined,
          {
            returnedCount: chunks.rows.length,
            candidateCount: finitePositiveInteger(row.candidate_count),
          },
          { supportApplicability: "not-applicable" },
        ),
      );
    }
    return output;
  }

  private async chunkNeighbors(
    client: PoolClient,
    node: SearchNode,
    question: string,
    principal: Principal,
  ): Promise<SearchEdge[]> {
    const chunk = await client.query(
      `SELECT c.resource_id
       FROM kontext_chunks c
       JOIN kontext_resources r
         ON r.organization_id = c.organization_id AND r.resource_id = c.resource_id
       WHERE c.organization_id = $1 AND c.chunk_id = $4
         AND c.status = 'active' AND r.status = 'active'
         AND ${aclPredicate("c")} AND ${aclPredicate("r")}`,
      [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
    );
    const resourceId = chunk.rows[0]?.resource_id;
    if (!resourceId) return [];
    const output = [
      makeEdge(
        node,
        { kind: "resource", id: String(resourceId) },
        "lift",
        { kind: "deterministic" },
        undefined,
        undefined,
        undefined,
        { queryApplicability: "not-applicable", supportApplicability: "not-applicable" },
      ),
    ];
    const entities = await client.query(
      `SELECT mention.entity_id, mention.extraction_confidence, mention.extractor_version,
              mention.origin,
              entity.name
       FROM kontext_entity_mentions mention
       JOIN kontext_entities entity
         ON entity.organization_id = mention.organization_id AND entity.entity_id = mention.entity_id
       WHERE mention.organization_id = $1 AND mention.chunk_id = $2
         AND mention.status = 'active' AND entity.status = 'active'`,
      [principal.organizationId, node.id],
    );
    for (const row of entities.rows) {
      output.push(
        makeEdge(
          node,
          { kind: "entity", id: String(row.entity_id) },
          "lift",
          extractedObservation(row.extraction_confidence, row.extractor_version, row.origin),
          queryMatchObservation(question, String(row.name)),
        ),
      );
    }
    const facts = await client.query(
      `SELECT evidence.fact_key, fact.predicate, fact.subject, fact.object,
              fact.extraction_confidence, fact.extractor_version, fact.origin
       FROM kontext_evidence evidence
       JOIN kontext_facts fact
         ON fact.organization_id = evidence.organization_id AND fact.fact_key = evidence.fact_key
       WHERE evidence.organization_id = $1 AND evidence.chunk_id = $2
         AND evidence.status = 'active' AND evidence.fact_key IS NOT NULL
         AND fact.status IN ('active', 'conflict')`,
      [principal.organizationId, node.id],
    );
    for (const row of facts.rows) {
      output.push(
        makeEdge(
          node,
          { kind: "fact", id: String(row.fact_key) },
          "lift",
          extractedObservation(row.extraction_confidence, row.extractor_version, row.origin),
          queryMatchObservation(question, factSearchText(row)),
        ),
      );
    }
    return output;
  }

  private async entityNeighbors(
    client: PoolClient,
    node: SearchNode,
    question: string,
    principal: Principal,
  ): Promise<SearchEdge[]> {
    const output: SearchEdge[] = [];
    const chunks = await client.query(
      `SELECT mention.chunk_id, mention.extraction_confidence, mention.extractor_version,
              mention.origin, count(*) OVER () AS candidate_count
       FROM kontext_entity_mentions mention
       JOIN kontext_chunks c
         ON c.organization_id = mention.organization_id AND c.chunk_id = mention.chunk_id
       JOIN kontext_resources r
         ON r.organization_id = mention.organization_id AND r.resource_id = mention.resource_id
       WHERE mention.organization_id = $1 AND mention.entity_id = $4
         AND mention.status = 'active' AND c.status = 'active' AND r.status = 'active'
         AND ${aclPredicate("c")} AND ${aclPredicate("r")}
       LIMIT 30`,
      [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
    );
    for (const row of chunks.rows) {
      output.push(
        makeEdge(
          node,
          { kind: "chunk", id: String(row.chunk_id) },
          "ground",
          extractedObservation(row.extraction_confidence, row.extractor_version, row.origin),
          undefined,
          undefined,
          {
            returnedCount: chunks.rows.length,
            candidateCount: finitePositiveInteger(row.candidate_count),
          },
          { supportApplicability: "not-applicable" },
        ),
      );
    }
    const facts = await client.query(
      `SELECT f.fact_key, f.predicate, f.subject, f.object,
              f.extraction_confidence, f.extractor_version, f.origin,
              count(DISTINCT e.evidence_id) AS active_evidence_count,
              count(DISTINCT e.resource_id) AS distinct_resource_count,
              count(DISTINCT e.evidence_id) FILTER (WHERE e.origin = 'curated') AS curated_evidence_count,
              count(DISTINCT e.evidence_id) FILTER (WHERE e.origin = 'derived') AS derived_evidence_count,
              count(DISTINCT e.evidence_id) FILTER (WHERE f.status = 'conflict') AS conflict_count
       FROM kontext_facts f
       JOIN kontext_evidence e
         ON e.organization_id = f.organization_id AND e.fact_key = f.fact_key
       JOIN kontext_chunks c
         ON c.organization_id = e.organization_id AND c.chunk_id = e.chunk_id
       JOIN kontext_resources r
         ON r.organization_id = e.organization_id AND r.resource_id = e.resource_id
       WHERE f.organization_id = $1 AND f.status IN ('active','conflict')
         AND e.status = 'active' AND c.status = 'active' AND r.status = 'active'
         AND (
           f.subject->>'entityId' = $4 OR
           (f.object->>'kind' = 'entity' AND f.object->'entity'->>'entityId' = $4)
         )
         AND ${aclPredicate("e")} AND ${aclPredicate("c")} AND ${aclPredicate("r")}
       GROUP BY f.fact_key, f.predicate, f.subject, f.object,
                f.extraction_confidence, f.extractor_version, f.origin
       LIMIT 30`,
      [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
    );
    for (const row of facts.rows) {
      output.push(
        makeEdge(
          node,
          { kind: "fact", id: String(row.fact_key) },
          "expand",
          extractedObservation(row.extraction_confidence, row.extractor_version, row.origin),
          queryMatchObservation(question, factSearchText(row)),
          supportObservation(row),
        ),
      );
    }
    return output;
  }

  private async factNeighbors(
    client: PoolClient,
    node: SearchNode,
    question: string,
    principal: Principal,
  ): Promise<SearchEdge[]> {
    const result = await client.query(
      `SELECT f.subject, f.predicate, f.object, f.status,
              f.extraction_confidence, f.extractor_version, f.origin AS fact_origin,
              e.chunk_id, e.resource_id, e.origin AS evidence_origin, e.confidence
       FROM kontext_facts f
       JOIN kontext_evidence e
         ON e.organization_id = f.organization_id AND e.fact_key = f.fact_key
       JOIN kontext_chunks c
         ON c.organization_id = e.organization_id AND c.chunk_id = e.chunk_id
       JOIN kontext_resources r
         ON r.organization_id = e.organization_id AND r.resource_id = e.resource_id
       WHERE f.organization_id = $1 AND f.fact_key = $4
         AND f.status IN ('active','conflict') AND e.status = 'active'
         AND c.status = 'active' AND r.status = 'active'
         AND ${aclPredicate("e")} AND ${aclPredicate("c")} AND ${aclPredicate("r")}`,
      [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
    );
    const output: SearchEdge[] = [];
    const support = supportFromEvidenceRows(result.rows);
    for (const row of result.rows) {
      output.push(
        makeEdge(
          node,
          { kind: "chunk", id: String(row.chunk_id) },
          "ground",
          extractedObservation(row.confidence, row.extractor_version, row.evidence_origin),
          undefined,
          support,
        ),
      );
      if (row.subject?.entityId) {
        output.push(
          makeEdge(
            node,
            { kind: "entity", id: String(row.subject.entityId) },
            "expand",
            extractedObservation(row.extraction_confidence, row.extractor_version, row.fact_origin),
            queryMatchObservation(question, factSearchText(row)),
            support,
          ),
        );
      }
      if (row.object?.kind === "entity" && row.object.entity?.entityId) {
        output.push(
          makeEdge(
            node,
            { kind: "entity", id: String(row.object.entity.entityId) },
            "expand",
            extractedObservation(row.extraction_confidence, row.extractor_version, row.fact_origin),
            queryMatchObservation(question, factSearchText(row)),
            support,
          ),
        );
      }
    }
    return output;
  }
}

async function loadActiveOntologyGraph(
  client: PoolClient,
  organizationId: string,
): Promise<StoredOntologyGraph | null> {
  const result = await client.query(
    `SELECT deployment.graph_data
     FROM kontext_organization_runtime runtime
     JOIN kontext_ontology_deployments deployment
       ON deployment.organization_id = runtime.organization_id
      AND deployment.content_hash = runtime.active_ontology_hash
     WHERE runtime.organization_id = $1`,
    [organizationId],
  );
  return (result.rows[0]?.graph_data as StoredOntologyGraph | undefined) ?? null;
}

async function visibleResource(
  client: PoolClient,
  resourceId: string,
  principal: Principal,
): Promise<boolean> {
  const result = await client.query(
    `SELECT 1 FROM kontext_resources r
     WHERE r.organization_id = $1 AND r.resource_id = $4 AND r.status = 'active'
       AND ${aclPredicate("r")}`,
    [principal.organizationId, principal.subjectId, [...principal.groupIds], resourceId],
  );
  return result.rowCount === 1;
}

function makeEdge(
  from: SearchNode,
  to: SearchNode,
  operation: SearchEdge["operation"],
  structural: SearchEdgeObservations["structural"],
  query?: SearchEdgeObservations["query"],
  support?: SearchEdgeObservations["support"],
  fanout?: SearchEdgeObservations["fanout"],
  applicability?: Pick<SearchEdgeObservations, "queryApplicability" | "supportApplicability">,
): SearchEdge {
  return {
    from,
    to,
    operation,
    observations: {
      ...(structural === undefined ? {} : { structural }),
      ...(query === undefined ? {} : { query }),
      ...(support === undefined ? {} : { support }),
      ...(fanout === undefined ? {} : { fanout }),
      ...applicability,
    },
  };
}

function ontologyLinkObservation(
  row: QueryResultRow,
): SearchEdgeObservations["structural"] | undefined {
  if (row.origin === "deterministic") return { kind: "deterministic" };
  if (row.origin === "manual") {
    return { kind: "declared", weight: optionalScore(row.confidence) };
  }
  if (row.origin === "automatic") {
    return { kind: "inferred", confidence: optionalScore(row.confidence) };
  }
  return undefined;
}

function normalizeOntologyNodes(
  nodes: StoredOntologyGraph["nodes"],
): readonly StoredOntologyNode[] {
  return Array.isArray(nodes) ? nodes : Object.values(nodes);
}

function terms(text: string): Set<string> {
  return new Set(
    text
      .toLocaleLowerCase()
      .split(/[^a-z0-9가-힣_-]+/)
      .filter((term) => term.length > 1),
  );
}

function nodeKey(node: SearchNode): string {
  return `${node.kind}:${node.id}`;
}

function deduplicateEdges(edges: readonly SearchEdge[]): SearchEdge[] {
  const best = new Map<string, SearchEdge>();
  for (const edge of edges) {
    const key = `${edge.operation}:${nodeKey(edge.to)}`;
    const previous = best.get(key);
    if (!previous || edgeOrderingScore(edge) > edgeOrderingScore(previous)) best.set(key, edge);
  }
  return Array.from(best.values());
}

function rankedObservation(row: QueryResultRow): {
  readonly rank: number;
  readonly candidateCount: number;
} {
  return {
    rank: finitePositiveInteger(row.retrieval_rank),
    candidateCount: finitePositiveInteger(row.candidate_count),
  };
}

function queryMatchObservation(
  question: string,
  searchableText: string,
): SearchEdgeObservations["query"] | undefined {
  const questionTerms = terms(question);
  const candidateTerms = terms(searchableText);
  if (questionTerms.size === 0 || candidateTerms.size === 0) return undefined;
  const overlap = Array.from(questionTerms).filter((term) => candidateTerms.has(term)).length;
  if (overlap === 0) return undefined;
  return {
    exactMatch: question.trim().toLocaleLowerCase() === searchableText.trim().toLocaleLowerCase(),
    lexical: {
      rank: 1,
      candidateCount: 1,
      normalizedScore: overlap / questionTerms.size,
    },
  };
}

function extractedObservation(
  confidence: unknown,
  extractorVersion: unknown,
  origin?: unknown,
): NonNullable<SearchEdgeObservations["structural"]> {
  if (origin === "curated") return { kind: "declared", weight: optionalScore(confidence) };
  return {
    kind: "extracted",
    ...(confidence === null || confidence === undefined ? {} : { confidence: Number(confidence) }),
    ...(extractorVersion === null || extractorVersion === undefined
      ? {}
      : { extractorVersion: String(extractorVersion) }),
  };
}

function optionalScore(value: unknown): number | undefined {
  if (value === null || value === undefined) return undefined;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(0, Math.min(1, numeric)) : undefined;
}

function supportObservation(row: QueryResultRow): SearchEdgeObservations["support"] {
  return {
    activeEvidenceCount: Number(row.active_evidence_count ?? 0),
    curatedEvidenceCount: Number(row.curated_evidence_count ?? 0),
    derivedEvidenceCount: Number(row.derived_evidence_count ?? 0),
    distinctResourceCount: Number(row.distinct_resource_count ?? 0),
    conflictCount: Number(row.conflict_count ?? 0),
    staleEvidenceCount: Number(row.stale_evidence_count ?? 0),
  };
}

function supportFromEvidenceRows(
  rows: readonly QueryResultRow[],
): SearchEdgeObservations["support"] {
  const resources = new Set(rows.map((row) => String(row.resource_id)));
  return {
    activeEvidenceCount: rows.length,
    curatedEvidenceCount: rows.filter((row) => row.evidence_origin === "curated").length,
    derivedEvidenceCount: rows.filter((row) => row.evidence_origin === "derived").length,
    distinctResourceCount: resources.size,
    conflictCount: rows.some((row) => row.status === "conflict") ? rows.length : 0,
    staleEvidenceCount: 0,
  };
}

function factSearchText(row: QueryResultRow): string {
  return `${String(row.fact_key ?? "")} ${String(row.predicate ?? "")} ${JSON.stringify(
    row.subject ?? {},
  )} ${JSON.stringify(row.object ?? {})}`;
}

function edgeOrderingScore(edge: SearchEdge): number {
  const structural = edge.observations?.structural;
  if (structural?.kind === "deterministic") return 4;
  if (structural?.kind === "declared") return 3 + (structural.weight ?? 0);
  if (structural?.kind === "extracted" || structural?.kind === "inferred") {
    return (structural.kind === "extracted" ? 2 : 1) + (structural.confidence ?? 0);
  }
  return edge.confidence ?? 0;
}

function ageInDays(value: unknown): number | undefined {
  const timestamp = new Date(String(value)).getTime();
  if (!Number.isFinite(timestamp)) return undefined;
  return Math.max(0, (Date.now() - timestamp) / 86_400_000);
}

function finitePositiveInteger(value: unknown): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(1, Math.floor(numeric)) : 1;
}
