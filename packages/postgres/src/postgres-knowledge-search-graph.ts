import type {
  EvidenceHit,
  Principal,
  ResourceContentStore,
  SearchEdge,
  SearchGraphPort,
  SearchGraphSession,
  SearchNode,
  SearchSeed,
} from "@kontext-brain/core";
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
    const best = new Map<string, SearchSeed>();
    for (const seed of [databaseSeeds, ...additional].flat()) {
      const key = nodeKey(seed.node);
      const previous = best.get(key);
      if (!previous || seed.score > previous.score) best.set(key, seed);
    }
    return Array.from(best.values());
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
        `SELECT e.evidence_id, e.fact_key, e.resource_id, e.chunk_id,
                  f.status AS fact_status, c.source_chunk_id, c.content_object_key
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
      hits.push({
        evidenceId: String(row.evidence_id),
        factKey: row.fact_key === null ? undefined : String(row.fact_key),
        factStatus:
          row.fact_status === null ? undefined : (row.fact_status as EvidenceHit["factStatus"]),
        resourceId: String(row.resource_id),
        chunkId: String(row.chunk_id),
        text,
        score: row.fact_status === "conflict" ? 0.6 : row.fact_key === null ? 0.75 : 1,
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
        `SELECT r.resource_id,
                ts_rank(to_tsvector('simple', r.title), plainto_tsquery('simple', $4)) AS rank
         FROM kontext_resources r
         WHERE r.organization_id = $1 AND r.status = 'active'
           AND to_tsvector('simple', r.title) @@ plainto_tsquery('simple', $4)
           AND ${aclPredicate("r")}
         ORDER BY rank DESC
         LIMIT 10`,
        [principal.organizationId, principal.subjectId, [...principal.groupIds], question],
      );
      for (const row of resources.rows) {
        seeds.push({
          node: { kind: "resource", id: String(row.resource_id) },
          score: Math.max(0.2, Math.min(1, Number(row.rank))),
        });
      }

      const entities = await client.query(
        `SELECT DISTINCT entity.entity_id,
                ts_rank(to_tsvector('simple', entity.name), plainto_tsquery('simple', $4)) AS rank
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
         ORDER BY rank DESC
         LIMIT 10`,
        [principal.organizationId, principal.subjectId, [...principal.groupIds], question],
      );
      for (const row of entities.rows) {
        seeds.push({
          node: { kind: "entity", id: String(row.entity_id) },
          score: Math.max(0.25, Math.min(1, Number(row.rank))),
        });
      }

      const graph = await loadActiveOntologyGraph(client, principal.organizationId);
      const queryTerms = terms(question);
      const ontologyNodes = normalizeOntologyNodes(graph?.nodes ?? []);
      let ontologyMatches = 0;
      for (const node of ontologyNodes) {
        const overlap = Array.from(terms(`${node.id} ${node.description ?? ""}`)).filter((term) =>
          queryTerms.has(term),
        ).length;
        // Only seed ontology nodes that actually overlap the query. Non-overlapping
        // nodes are still reachable via resource/entity "lift" edges, so seeding all
        // of them just floods the frontier and wastes the traversal budget.
        if (overlap === 0) continue;
        ontologyMatches++;
        seeds.push({
          node: { kind: "ontology", id: node.id },
          score: Math.min(0.9, 0.45 + overlap * 0.15),
        });
      }
      if (seeds.length === 0 && ontologyMatches === 0) {
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
          seeds.push({ node: { kind: "ontology", id: node.id }, score: 0.08 });
        }
      }
      return seeds;
    });
  }

  private async ontologyNeighbors(
    client: PoolClient,
    node: SearchNode,
    _question: string,
    principal: Principal,
  ): Promise<SearchEdge[]> {
    const output: SearchEdge[] = [];
    const graph = await loadActiveOntologyGraph(client, principal.organizationId);
    const ontologyNodes = normalizeOntologyNodes(graph?.nodes ?? []);
    for (const relation of graph?.edges ?? []) {
      if (relation.from === node.id) {
        output.push(
          makeEdge(node, { kind: "ontology", id: relation.to }, "expand", relation.weight),
        );
      } else if (relation.to === node.id) {
        output.push(
          makeEdge(node, { kind: "ontology", id: relation.from }, "expand", relation.weight),
        );
      }
    }
    const current = ontologyNodes.find((candidate) => candidate.id === node.id);
    if (current?.parentId) {
      output.push(makeEdge(node, { kind: "ontology", id: current.parentId }, "lift", 0.9));
    }
    for (const child of ontologyNodes.filter((candidate) => candidate.parentId === node.id)) {
      output.push(makeEdge(node, { kind: "ontology", id: child.id }, "ground", 0.9));
    }
    const resources = await client.query(
      `SELECT r.resource_id
       FROM kontext_resource_ontology_links link
       JOIN kontext_resources r
         ON r.organization_id = link.organization_id AND r.resource_id = link.resource_id
       WHERE link.organization_id = $1 AND link.ontology_node_id = $4
         AND r.status = 'active' AND ${aclPredicate("r")}
       LIMIT 30`,
      [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
    );
    for (const row of resources.rows) {
      output.push(makeEdge(node, { kind: "resource", id: String(row.resource_id) }, "ground", 0.9));
    }
    return output;
  }

  private async resourceNeighbors(
    client: PoolClient,
    node: SearchNode,
    _question: string,
    principal: Principal,
  ): Promise<SearchEdge[]> {
    if (!(await visibleResource(client, node.id, principal))) return [];
    const output: SearchEdge[] = [];
    const links = await client.query(
      `SELECT ontology_node_id FROM kontext_resource_ontology_links
       WHERE organization_id = $1 AND resource_id = $2`,
      [principal.organizationId, node.id],
    );
    for (const row of links.rows) {
      output.push(
        makeEdge(node, { kind: "ontology", id: String(row.ontology_node_id) }, "lift", 0.9),
      );
    }
    const chunks = await client.query(
      `SELECT c.chunk_id
       FROM kontext_chunks c
       WHERE c.organization_id = $1 AND c.resource_id = $4 AND c.status = 'active'
         AND ${aclPredicate("c")}
       ORDER BY c.position
       LIMIT 50`,
      [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
    );
    for (const row of chunks.rows) {
      output.push(makeEdge(node, { kind: "chunk", id: String(row.chunk_id) }, "ground", 0.95));
    }
    return output;
  }

  private async chunkNeighbors(
    client: PoolClient,
    node: SearchNode,
    _question: string,
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
    const output = [makeEdge(node, { kind: "resource", id: String(resourceId) }, "lift", 0.95)];
    const entities = await client.query(
      `SELECT entity_id FROM kontext_entity_mentions
       WHERE organization_id = $1 AND chunk_id = $2 AND status = 'active'`,
      [principal.organizationId, node.id],
    );
    for (const row of entities.rows) {
      output.push(makeEdge(node, { kind: "entity", id: String(row.entity_id) }, "lift", 0.9));
    }
    const facts = await client.query(
      `SELECT fact_key FROM kontext_evidence
       WHERE organization_id = $1 AND chunk_id = $2 AND status = 'active'
         AND fact_key IS NOT NULL`,
      [principal.organizationId, node.id],
    );
    for (const row of facts.rows) {
      output.push(makeEdge(node, { kind: "fact", id: String(row.fact_key) }, "lift", 0.95));
    }
    return output;
  }

  private async entityNeighbors(
    client: PoolClient,
    node: SearchNode,
    _question: string,
    principal: Principal,
  ): Promise<SearchEdge[]> {
    const output: SearchEdge[] = [];
    const chunks = await client.query(
      `SELECT mention.chunk_id
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
      output.push(makeEdge(node, { kind: "chunk", id: String(row.chunk_id) }, "ground", 0.85));
    }
    const facts = await client.query(
      `SELECT DISTINCT f.fact_key
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
       LIMIT 30`,
      [principal.organizationId, principal.subjectId, [...principal.groupIds], node.id],
    );
    for (const row of facts.rows) {
      output.push(makeEdge(node, { kind: "fact", id: String(row.fact_key) }, "expand", 0.9));
    }
    return output;
  }

  private async factNeighbors(
    client: PoolClient,
    node: SearchNode,
    _question: string,
    principal: Principal,
  ): Promise<SearchEdge[]> {
    const result = await client.query(
      `SELECT f.subject, f.object, e.chunk_id
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
    for (const row of result.rows) {
      output.push(makeEdge(node, { kind: "chunk", id: String(row.chunk_id) }, "ground", 1));
      if (row.subject?.entityId) {
        output.push(
          makeEdge(node, { kind: "entity", id: String(row.subject.entityId) }, "expand", 0.95),
        );
      }
      if (row.object?.kind === "entity" && row.object.entity?.entityId) {
        output.push(
          makeEdge(
            node,
            { kind: "entity", id: String(row.object.entity.entityId) },
            "expand",
            0.95,
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
  confidence = 1,
): SearchEdge {
  return { from, to, operation, confidence, queryRelevance: 0.8, evidenceSupport: 0.8 };
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
    if (!previous || edge.confidence > previous.confidence) best.set(key, edge);
  }
  return Array.from(best.values());
}
