CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kontext_organizations (
  organization_id text PRIMARY KEY,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kontext_ontology_deployments (
  organization_id text NOT NULL REFERENCES kontext_organizations(organization_id),
  content_hash text NOT NULL,
  yaml text NOT NULL,
  graph_data jsonb NOT NULL,
  git_commit text,
  status text NOT NULL CHECK (status IN ('staged', 'active', 'retired', 'failed')),
  failure text,
  created_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, content_hash)
);

CREATE TABLE IF NOT EXISTS kontext_organization_runtime (
  organization_id text PRIMARY KEY REFERENCES kontext_organizations(organization_id),
  active_ontology_hash text,
  CONSTRAINT kontext_runtime_active_ontology_fk
    FOREIGN KEY (organization_id, active_ontology_hash)
    REFERENCES kontext_ontology_deployments(organization_id, content_hash)
);

CREATE TABLE IF NOT EXISTS kontext_resources (
  organization_id text NOT NULL REFERENCES kontext_organizations(organization_id),
  resource_id text NOT NULL,
  connector_id text NOT NULL,
  external_id text NOT NULL,
  source_type text NOT NULL,
  title text NOT NULL,
  content_hash text NOT NULL,
  content_object_key text NOT NULL,
  acl jsonb NOT NULL,
  status text NOT NULL CHECK (status IN ('active', 'stale', 'purged')),
  updated_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, resource_id),
  UNIQUE (organization_id, connector_id, external_id)
);

CREATE TABLE IF NOT EXISTS kontext_resource_ontology_links (
  organization_id text NOT NULL,
  resource_id text NOT NULL,
  ontology_node_id text NOT NULL,
  PRIMARY KEY (organization_id, resource_id, ontology_node_id),
  FOREIGN KEY (organization_id, resource_id)
    REFERENCES kontext_resources(organization_id, resource_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kontext_chunks (
  organization_id text NOT NULL,
  resource_id text NOT NULL,
  chunk_id text NOT NULL,
  source_chunk_id text NOT NULL,
  content_hash text NOT NULL,
  content_object_key text NOT NULL,
  position integer NOT NULL,
  acl jsonb NOT NULL,
  status text NOT NULL CHECK (status IN ('active', 'stale', 'purged')),
  embedding vector(1536),
  PRIMARY KEY (organization_id, chunk_id),
  FOREIGN KEY (organization_id, resource_id)
    REFERENCES kontext_resources(organization_id, resource_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS kontext_chunks_resource_idx
  ON kontext_chunks (organization_id, resource_id, status);

CREATE INDEX IF NOT EXISTS kontext_chunks_embedding_hnsw_idx
  ON kontext_chunks USING hnsw (embedding vector_cosine_ops)
  WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS kontext_chunk_ontology_links (
  organization_id text NOT NULL,
  chunk_id text NOT NULL,
  ontology_node_id text NOT NULL,
  PRIMARY KEY (organization_id, chunk_id, ontology_node_id),
  FOREIGN KEY (organization_id, chunk_id)
    REFERENCES kontext_chunks(organization_id, chunk_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kontext_entities (
  organization_id text NOT NULL,
  entity_id text NOT NULL,
  scope text NOT NULL CHECK (scope IN ('resource', 'global')),
  resource_id text,
  name text NOT NULL,
  entity_type text,
  status text NOT NULL CHECK (status IN ('active', 'stale', 'purged')),
  PRIMARY KEY (organization_id, entity_id),
  FOREIGN KEY (organization_id, resource_id)
    REFERENCES kontext_resources(organization_id, resource_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kontext_entity_mentions (
  organization_id text NOT NULL,
  entity_id text NOT NULL,
  resource_id text NOT NULL,
  chunk_id text NOT NULL,
  status text NOT NULL CHECK (status IN ('active', 'stale', 'purged')),
  PRIMARY KEY (organization_id, entity_id, chunk_id),
  FOREIGN KEY (organization_id, entity_id)
    REFERENCES kontext_entities(organization_id, entity_id) ON DELETE CASCADE,
  FOREIGN KEY (organization_id, resource_id)
    REFERENCES kontext_resources(organization_id, resource_id) ON DELETE CASCADE,
  FOREIGN KEY (organization_id, chunk_id)
    REFERENCES kontext_chunks(organization_id, chunk_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kontext_facts (
  organization_id text NOT NULL,
  fact_key text NOT NULL,
  subject jsonb NOT NULL,
  predicate text NOT NULL,
  object jsonb NOT NULL,
  single_value boolean NOT NULL DEFAULT false,
  status text NOT NULL CHECK (status IN ('active', 'inactive', 'conflict')),
  updated_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, fact_key)
);

CREATE INDEX IF NOT EXISTS kontext_facts_slot_idx
  ON kontext_facts (organization_id, predicate, status);

CREATE TABLE IF NOT EXISTS kontext_evidence (
  organization_id text NOT NULL,
  evidence_id text NOT NULL,
  fact_key text,
  resource_id text NOT NULL,
  chunk_id text NOT NULL,
  acl jsonb NOT NULL,
  origin text NOT NULL CHECK (origin IN ('derived', 'curated')),
  status text NOT NULL CHECK (status IN ('active', 'stale', 'purged')),
  PRIMARY KEY (organization_id, evidence_id),
  FOREIGN KEY (organization_id, fact_key)
    REFERENCES kontext_facts(organization_id, fact_key) ON DELETE CASCADE,
  FOREIGN KEY (organization_id, resource_id)
    REFERENCES kontext_resources(organization_id, resource_id) ON DELETE CASCADE,
  FOREIGN KEY (organization_id, chunk_id)
    REFERENCES kontext_chunks(organization_id, chunk_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS kontext_evidence_fact_idx
  ON kontext_evidence (organization_id, fact_key, status);

CREATE TABLE IF NOT EXISTS kontext_fact_events (
  event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  organization_id text NOT NULL,
  fact_key text NOT NULL,
  event_type text NOT NULL CHECK (
    event_type IN ('created', 'invalidated', 'restored', 'conflict_detected', 'conflict_resolved')
  ),
  occurred_at timestamptz NOT NULL,
  resource_id text,
  FOREIGN KEY (organization_id, fact_key)
    REFERENCES kontext_facts(organization_id, fact_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kontext_extraction_jobs (
  organization_id text NOT NULL,
  resource_id text NOT NULL,
  content_hash text NOT NULL,
  ontology_hash text NOT NULL,
  state text NOT NULL CHECK (state IN ('pending', 'running', 'succeeded', 'failed', 'dead')),
  attempts integer NOT NULL DEFAULT 0,
  available_at timestamptz NOT NULL DEFAULT now(),
  locked_by text,
  locked_until timestamptz,
  last_error text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (organization_id, resource_id, content_hash, ontology_hash)
);

CREATE TABLE IF NOT EXISTS kontext_ontology_proposals (
  organization_id text NOT NULL,
  proposal_key text NOT NULL,
  suggested_node_id text NOT NULL,
  description text NOT NULL,
  resource_ids text[] NOT NULL DEFAULT '{}',
  occurrences integer NOT NULL DEFAULT 1,
  status text NOT NULL CHECK (status IN ('open', 'published', 'accepted', 'rejected')),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (organization_id, proposal_key)
);

CREATE TABLE IF NOT EXISTS kontext_audit_events (
  event_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  organization_id text NOT NULL,
  principal_id text NOT NULL,
  event_type text NOT NULL,
  structured_ids jsonb NOT NULL DEFAULT '{}'::jsonb,
  tool_path jsonb NOT NULL DEFAULT '[]'::jsonb,
  latency_ms integer,
  cost_micros bigint,
  citation_coverage real,
  acl_outcome text,
  occurred_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE kontext_ontology_deployments ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_organization_runtime ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_resources ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_resource_ontology_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_chunk_ontology_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_entity_mentions ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_facts ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_fact_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_extraction_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_ontology_proposals ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_audit_events ENABLE ROW LEVEL SECURITY;

ALTER TABLE kontext_ontology_deployments FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_organization_runtime FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_resources FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_resource_ontology_links FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_chunk_ontology_links FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_entities FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_entity_mentions FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_facts FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_evidence FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_fact_events FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_extraction_jobs FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_ontology_proposals FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_audit_events FORCE ROW LEVEL SECURITY;

DO $$
DECLARE table_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY[
    'kontext_ontology_deployments', 'kontext_organization_runtime', 'kontext_resources',
    'kontext_resource_ontology_links', 'kontext_chunks', 'kontext_chunk_ontology_links',
    'kontext_entities', 'kontext_entity_mentions', 'kontext_facts', 'kontext_evidence',
    'kontext_fact_events', 'kontext_extraction_jobs', 'kontext_ontology_proposals',
    'kontext_audit_events'
  ]
  LOOP
    IF NOT EXISTS (
      SELECT 1 FROM pg_policies
      WHERE schemaname = current_schema() AND tablename = table_name
        AND policyname = 'kontext_organization_isolation'
    ) THEN
      EXECUTE format(
        'CREATE POLICY kontext_organization_isolation ON %I USING '
        || '(organization_id = current_setting(''kontext.organization_id'', true)) '
        || 'WITH CHECK (organization_id = current_setting(''kontext.organization_id'', true))',
        table_name
      );
    END IF;
  END LOOP;
END $$;
