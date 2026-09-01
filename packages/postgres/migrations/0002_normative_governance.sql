CREATE TABLE IF NOT EXISTS kontext_normative_manifests (
  organization_id text NOT NULL REFERENCES kontext_organizations(organization_id),
  manifest_digest text NOT NULL,
  manifest_data jsonb NOT NULL,
  source_repository text NOT NULL,
  source_commit text NOT NULL,
  projected_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, manifest_digest)
);

CREATE TABLE IF NOT EXISTS kontext_normative_revisions (
  organization_id text NOT NULL REFERENCES kontext_organizations(organization_id),
  kind text NOT NULL CHECK (kind IN ('decision', 'domain_term', 'invariant')),
  record_id text NOT NULL,
  revision_id text NOT NULL,
  revision_data jsonb NOT NULL,
  first_manifest_digest text NOT NULL,
  projected_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, kind, record_id, revision_id),
  FOREIGN KEY (organization_id, first_manifest_digest)
    REFERENCES kontext_normative_manifests(organization_id, manifest_digest)
);

CREATE TABLE IF NOT EXISTS kontext_normative_activations (
  organization_id text NOT NULL REFERENCES kontext_organizations(organization_id),
  kind text NOT NULL CHECK (kind IN ('decision', 'domain_term', 'invariant')),
  record_id text NOT NULL,
  scope_key text NOT NULL,
  revision_id text NOT NULL,
  activation_data jsonb NOT NULL,
  manifest_digest text NOT NULL,
  projected_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, kind, record_id, scope_key),
  FOREIGN KEY (organization_id, kind, record_id, revision_id)
    REFERENCES kontext_normative_revisions(organization_id, kind, record_id, revision_id),
  FOREIGN KEY (organization_id, manifest_digest)
    REFERENCES kontext_normative_manifests(organization_id, manifest_digest)
);

CREATE TABLE IF NOT EXISTS kontext_normative_runtime (
  organization_id text PRIMARY KEY REFERENCES kontext_organizations(organization_id),
  current_manifest_digest text NOT NULL,
  source_commit text NOT NULL,
  projected_at timestamptz NOT NULL,
  FOREIGN KEY (organization_id, current_manifest_digest)
    REFERENCES kontext_normative_manifests(organization_id, manifest_digest)
);

CREATE INDEX IF NOT EXISTS kontext_normative_revisions_record_idx
  ON kontext_normative_revisions (organization_id, kind, record_id);

ALTER TABLE kontext_normative_manifests ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_normative_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_normative_activations ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_normative_runtime ENABLE ROW LEVEL SECURITY;

ALTER TABLE kontext_normative_manifests FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_normative_revisions FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_normative_activations FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_normative_runtime FORCE ROW LEVEL SECURITY;

DO $$
DECLARE table_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY[
    'kontext_normative_manifests', 'kontext_normative_revisions',
    'kontext_normative_activations', 'kontext_normative_runtime'
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
