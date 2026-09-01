CREATE TABLE IF NOT EXISTS kontext_tasks (
  organization_id text NOT NULL REFERENCES kontext_organizations(organization_id),
  task_id text NOT NULL,
  task_contract_digest text NOT NULL,
  task_contract_data jsonb NOT NULL,
  created_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, task_id)
);

CREATE TABLE IF NOT EXISTS kontext_task_context_snapshots (
  organization_id text NOT NULL,
  task_id text NOT NULL,
  context_digest text NOT NULL,
  base_code_revision text NOT NULL,
  snapshot_data jsonb NOT NULL,
  created_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, task_id, context_digest),
  FOREIGN KEY (organization_id, task_id)
    REFERENCES kontext_tasks(organization_id, task_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kontext_verification_runs (
  organization_id text NOT NULL,
  task_id text NOT NULL,
  verification_run_id text NOT NULL,
  tier text NOT NULL CHECK (tier IN ('fast', 'targeted', 'full')),
  verifier_kind text NOT NULL CHECK (
    verifier_kind IN ('test', 'typecheck', 'build', 'lint', 'query', 'manual_review')
  ),
  verifier_ref text NOT NULL,
  code_revision text NOT NULL,
  context_digest text NOT NULL,
  result text NOT NULL CHECK (result IN ('passed', 'failed', 'inconclusive')),
  run_data jsonb NOT NULL,
  observed_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, verification_run_id),
  FOREIGN KEY (organization_id, task_id)
    REFERENCES kontext_tasks(organization_id, task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS kontext_verification_runs_binding_idx
  ON kontext_verification_runs (
    organization_id, task_id, code_revision, context_digest, tier, result
  );

CREATE TABLE IF NOT EXISTS kontext_verification_retry_jobs (
  organization_id text NOT NULL,
  task_id text NOT NULL,
  job_id text NOT NULL,
  code_revision text NOT NULL,
  context_digest text NOT NULL,
  status text NOT NULL CHECK (
    status IN ('queued', 'claimed', 'completed', 'superseded', 'exhausted')
  ),
  retry_count integer NOT NULL CHECK (retry_count >= 0),
  max_retries integer NOT NULL CHECK (max_retries >= 0),
  next_attempt_at timestamptz NOT NULL,
  lease_expires_at timestamptz,
  job_data jsonb NOT NULL,
  updated_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, job_id),
  FOREIGN KEY (organization_id, task_id)
    REFERENCES kontext_tasks(organization_id, task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS kontext_verification_retry_ready_idx
  ON kontext_verification_retry_jobs (organization_id, status, next_attempt_at)
  WHERE status = 'queued';

CREATE TABLE IF NOT EXISTS kontext_change_bundles (
  organization_id text NOT NULL,
  task_id text NOT NULL,
  bundle_id text NOT NULL,
  work_item_id text NOT NULL,
  base_revision text NOT NULL,
  result_revision text NOT NULL,
  context_digest text NOT NULL,
  bundle_data jsonb NOT NULL,
  submitted_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, bundle_id),
  FOREIGN KEY (organization_id, task_id)
    REFERENCES kontext_tasks(organization_id, task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS kontext_change_bundles_task_idx
  ON kontext_change_bundles (organization_id, task_id, result_revision, context_digest);

CREATE TABLE IF NOT EXISTS kontext_accuracy_manifests (
  organization_id text NOT NULL,
  task_id text NOT NULL,
  manifest_id text NOT NULL,
  result_code_revision text NOT NULL,
  context_digest text NOT NULL,
  manifest_data jsonb NOT NULL,
  created_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, manifest_id),
  FOREIGN KEY (organization_id, task_id)
    REFERENCES kontext_tasks(organization_id, task_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kontext_quarantine_records (
  organization_id text NOT NULL,
  quarantine_id text NOT NULL,
  task_id text,
  work_item_id text,
  code_revision text NOT NULL,
  context_digest text,
  status text NOT NULL CHECK (status IN ('active', 'released')),
  record_data jsonb NOT NULL,
  observed_at timestamptz NOT NULL,
  released_at timestamptz,
  PRIMARY KEY (organization_id, quarantine_id)
);

CREATE INDEX IF NOT EXISTS kontext_quarantine_active_idx
  ON kontext_quarantine_records (organization_id, task_id, work_item_id, status)
  WHERE status = 'active';

CREATE TABLE IF NOT EXISTS kontext_code_symbol_ontology_links (
  organization_id text NOT NULL REFERENCES kontext_organizations(organization_id),
  link_id text NOT NULL,
  symbol_id text NOT NULL,
  origin text NOT NULL CHECK (origin IN ('curated', 'deterministic', 'proposed')),
  link_data jsonb NOT NULL,
  created_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, link_id)
);

CREATE INDEX IF NOT EXISTS kontext_code_symbol_ontology_links_symbol_idx
  ON kontext_code_symbol_ontology_links (organization_id, symbol_id, origin);

CREATE TABLE IF NOT EXISTS kontext_drift_findings (
  organization_id text NOT NULL REFERENCES kontext_organizations(organization_id),
  finding_id text NOT NULL,
  normative_kind text NOT NULL CHECK (normative_kind IN ('decision', 'domain_term', 'invariant')),
  record_id text NOT NULL,
  from_revision_id text NOT NULL,
  to_revision_id text NOT NULL,
  code_revision text NOT NULL,
  status text NOT NULL CHECK (status IN ('open', 'resolved', 'dismissed')),
  finding_data jsonb NOT NULL,
  created_at timestamptz NOT NULL,
  PRIMARY KEY (organization_id, finding_id)
);

CREATE INDEX IF NOT EXISTS kontext_drift_findings_record_idx
  ON kontext_drift_findings (organization_id, normative_kind, record_id, status);

ALTER TABLE kontext_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_task_context_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_verification_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_verification_retry_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_change_bundles ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_accuracy_manifests ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_quarantine_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_code_symbol_ontology_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_drift_findings ENABLE ROW LEVEL SECURITY;

ALTER TABLE kontext_tasks FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_task_context_snapshots FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_verification_runs FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_verification_retry_jobs FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_change_bundles FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_accuracy_manifests FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_quarantine_records FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_code_symbol_ontology_links FORCE ROW LEVEL SECURITY;
ALTER TABLE kontext_drift_findings FORCE ROW LEVEL SECURITY;

DO $$
DECLARE table_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY[
    'kontext_tasks', 'kontext_task_context_snapshots', 'kontext_verification_runs',
    'kontext_verification_retry_jobs', 'kontext_change_bundles',
    'kontext_accuracy_manifests', 'kontext_quarantine_records',
    'kontext_code_symbol_ontology_links', 'kontext_drift_findings'
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
