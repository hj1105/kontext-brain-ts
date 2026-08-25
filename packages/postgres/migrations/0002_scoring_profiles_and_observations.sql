ALTER TABLE kontext_resource_ontology_links
  ADD COLUMN IF NOT EXISTS origin text,
  ADD COLUMN IF NOT EXISTS confidence real,
  ADD COLUMN IF NOT EXISTS created_at timestamptz;

ALTER TABLE kontext_chunk_ontology_links
  ADD COLUMN IF NOT EXISTS origin text,
  ADD COLUMN IF NOT EXISTS confidence real,
  ADD COLUMN IF NOT EXISTS created_at timestamptz;

ALTER TABLE kontext_entity_mentions
  ADD COLUMN IF NOT EXISTS origin text,
  ADD COLUMN IF NOT EXISTS extraction_confidence real,
  ADD COLUMN IF NOT EXISTS extractor_version text,
  ADD COLUMN IF NOT EXISTS observed_at timestamptz;

ALTER TABLE kontext_facts
  ADD COLUMN IF NOT EXISTS origin text,
  ADD COLUMN IF NOT EXISTS extraction_confidence real,
  ADD COLUMN IF NOT EXISTS extractor_version text,
  ADD COLUMN IF NOT EXISTS observed_at timestamptz,
  ADD COLUMN IF NOT EXISTS verified_at timestamptz;

ALTER TABLE kontext_evidence
  ADD COLUMN IF NOT EXISTS confidence real,
  ADD COLUMN IF NOT EXISTS observed_at timestamptz,
  ADD COLUMN IF NOT EXISTS verified_at timestamptz;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_resource_links_origin_values'
  ) THEN
    ALTER TABLE kontext_resource_ontology_links
      ADD CONSTRAINT kontext_resource_links_origin_values
      CHECK (origin IS NULL OR origin IN ('manual', 'automatic', 'deterministic'));
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_chunk_links_origin_values'
  ) THEN
    ALTER TABLE kontext_chunk_ontology_links
      ADD CONSTRAINT kontext_chunk_links_origin_values
      CHECK (origin IS NULL OR origin IN ('manual', 'automatic', 'deterministic'));
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_mentions_origin_values'
  ) THEN
    ALTER TABLE kontext_entity_mentions
      ADD CONSTRAINT kontext_mentions_origin_values
      CHECK (origin IS NULL OR origin IN ('derived', 'curated'));
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_facts_origin_values'
  ) THEN
    ALTER TABLE kontext_facts
      ADD CONSTRAINT kontext_facts_origin_values
      CHECK (origin IS NULL OR origin IN ('derived', 'curated'));
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_resource_links_confidence_range'
  ) THEN
    ALTER TABLE kontext_resource_ontology_links
      ADD CONSTRAINT kontext_resource_links_confidence_range
      CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_chunk_links_confidence_range'
  ) THEN
    ALTER TABLE kontext_chunk_ontology_links
      ADD CONSTRAINT kontext_chunk_links_confidence_range
      CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_mentions_confidence_range'
  ) THEN
    ALTER TABLE kontext_entity_mentions
      ADD CONSTRAINT kontext_mentions_confidence_range
      CHECK (extraction_confidence IS NULL OR extraction_confidence BETWEEN 0 AND 1);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_facts_confidence_range'
  ) THEN
    ALTER TABLE kontext_facts
      ADD CONSTRAINT kontext_facts_confidence_range
      CHECK (extraction_confidence IS NULL OR extraction_confidence BETWEEN 0 AND 1);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_evidence_confidence_range'
  ) THEN
    ALTER TABLE kontext_evidence
      ADD CONSTRAINT kontext_evidence_confidence_range
      CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1);
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS kontext_scoring_profiles (
  organization_id text NOT NULL REFERENCES kontext_organizations(organization_id),
  profile_id text NOT NULL,
  version integer NOT NULL CHECK (version > 0),
  feature_schema_version text NOT NULL,
  profile_digest text NOT NULL,
  profile_data jsonb NOT NULL,
  evaluation_summary jsonb,
  status text NOT NULL CHECK (status IN ('staged', 'active', 'retired', 'failed')),
  failure text,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (organization_id, profile_digest),
  UNIQUE (organization_id, profile_id, version)
);

ALTER TABLE kontext_scoring_profiles
  ADD COLUMN IF NOT EXISTS feature_schema_version text,
  ADD COLUMN IF NOT EXISTS evaluation_summary jsonb;

ALTER TABLE kontext_organization_runtime
  ADD COLUMN IF NOT EXISTS active_scoring_profile_digest text,
  ADD COLUMN IF NOT EXISTS shadow_scoring_profile_digest text,
  ADD COLUMN IF NOT EXISTS scoring_canary_percent smallint NOT NULL DEFAULT 100;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_runtime_scoring_canary_range'
  ) THEN
    ALTER TABLE kontext_organization_runtime
      ADD CONSTRAINT kontext_runtime_scoring_canary_range
      CHECK (scoring_canary_percent BETWEEN 0 AND 100);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_runtime_active_scoring_profile_fk'
  ) THEN
    ALTER TABLE kontext_organization_runtime
      ADD CONSTRAINT kontext_runtime_active_scoring_profile_fk
      FOREIGN KEY (organization_id, active_scoring_profile_digest)
      REFERENCES kontext_scoring_profiles(organization_id, profile_digest);
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'kontext_runtime_shadow_scoring_profile_fk'
  ) THEN
    ALTER TABLE kontext_organization_runtime
      ADD CONSTRAINT kontext_runtime_shadow_scoring_profile_fk
      FOREIGN KEY (organization_id, shadow_scoring_profile_digest)
      REFERENCES kontext_scoring_profiles(organization_id, profile_digest);
  END IF;
END $$;

ALTER TABLE kontext_scoring_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE kontext_scoring_profiles FORCE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname = current_schema()
      AND tablename = 'kontext_scoring_profiles'
      AND policyname = 'kontext_organization_isolation'
  ) THEN
    CREATE POLICY kontext_organization_isolation ON kontext_scoring_profiles
      USING (organization_id = current_setting('kontext.organization_id', true))
      WITH CHECK (organization_id = current_setting('kontext.organization_id', true));
  END IF;
END $$;
