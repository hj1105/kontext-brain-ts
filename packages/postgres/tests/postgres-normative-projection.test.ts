import type { NormativeManifest } from "@kontext-brain/spec";
import type { Pool, PoolClient, QueryResult } from "pg";
import { describe, expect, it } from "vitest";
import { PostgresNormativeProjection, migratePostgres } from "../src/index.js";

const managedManifest: NormativeManifest = {
  schemaVersion: 1,
  organizationId: "org:acme",
  revisions: [
    {
      kind: "decision",
      organizationId: "org:acme",
      recordId: "decision:runtime",
      revisionId: "revision:managed:1",
      scope: { kind: "codebase", codebaseId: "codebase:example" },
      evidence: [{ evidenceId: "evidence:adr:8", sourceSpan: "ADR 0008" }],
      egress: {
        dataClassification: "internal",
        allowedRuntimeProviders: ["codex", "claude"],
      },
      authoredBy: "user:owner",
      authoredAt: "2026-08-28T00:00:00.000Z",
      statement: "Use Codex CLI through public extension points.",
    },
  ],
  activations: [
    {
      organizationId: "org:acme",
      kind: "decision",
      recordId: "decision:runtime",
      revisionId: "revision:managed:1",
      scope: { kind: "codebase", codebaseId: "codebase:example" },
      state: "accepted",
      acceptedBy: "user:owner",
      acceptedAt: "2026-08-28T00:01:00.000Z",
      mergeCommit: "abc123",
    },
  ],
};

describe("Postgres normative projection", () => {
  it("loads every ordered SQL migration", async () => {
    const statements: string[] = [];
    await migratePostgres({
      async query(sql: string) {
        statements.push(sql);
        return emptyResult();
      },
    } as unknown as Pool);

    expect(statements).toHaveLength(4);
    expect(statements[0]).toContain("kontext_resources");
    expect(statements[1]).toContain("kontext_normative_manifests");
    expect(statements[2]).toContain("kontext_scoring_profiles");
    expect(statements[3]).toContain("kontext_accuracy_manifests");
  });

  it("projects revisions before atomically moving the current manifest pointer", async () => {
    const queries: string[] = [];
    const client = {
      async query(sql: string) {
        queries.push(sql);
        return { ...emptyResult(), rowCount: 1 };
      },
      release() {},
    } as unknown as PoolClient;
    const projection = new PostgresNormativeProjection({
      async connect() {
        return client;
      },
    } as unknown as Pool);

    const digest = await projection.project({
      manifest: managedManifest,
      sourceRepository: "github:acme/governance",
      sourceCommit: "abc123",
      projectedAt: "2026-08-28T00:02:00.000Z",
    });

    expect(digest).toMatch(/^sha256:/);
    const revisionIndex = queries.findIndex((sql) =>
      sql.includes("INSERT INTO kontext_normative_revisions"),
    );
    const runtimeIndex = queries.findIndex((sql) =>
      sql.includes("INSERT INTO kontext_normative_runtime"),
    );
    expect(revisionIndex).toBeGreaterThan(-1);
    expect(runtimeIndex).toBeGreaterThan(revisionIndex);
    expect(queries.at(-1)).toBe("COMMIT");
  });

  it("rejects Local Acceptance before opening a database transaction", async () => {
    let connected = false;
    const projection = new PostgresNormativeProjection({
      async connect() {
        connected = true;
        throw new Error("must not connect");
      },
    } as unknown as Pool);
    const managedRevision = managedManifest.revisions[0];
    const managedActivation = managedManifest.activations[0];
    if (!managedRevision || !managedActivation) {
      throw new Error("Test fixture requires a managed revision and activation");
    }
    const local: NormativeManifest = {
      ...managedManifest,
      revisions: [
        {
          ...managedRevision,
          scope: { kind: "workspace", workspaceId: "workspace:local" },
        },
      ],
      activations: [
        {
          ...managedActivation,
          scope: { kind: "workspace", workspaceId: "workspace:local" },
          state: "accepted_local",
          mergeCommit: undefined,
        },
      ],
    };

    await expect(
      projection.project({
        manifest: local,
        sourceRepository: "local",
        sourceCommit: "none",
      }),
    ).rejects.toThrow("only Codebase or Organization scope");
    expect(connected).toBe(false);
  });
});

function emptyResult(): QueryResult {
  return {
    command: "",
    rowCount: 0,
    oid: 0,
    rows: [],
    fields: [],
  };
}
