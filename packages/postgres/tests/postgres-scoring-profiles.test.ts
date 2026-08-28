import {
  DEFAULT_CALIBRATED_SCORING_PROFILE,
  type TraversalScoringProfile,
  scoringProfileDigest,
} from "@kontext-brain/core";
import type { Pool, PoolClient, QueryResultRow } from "pg";
import { describe, expect, it, vi } from "vitest";
import { PostgresScoringProfileRepository } from "../src/index.js";

const forbiddenPool = {
  async connect(): Promise<never> {
    throw new Error("database must not be consulted");
  },
} as unknown as Pool;

interface RuntimeState {
  activeDigest: string | null;
  shadowDigest: string | null;
  canaryPercent: number;
}

function fakeScoringDatabase() {
  const profiles = new Map<string, QueryResultRow>();
  const runtimes = new Map<string, RuntimeState>();
  const release = vi.fn();
  const runtimeFor = (organizationId: string): RuntimeState => {
    const existing = runtimes.get(organizationId);
    if (existing) return existing;
    const created = { activeDigest: null, shadowDigest: null, canaryPercent: 100 };
    runtimes.set(organizationId, created);
    return created;
  };
  const result = (rows: QueryResultRow[]) => ({ rows, rowCount: rows.length });
  const client = {
    async query(statement: string, values: readonly unknown[] = []) {
      const sql = statement.replace(/\s+/g, " ").trim();
      if (
        ["BEGIN", "BEGIN READ ONLY", "COMMIT", "ROLLBACK"].includes(sql) ||
        sql.startsWith("SELECT set_config") ||
        sql.startsWith("INSERT INTO kontext_organizations")
      ) {
        return result([]);
      }
      const organizationId = String(values[0]);
      if (
        sql.startsWith("INSERT INTO kontext_organization_runtime") &&
        sql.includes("scoring_canary_percent")
      ) {
        runtimeFor(organizationId).canaryPercent = Number(values[1]);
        return result([]);
      }
      if (
        sql.startsWith("SELECT 1 FROM kontext_scoring_profiles") &&
        sql.includes("status <> 'failed'")
      ) {
        const digest = String(values[1]);
        const row = profiles.get(`${organizationId}:${digest}`);
        return result(row && row.status !== "failed" ? [{ available: 1 }] : []);
      }
      if (
        sql.startsWith("SELECT 1 FROM kontext_organization_runtime") &&
        sql.includes("active_scoring_profile_digest")
      ) {
        const runtime = runtimes.get(organizationId);
        return result(runtime?.activeDigest === String(values[1]) ? [{ active: 1 }] : []);
      }
      if (
        sql.startsWith("INSERT INTO kontext_organization_runtime") &&
        sql.includes("shadow_scoring_profile_digest")
      ) {
        runtimeFor(organizationId).shadowDigest = values[1] === null ? null : String(values[1]);
        return result([]);
      }
      if (
        sql.startsWith("UPDATE kontext_scoring_profiles") &&
        sql.includes("SET status = 'failed'")
      ) {
        const digest = String(values[1]);
        const key = `${organizationId}:${digest}`;
        const row = profiles.get(key);
        if (row && row.status !== "active") {
          profiles.set(key, { ...row, status: "failed", failure: String(values[2]) });
        }
        return result([]);
      }
      if (
        sql.startsWith("UPDATE kontext_organization_runtime") &&
        sql.includes("SET shadow_scoring_profile_digest = NULL")
      ) {
        const runtime = runtimeFor(organizationId);
        if (runtime.shadowDigest === String(values[1])) runtime.shadowDigest = null;
        return result([]);
      }
      if (sql.includes("LEFT JOIN kontext_scoring_profiles")) {
        const runtime = runtimes.get(organizationId);
        if (!runtime) return result([]);
        const row = runtime.activeDigest
          ? profiles.get(`${organizationId}:${runtime.activeDigest}`)
          : undefined;
        return result([
          {
            ...(row ?? {}),
            scoring_canary_percent: runtime.canaryPercent,
          },
        ]);
      }
      if (sql.includes("runtime.active_scoring_profile_digest")) {
        const runtime = runtimes.get(organizationId);
        const row = runtime?.activeDigest
          ? profiles.get(`${organizationId}:${runtime.activeDigest}`)
          : undefined;
        return result(row ? [row] : []);
      }
      if (sql.includes("runtime.shadow_scoring_profile_digest")) {
        const runtime = runtimes.get(organizationId);
        const row = runtime?.shadowDigest
          ? profiles.get(`${organizationId}:${runtime.shadowDigest}`)
          : undefined;
        return result(row ? [row] : []);
      }
      throw new Error(`Unhandled scoring repository SQL: ${sql}`);
    },
    release,
  } as unknown as PoolClient;
  const pool = { connect: vi.fn(async () => client) } as unknown as Pool;

  const putProfile = (
    organizationId: string,
    profile: TraversalScoringProfile,
    status: "staged" | "active" | "retired" | "failed" = "active",
  ): string => {
    const digest = scoringProfileDigest(profile);
    profiles.set(`${organizationId}:${digest}`, {
      organization_id: organizationId,
      profile_id: profile.id,
      version: profile.version,
      feature_schema_version: profile.featureSchemaVersion,
      profile_digest: digest,
      profile_data: profile,
      evaluation_summary: { split: "validation" },
      status,
      failure: null,
      created_at: "2026-08-25T00:00:00.000Z",
    });
    return digest;
  };
  const setActive = (organizationId: string, digest: string | null, canaryPercent = 100) => {
    const runtime = runtimeFor(organizationId);
    runtime.activeDigest = digest;
    runtime.canaryPercent = canaryPercent;
  };

  return { pool, profiles, runtimes, runtimeFor, putProfile, setActive };
}

function resolvedProfileId(
  policy: Awaited<ReturnType<PostgresScoringProfileRepository["resolve"]>>,
): string {
  return "descriptor" in policy ? policy.descriptor.profileId : "legacy-v1";
}

describe("PostgresScoringProfileRepository", () => {
  it.each([-1, 101, 0.5, Number.NaN])(
    "rejects invalid canary percentage %s before opening a transaction",
    async (percent) => {
      const repository = new PostgresScoringProfileRepository(forbiddenPool);

      await expect(repository.setCanaryPercent("acme", percent)).rejects.toThrow(
        "Scoring canary percent must be an integer between 0 and 100",
      );
    },
  );

  it("rejects an invalid profile before opening a transaction", async () => {
    const repository = new PostgresScoringProfileRepository(forbiddenPool);
    const invalidProfile = {
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      seed: {
        ...DEFAULT_CALIBRATED_SCORING_PROFILE.seed,
        exactMatchScore: 1.1,
      },
    };

    await expect(repository.stage("acme", invalidProfile)).rejects.toThrow(
      "profile.seed.exactMatchScore must be between zero and one",
    );
  });

  it("uses the compatibility policy when an Organization has no active profile", async () => {
    const database = fakeScoringDatabase();
    const repository = new PostgresScoringProfileRepository(database.pool);

    const resolved = await repository.resolve({
      organizationId: "acme",
      subjectId: "user:1",
      groupIds: [],
    });

    expect(resolvedProfileId(resolved)).toBe("legacy-v1");
  });

  it("keeps subjects in a deterministic canary cohort", async () => {
    const database = fakeScoringDatabase();
    const profile = {
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      id: "canary-profile",
      version: 2,
    };
    database.setActive("acme", database.putProfile("acme", profile), 50);
    const repository = new PostgresScoringProfileRepository(database.pool);

    const selected = await repository.resolve({
      organizationId: "acme",
      subjectId: "user:2",
      groupIds: [],
    });
    const fallback = await repository.resolve({
      organizationId: "acme",
      subjectId: "user:1",
      groupIds: [],
    });
    const selectedAgain = await repository.resolve({
      organizationId: "acme",
      subjectId: "user:2",
      groupIds: ["different-groups-do-not-change-the-cohort"],
    });

    expect(resolvedProfileId(selected)).toBe("canary-profile");
    expect(resolvedProfileId(fallback)).toBe("legacy-v1");
    expect(resolvedProfileId(selectedAgain)).toBe("canary-profile");
  });

  it("keeps the active profile stable for the cache TTL and reloads it after expiry", async () => {
    const database = fakeScoringDatabase();
    const firstProfile = {
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      id: "active-v1",
      version: 1,
    };
    const secondProfile = {
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      id: "active-v2",
      version: 2,
    };
    const firstDigest = database.putProfile("acme", firstProfile);
    const secondDigest = database.putProfile("acme", secondProfile);
    database.setActive("acme", firstDigest);
    let now = 1_000;
    const repository = new PostgresScoringProfileRepository(
      database.pool,
      100,
      undefined,
      () => now,
    );
    const principal = { organizationId: "acme", subjectId: "user:1", groupIds: [] };

    expect(resolvedProfileId(await repository.resolve(principal))).toBe("active-v1");
    database.setActive("acme", secondDigest);
    now = 1_099;
    expect(resolvedProfileId(await repository.resolve(principal))).toBe("active-v1");
    now = 1_101;
    expect(resolvedProfileId(await repository.resolve(principal))).toBe("active-v2");
  });

  it("invalidates the active cache when the canary percentage changes", async () => {
    const database = fakeScoringDatabase();
    const profile = {
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      id: "active-canary",
      version: 2,
    };
    database.setActive("acme", database.putProfile("acme", profile));
    const repository = new PostgresScoringProfileRepository(database.pool, 60_000);
    const principal = { organizationId: "acme", subjectId: "user:2", groupIds: [] };

    expect(resolvedProfileId(await repository.resolve(principal))).toBe("active-canary");
    await repository.setCanaryPercent("acme", 0);
    expect(resolvedProfileId(await repository.resolve(principal))).toBe("legacy-v1");
    await repository.setCanaryPercent("acme", 100);
    expect(resolvedProfileId(await repository.resolve(principal))).toBe("active-canary");
  });

  it("rejects an active row whose feature schema column disagrees with its profile", async () => {
    const database = fakeScoringDatabase();
    const digest = database.putProfile("acme", DEFAULT_CALIBRATED_SCORING_PROFILE);
    const key = `acme:${digest}`;
    database.profiles.set(key, {
      ...database.profiles.get(key),
      feature_schema_version: "corrupted-schema",
    });
    database.setActive("acme", digest);
    const repository = new PostgresScoringProfileRepository(database.pool);

    await expect(repository.getActive("acme")).rejects.toThrow(
      "Scoring profile feature schema column does not match profile data",
    );
  });

  it("rejects an active row whose stored digest disagrees with its profile", async () => {
    const database = fakeScoringDatabase();
    const digest = database.putProfile("acme", DEFAULT_CALIBRATED_SCORING_PROFILE);
    const key = `acme:${digest}`;
    database.profiles.set(key, {
      ...database.profiles.get(key),
      profile_digest: "sha256:corrupted",
    });
    database.setActive("acme", digest);
    const repository = new PostgresScoringProfileRepository(database.pool);

    await expect(repository.getActive("acme")).rejects.toThrow("Scoring profile digest mismatch");
  });

  it("stops resolving a cached shadow profile after it is marked failed", async () => {
    const database = fakeScoringDatabase();
    const profile = {
      ...DEFAULT_CALIBRATED_SCORING_PROFILE,
      id: "failing-shadow",
      version: 2,
    };
    const digest = database.putProfile("acme", profile, "staged");
    const repository = new PostgresScoringProfileRepository(database.pool, 60_000);
    const principal = { organizationId: "acme", subjectId: "user:1", groupIds: [] };

    await repository.setShadow("acme", digest);
    expect(resolvedProfileId(await repository.resolveShadow(principal))).toBe("failing-shadow");
    await repository.markFailed("acme", digest, "holdout regression");

    expect(await repository.resolveShadow(principal)).toBeNull();
    expect(await repository.getShadow("acme")).toBeNull();
  });
});
