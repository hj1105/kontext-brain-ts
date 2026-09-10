import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  FileResourceContentStore,
  LocalKnowledgeSearch,
  type Principal,
  type ResourceSnapshot,
  SqliteKnowledgeGraphRepository,
  SyncResourceUseCase,
  tokenize,
} from "../src/index.js";

const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

const principal: Principal = { organizationId: "org-1", subjectId: "me", groupIds: [] };

function snapshot(
  externalId: string,
  title: string,
  chunks: readonly string[],
  ontologyNodeIds: readonly string[],
  connectorId = "handbook",
): ResourceSnapshot {
  return {
    organizationId: principal.organizationId,
    source: { connectorId, externalId, type: "local" },
    title,
    contentHash: `hash:${externalId}`,
    body: chunks.join("\n\n"),
    acl: { organizationWide: true },
    ontologyNodeIds,
    chunks: chunks.map((text, index) => ({
      id: `c${index}`,
      contentHash: `chunk:${externalId}:${index}`,
      text,
      position: index,
    })),
  };
}

async function graph() {
  const data = await mkdtemp(path.join(tmpdir(), "kontext-local-search-"));
  roots.push(data);
  const repository = await SqliteKnowledgeGraphRepository.open(data);
  const contentStore = new FileResourceContentStore(path.join(data, "knowledge-content"));
  const sync = new SyncResourceUseCase(repository, contentStore);
  await sync.execute(
    snapshot(
      "docs/billing.md",
      "Billing decisions",
      ["Invoices round half up.", "Failed payments retry twice before an alert."],
      ["Billing"],
    ),
  );
  await sync.execute(snapshot("docs/hiring.md", "Hiring", ["Interviews take two rounds."], ["HR"]));
  await sync.execute(
    snapshot(
      "aims:src/retry.ts",
      "src/retry.ts",
      ["export function retryPayment(times = 2) {}"],
      ["Billing"],
      "code",
    ),
  );
  return { repository, contentStore };
}

describe("tokenize", () => {
  it("keeps word-like runs in any script and drops one-character noise", () => {
    expect(tokenize("Retry payments 2회 재시도, a b")).toEqual([
      "retry",
      "payments",
      "2회",
      "재시도",
    ]);
  });
});

describe("LocalKnowledgeSearch", () => {
  it("returns Evidence-cited chunks ranked by the question's terms, titles boosted", async () => {
    const { repository, contentStore } = await graph();
    const search = new LocalKnowledgeSearch(repository, contentStore);
    const result = await search.search({ question: "how many times do payments retry", principal });
    expect(result.resourcesScanned).toBe(3);
    expect(result.chunksScanned).toBe(4);
    const [first] = result.hits;
    expect(first?.source.externalId).toBe("docs/billing.md");
    expect(first?.text).toContain("retry twice");
    expect(first?.evidenceId).toBe(`${first?.resourceId}|source|${first?.chunkId}`);
    expect(first?.matchedTerms).toEqual(expect.arrayContaining(["payments", "retry"]));
    // The code module that also mentions retry follows; hiring does not appear at all.
    expect(result.hits.map((hit) => hit.source.connectorId)).toEqual(["handbook", "code"]);
  });

  it("filters by ontology node and by connector", async () => {
    const { repository, contentStore } = await graph();
    const search = new LocalKnowledgeSearch(repository, contentStore);
    const code = await search.search({ question: "retry", principal, connectorIds: ["code"] });
    expect(code.hits.map((hit) => hit.source.externalId)).toEqual(["aims:src/retry.ts"]);
    const hr = await search.search({ question: "rounds", principal, ontologyNodeIds: ["HR"] });
    expect(hr.hits.map((hit) => hit.title)).toEqual(["Hiring"]);
    const none = await search.search({
      question: "rounds",
      principal,
      ontologyNodeIds: ["Billing"],
    });
    expect(none.hits).toEqual([]);
  });

  it("refuses resources the principal cannot read and answers nothing to an empty question", async () => {
    const { repository, contentStore } = await graph();
    const search = new LocalKnowledgeSearch(repository, contentStore);
    const stranger: Principal = { organizationId: "org-2", subjectId: "x", groupIds: [] };
    expect((await search.search({ question: "retry", principal: stranger })).hits).toEqual([]);
    expect((await search.search({ question: "  ", principal })).hits).toEqual([]);
  });
});
