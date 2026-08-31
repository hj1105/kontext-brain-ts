import type { OntologyProposal } from "@kontext-brain/core";
import { describe, expect, it } from "vitest";
import { GitHubCanonicalOntologySource, GitHubOntologyProposalPublisher } from "../src/index.js";

describe("GitHubOntologyProposalPublisher", () => {
  it("reads the canonical ontology and exposes its blob revision", async () => {
    const fakeFetch: typeof fetch = async () =>
      response(200, {
        sha: "ontology-sha",
        encoding: "base64",
        content: Buffer.from("ontology:\n  - id: engineering\n").toString("base64"),
      });
    const source = new GitHubCanonicalOntologySource(
      {
        owner: "acme",
        repository: "ontology",
        token: "secret",
        ontologyPath: "config/kontext.yaml",
      },
      fakeFetch,
    );

    await expect(source.read()).resolves.toEqual({
      revision: "ontology-sha",
      yaml: "ontology:\n  - id: engineering\n",
    });
  });

  it("creates a stable proposal branch, updates YAML, and opens one draft PR", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const responses = [
      response(404, {}),
      response(200, { object: { sha: "base-sha" } }),
      response(201, {}),
      response(404, {}),
      response(200, { sha: "file-sha" }),
      response(200, []),
      response(201, { html_url: "https://github.test/acme/ontology/pull/1", number: 1 }),
    ];
    const fakeFetch: typeof fetch = async (input, init) => {
      requests.push({ url: String(input), init });
      const next = responses.shift();
      if (!next) throw new Error("Missing fake GitHub response");
      return next;
    };
    const publisher = new GitHubOntologyProposalPublisher(
      {
        owner: "acme",
        repository: "ontology",
        token: "secret",
        ontologyPath: "kontext.yaml",
      },
      fakeFetch,
    );
    const proposal: OntologyProposal = {
      organizationId: "acme",
      proposalKey: "refund",
      suggestedNodeId: "refund",
      description: "Customer refunds",
      resourceIds: ["notion:p1"],
      occurrences: 3,
      status: "open",
      updatedAt: new Date().toISOString(),
    };

    const result = await publisher.upsert({
      organizationId: "acme",
      yaml: "ontology:\n  - id: refund\n",
      proposals: [proposal],
    });

    expect(result.url).toBe("https://github.test/acme/ontology/pull/1");
    expect(requests.map((request) => request.init?.method ?? "GET")).toEqual([
      "GET",
      "GET",
      "POST",
      "GET",
      "PUT",
      "GET",
      "POST",
    ]);
    const pullBody = JSON.parse(String(requests.at(-1)?.init?.body));
    expect(pullBody).toMatchObject({ draft: true, head: "kontext/ontology-proposals" });
  });
});

function response(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}
