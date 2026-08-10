import type { OntologyProposal } from "@kontext-brain/core";
import { describe, expect, it } from "vitest";
import { GitHubOntologyProposalPublisher } from "../src/index.js";

describe("GitHubOntologyProposalPublisher", () => {
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
