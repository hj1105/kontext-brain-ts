import type { OntologyProposalPublisher } from "@kontext-brain/core";
import type { CanonicalOntologySnapshot } from "@kontext-brain/github";
import { describe, expect, it, vi } from "vitest";
import { OntologyMaintenanceService, type OntologyMaintenanceTarget } from "../src/index.js";

describe("OntologyMaintenanceService", () => {
  it("activates each canonical revision once and publishes against the latest base YAML", async () => {
    let canonical: CanonicalOntologySnapshot = {
      revision: "revision-1",
      yaml: "ontology:\n  - id: engineering\n",
    };
    const activateOntologyYaml = vi.fn(async (yaml: string) => ({
      changed: true,
      contentHash: `hash:${yaml.length}`,
      nodeCount: yaml.includes("HR") ? 2 : 1,
    }));
    const publishOntologyProposals = vi.fn(async (yaml: string) => ({
      changed: true,
      yaml,
      url: "https://github.test/acme/ontology/pull/1",
    }));
    const target: OntologyMaintenanceTarget = {
      activeOntologyContentHash: "hash:active",
      activeOntologyNodeCount: 1,
      activateOntologyYaml,
      publishOntologyProposals,
    };
    const source = {
      async read() {
        return canonical;
      },
    };
    const publisher: OntologyProposalPublisher = {
      async upsert() {
        return { url: "https://github.test/acme/ontology/pull/1" };
      },
    };
    const service = new OntologyMaintenanceService(target, source, publisher);

    await service.refreshCanonical();
    await service.refreshCanonical();
    expect(activateOntologyYaml).toHaveBeenCalledTimes(1);

    canonical = {
      revision: "revision-2",
      yaml: "ontology:\n  - id: engineering\n  - id: HR\n",
    };
    const activation = await service.refreshCanonical();
    const publication = await service.publishProposals();

    expect(activation).toMatchObject({ changed: true, nodeCount: 2 });
    expect(activateOntologyYaml).toHaveBeenCalledTimes(2);
    expect(publishOntologyProposals).toHaveBeenCalledWith(canonical.yaml, publisher);
    expect(publication.url).toContain("/pull/1");
    expect(service.getStatus()).toMatchObject({
      canonicalRevision: "revision-2",
      lastProposalUrl: "https://github.test/acme/ontology/pull/1",
    });
  });
});
