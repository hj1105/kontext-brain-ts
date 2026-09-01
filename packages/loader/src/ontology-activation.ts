import { createHash } from "node:crypto";
import type { OntologyNodeConfig } from "./kontext-config.js";

export function computeOntologyContentHash(ontology: readonly OntologyNodeConfig[]): string {
  return createHash("sha256").update(JSON.stringify(ontology)).digest("hex");
}

export interface KnowledgeOntologyActivationPort {
  activate(input: {
    readonly organizationId: string;
    readonly yaml: string;
    readonly graph: {
      readonly nodes: ReadonlyArray<{
        readonly id: string;
        readonly description: string;
        readonly parentId?: string | null;
      }>;
      readonly edges: ReadonlyArray<{
        readonly from: string;
        readonly to: string;
        readonly weight: number;
        readonly type?: string;
      }>;
    };
  }): Promise<void>;
}
