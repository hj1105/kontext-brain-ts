export type CodeLanguage = "typescript" | "javascript";

export type CodeSymbolKind =
  | "module"
  | "class"
  | "interface"
  | "type"
  | "enum"
  | "function"
  | "method"
  | "constructor"
  | "getter"
  | "setter"
  | "named_arrow"
  | "field"
  | "constant";

export interface CodeSymbolIdentity {
  readonly codebaseId: string;
  readonly relativePath: string;
  readonly language: CodeLanguage;
  readonly kind: CodeSymbolKind;
  readonly qualifiedName: string;
  readonly signatureDiscriminator: string;
}

export interface CodeSymbolRecord {
  readonly symbolId: string;
  readonly sourceChunkId: string;
  readonly identity: CodeSymbolIdentity;
  readonly behaviorBearing: boolean;
  readonly exported: boolean;
  readonly signature: string;
  readonly contentHash: string;
  readonly text: string;
  readonly position: number;
  readonly semanticSupport: "certified" | "syntactic";
}

export interface PlannedSymbolRecord {
  readonly plannedSymbolId: string;
  readonly taskId: string;
  readonly intendedIdentity: Partial<CodeSymbolIdentity>;
  readonly responsibility: string;
  readonly boundSymbolId?: string;
}

export interface PlannedSymbolBinding {
  readonly plannedSymbolId: string;
  readonly symbolId: string;
  readonly boundBy: "recorded_binding" | "existing_symbol_id" | "intended_identity";
}

export interface PlannedSymbolBindingIssue {
  readonly plannedSymbolId: string;
  readonly code: "bound_symbol_missing" | "identity_not_found" | "identity_ambiguous";
  readonly candidateSymbolIds: readonly string[];
}

export interface PlannedSymbolResolution {
  readonly bindings: readonly PlannedSymbolBinding[];
  readonly issues: readonly PlannedSymbolBindingIssue[];
}

export type CodeRelationshipPredicate =
  | "imports"
  | "calls"
  | "implements"
  | "extends"
  | "returns"
  | "throws"
  | "reads_env";

export type CodeRelationshipObject =
  | {
      readonly kind: "symbol";
      readonly symbolId: string;
      readonly qualifiedName: string;
      readonly entityScope: "global" | "resource";
    }
  | { readonly kind: "literal"; readonly value: string };

export interface CodeRelationship {
  readonly relationshipId: string;
  readonly subjectSymbolId: string;
  readonly predicate: CodeRelationshipPredicate;
  readonly object: CodeRelationshipObject;
  readonly evidenceSymbolId: string;
}

export interface UnresolvedCodeRelationship {
  readonly subjectSymbolId: string;
  readonly predicate: CodeRelationshipPredicate;
  readonly expression: string;
  readonly reason: "no_symbol" | "no_concrete_declaration" | "outside_project";
}

export interface CodeProjectFile {
  readonly path: string;
  readonly content: string;
}

export interface CodeAnalysisInput {
  readonly codebaseId: string;
  readonly targetPath: string;
  readonly files: readonly CodeProjectFile[];
}

export interface CodeFileAnalysis {
  readonly codebaseId: string;
  readonly relativePath: string;
  readonly language: CodeLanguage;
  readonly sourceText: string;
  readonly contentHash: string;
  readonly symbols: readonly CodeSymbolRecord[];
  readonly relationships: readonly CodeRelationship[];
  readonly unresolvedRelationships: readonly UnresolvedCodeRelationship[];
  readonly diagnostics: readonly string[];
}

export type CodeSymbolChangeKind = "added" | "modified" | "removed";

export interface CodeSymbolChange {
  readonly kind: CodeSymbolChangeKind;
  readonly symbolId: string;
  readonly before?: CodeSymbolRecord;
  readonly after?: CodeSymbolRecord;
}

export interface ReverseCodeDependency {
  readonly dependencySymbolId: string;
  readonly dependentSymbolId: string;
  readonly predicate: CodeRelationshipPredicate;
  readonly relationshipId: string;
}

export interface CodeImpactResult {
  readonly requestedSymbolIds: readonly string[];
  readonly affectedSymbols: readonly CodeSymbolRecord[];
  readonly traversedDependencies: readonly ReverseCodeDependency[];
  readonly missingSymbolIds: readonly string[];
}

export type CodeSymbolOntologyLinkOrigin = "curated" | "deterministic" | "proposed";

export type CodeSymbolOntologyTarget =
  | {
      readonly kind: "normative";
      readonly normativeKind: "decision" | "domain_term" | "invariant";
      readonly recordId: string;
      readonly revisionId: string;
    }
  | {
      readonly kind: "ontology_node";
      readonly nodeId: string;
    };

export interface CodeSymbolOntologyLink {
  readonly linkId: string;
  readonly symbolId: string;
  readonly target: CodeSymbolOntologyTarget;
  readonly origin: CodeSymbolOntologyLinkOrigin;
  readonly evidenceIds: readonly string[];
  readonly createdAt: string;
}

export interface CodeRelationshipLabel {
  readonly subject: {
    readonly relativePath: string;
    readonly qualifiedName: string;
  };
  readonly predicate: CodeRelationshipPredicate;
  readonly object:
    | {
        readonly kind: "symbol";
        readonly relativePath: string;
        readonly qualifiedName: string;
      }
    | { readonly kind: "literal"; readonly value: string };
}

export interface CodeRelationshipEvaluation {
  readonly truePositives: number;
  readonly falsePositives: number;
  readonly falseNegatives: number;
  readonly precision: number;
  readonly recall: number;
}

export interface CodeIdentityStabilityEvaluation {
  readonly comparableBehaviorSymbols: number;
  readonly stableSymbolIds: number;
  readonly stableContentHashes: number;
  readonly identityStability: number;
  readonly contentStability: number;
}

export interface LanguageCodeProvider {
  readonly language: CodeLanguage;
  readonly semanticSupport: "certified" | "syntactic";
  analyze(input: CodeAnalysisInput): CodeFileAnalysis;
}

export interface AccessControlList {
  readonly organizationWide?: boolean;
  readonly subjectIds?: readonly string[];
  readonly groupIds?: readonly string[];
}

export interface CodeResourceSnapshot {
  readonly organizationId: string;
  readonly source: {
    readonly connectorId: string;
    readonly externalId: string;
    readonly type: string;
  };
  readonly title: string;
  readonly contentHash: string;
  readonly body: string;
  readonly acl: AccessControlList;
  readonly ontologyNodeIds?: readonly string[];
  readonly chunks: readonly {
    readonly id: string;
    readonly contentHash: string;
    readonly text: string;
    readonly position: number;
    readonly ontologyNodeIds?: readonly string[];
    readonly acl?: AccessControlList;
  }[];
  readonly entities?: readonly {
    readonly entityId: string;
    readonly scope: "resource" | "global";
    readonly name: string;
    readonly type?: string;
    readonly mentionChunkIds: readonly string[];
    readonly promotionEvidence?: "deterministic" | "manual" | "resolved";
  }[];
  readonly facts?: readonly {
    readonly factKey: string;
    readonly subject: { readonly entityId: string; readonly scope: "resource" | "global" };
    readonly predicate: string;
    readonly object:
      | {
          readonly kind: "entity";
          readonly entity: { readonly entityId: string; readonly scope: "resource" | "global" };
        }
      | { readonly kind: "literal"; readonly value: string | number | boolean };
    readonly evidenceChunkIds: readonly string[];
    readonly singleValue?: boolean;
  }[];
}

export interface CodeResourceSyncResult {
  readonly resourceId: string;
  readonly changed: boolean;
  readonly affectedFactKeys: readonly string[];
}

export interface CodeResourceSyncPort {
  execute(snapshot: CodeResourceSnapshot): Promise<CodeResourceSyncResult>;
  remove(organizationId: string, source: CodeResourceSnapshot["source"]): Promise<boolean>;
}

export interface CodeSnapshotNormalizationInput {
  readonly organizationId: string;
  readonly analysis: CodeFileAnalysis;
  readonly acl: AccessControlList;
  readonly ontologyNodeIds: readonly string[];
}

export interface CodeSyncInput extends CodeAnalysisInput {
  readonly organizationId: string;
  readonly acl: AccessControlList;
  readonly ontologyNodeIds: readonly string[];
}
