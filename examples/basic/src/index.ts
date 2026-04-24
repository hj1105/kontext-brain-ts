/**
 * Basic kontext-brain example.
 *
 * Demonstrates:
 *   - Building a graph directly in code (no YAML)
 *   - Using a mock LLM adapter (no API keys needed)
 *   - Running a query through the layered pipeline
 *
 * Run: pnpm --filter @kontext-brain/example-basic start
 */

import {
  type ContentFetcherRegistry,
  ContentFetcherRegistry as FetcherRegistry,
  DataSource,
  InMemoryMetaIndexStore,
  KeywordMappingStrategy,
  LayeredQueryPipeline,
  type LLMAdapter,
  OntologyGraph,
  RouterLLMAdapter,
  ScoreBasedSelector,
  TraversalStrategy,
  createMetaDocument,
  createNode,
} from "@kontext-brain/core";

class EchoLLM implements LLMAdapter {
  async complete(system: string, context: string, query: string): Promise<string> {
    return `[EchoLLM] Given context (${context.length} chars), answering: "${query}"`;
  }
}

async function main(): Promise<void> {
  // 1. Build ontology
  const nodes = new Map([
    ["engineering", createNode({ id: "engineering", description: "software development api backend", weight: 1.0 })],
    ["operations", createNode({ id: "operations", description: "deploy infrastructure monitoring", weight: 0.9 })],
  ]);
  const graph = new OntologyGraph(nodes, [], {
    maxDepth: 2,
    maxTokens: 2000,
    strategy: TraversalStrategy.WEIGHTED_DFS,
  });

  // 2. Index some meta documents
  const metaIndex = new InMemoryMetaIndexStore();
  await metaIndex.index("engineering", [
    createMetaDocument({
      id: "d1",
      title: "REST API Design Guide",
      source: DataSource.NOTION,
      ontologyNodeId: "engineering",
    }),
    createMetaDocument({
      id: "d2",
      title: "Database Schema Conventions",
      source: DataSource.NOTION,
      ontologyNodeId: "engineering",
    }),
  ]);

  // 3. Register a content fetcher
  const fetcherRegistry: ContentFetcherRegistry = new FetcherRegistry();
  fetcherRegistry.register({
    source: DataSource.NOTION,
    async fetch(doc) {
      return {
        metaDocumentId: doc.id,
        title: doc.title,
        body: `Body of ${doc.title}: uses REST conventions, nouns as resources.`,
        source: doc.source,
        sectionContent: null,
        fetchedAt: new Date(),
      };
    },
  });

  // 4. Run query
  const llm = new EchoLLM();
  const pipeline = new LayeredQueryPipeline(
    graph,
    new RouterLLMAdapter(llm, llm),
    metaIndex,
    fetcherRegistry,
    { mappingStrategy: new KeywordMappingStrategy(), metaSelector: new ScoreBasedSelector() },
  );

  const result = await pipeline.execute("How should I design a REST API?");
  console.log("Answer:", result.answer);
  console.log(
    "Used nodes:",
    result.usedOntologyNodes.map((n) => n.id),
  );
  console.log(
    "Selected docs:",
    result.selectedMetaDocs.map((d) => d.title),
  );
  console.log("Tokens used:", result.contextTokensUsed);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
