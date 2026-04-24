/**
 * Configurable LLM prompt templates. All system prompts used across the pipeline
 * live here, allowing domain/language customization without touching core logic.
 */
export interface PromptTemplates {
  readonly entityExtraction: string;
  readonly categoryExtraction: string;
  nodeDesign(targetNodeCount: number): string;
  readonly edgeInference: string;
  readonly reasoning: string;
  readonly layeredReasoning: string;
  readonly nodeClassifier: string;
  readonly metaDocumentSelector: string;
  readonly domainClassifier: string;
  documentSelection(maxSelect: number): string;
  readonly queryExpansion: string;
  readonly relevanceScoring: string;
  readonly documentClassification: string;
  readonly nodeExpansion: string;
  formatUserMessage(context: string, query: string): string;
}

export const DefaultPromptTemplates: PromptTemplates = {
  entityExtraction: `
Extract entities and relationships from the following text to build an ontology graph.
Return JSON only. Do not include any other text.
Format: {"nodes":[{"id":"entity","description":"description","weight":0.8}],"edges":[{"from":"A","to":"B","weight":0.7}]}
  `.trim(),

  categoryExtraction: `
Examine the document list and extract common topic categories.
Categories should be specific (e.g., "Machine Learning", "Database Design", "API Security").
Return only a JSON array: ["category1", "category2"]
Maximum 5 categories.
  `.trim(),

  nodeDesign: (targetNodeCount: number) =>
    `
Based on the document list and extracted categories, design approximately ${targetNodeCount} ontology nodes.
Organize nodes into a 2-level hierarchy: top-level domains (level=0) and subcategories (level=1).

Rules:
- Node ID should be a short descriptive label (e.g., "ML", "Security", "Backend")
- description: list 5-8 core keywords belonging to this node
- weight: 0.5-1.0 based on document frequency
- level: 0 for top-level domains, 1 for subcategories
- parentId: for level=1 nodes, the ID of their parent domain (level=0); null for level=0
- Aim for 3-5 top-level domains, each with 1-3 subcategories

Return JSON only:
{
  "nodes": [
    {"id": "Engineering", "description": "keywords", "weight": 1.0, "level": 0, "parentId": null},
    {"id": "Backend", "description": "api server database", "weight": 0.8, "level": 1, "parentId": "Engineering"}
  ]
}
    `.trim(),

  edgeInference: `
Infer relationships between ontology nodes.

Rules:
- Connect only highly related node pairs (do not connect all)
- weight: 0.5-0.9 (0.9 = very closely related)
- Unidirectional (from -> to based on traversal flow)
- Maximum 3 edges per node

Return JSON only:
{
  "edges": [
    {"from": "NodeA", "to": "NodeB", "weight": 0.7}
  ]
}
  `.trim(),

  reasoning:
    "You are an expert who deeply analyzes and answers questions based on the given context.",

  layeredReasoning: `
You are an expert who accurately answers questions based on the given context.
The context consists of relevant documents selected through ontology graph traversal.
Cite sources (Notion, Jira, GitHub, etc.) in your answer.
  `.trim(),

  nodeClassifier:
    "Node classifier. Return only relevant node IDs separated by commas. Example: backend,security",

  metaDocumentSelector: "Return only relevant numbers separated by commas. Example: 0,2,3",

  domainClassifier: `
Domain relevance evaluator.
Rate how relevant each domain is to the question on a scale of 0.0 to 1.0.
Return JSON only: {"domainId": score}
No explanations, just JSON.
  `.trim(),

  documentSelection: (maxSelect: number) =>
    `Return related document numbers as a JSON array only. Select up to ${maxSelect}. Example: [0,1,3]`,

  queryExpansion:
    "Search query expander. Output only 8 related keywords separated by spaces. No other text.",

  relevanceScoring: `
Document relevance evaluator.
Rate how relevant each document is to the question on a scale of 0.0 to 1.0.
Return JSON only: {"key": score, "key2": score2}
No explanations, just JSON.
  `.trim(),

  documentClassification: `
Classify each document into the most relevant ontology node.

Return JSON only:
{"mappings": {"nodeId": [0, 2, 5], "otherNodeId": [1, 3]}, "unmapped": [4, 6]}

- Each number is the document index from the list
- "unmapped" contains documents that don't fit any node
- A document should map to exactly one node (the best fit)
- No explanations, just JSON
  `.trim(),

  nodeExpansion: `
These documents don't fit into any existing ontology node.
Create new ontology nodes to categorize them.

Rules:
- Create the minimum number of nodes needed
- Node ID should be a short descriptive label (e.g., "HR", "Legal", "Design")
- description: list 5-8 core keywords
- weight: 0.5-0.8 (lower than manually defined nodes)

Return JSON only:
{
  "nodes": [{"id": "NodeName", "description": "keyword1 keyword2", "weight": 0.7}],
  "mappings": {"NodeName": [0, 1, 3]}
}
  `.trim(),

  formatUserMessage(context: string, query: string) {
    return `Context:\n${context}\n\nQuestion: ${query}`;
  },
};
