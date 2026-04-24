import { z } from "zod";

export const LLMProviderConfigSchema = z.object({
  provider: z.string(),
  model: z.string(),
  apiKey: z.string().optional().default(""),
  baseUrl: z.string().optional(),
});
export type LLMProviderConfigDto = z.infer<typeof LLMProviderConfigSchema>;

export const LLMConfigSchema = z.object({
  traversal: LLMProviderConfigSchema,
  reasoning: LLMProviderConfigSchema,
});

export const MCPConfigSchema = z.object({
  name: z.string(),
  /** For stdio: command. For SSE: URL. */
  url: z.string().optional(),
  command: z.string().optional(),
  args: z.array(z.string()).optional(),
  /** "notion" | "jira" | "github_pr" | "slack" | undefined */
  type: z.string().optional(),
  /** "stdio" | "sse" — defaults to "sse" if url is given, "stdio" if command is given */
  transport: z.enum(["stdio", "sse"]).optional(),
});

export type MCPConfigDto = z.infer<typeof MCPConfigSchema>;

export const RelationSchema = z.object({
  to: z.string(),
  weight: z.number().default(1.0),
});

export const OntologyNodeConfigSchema: z.ZodType<OntologyNodeConfig> = z.lazy(() =>
  z.object({
    id: z.string(),
    description: z.string(),
    weight: z.number().default(1.0),
    mcpSource: z.string().nullable().optional(),
    webSearch: z.boolean().default(false),
    relates: z.array(RelationSchema).default([]),
    parentId: z.string().nullable().optional(),
    level: z.number().default(0),
    nodeType: z.string().default("DOMAIN"),
    children: z.array(OntologyNodeConfigSchema).default([]),
    keywords: z.array(z.string()).default([]),
  }),
);

export interface OntologyNodeConfig {
  id: string;
  description: string;
  weight?: number;
  mcpSource?: string | null;
  webSearch?: boolean;
  relates?: Array<{ to: string; weight?: number }>;
  parentId?: string | null;
  level?: number;
  nodeType?: string;
  children?: OntologyNodeConfig[];
  keywords?: string[];
}

export const StorageConfigSchema = z.object({
  type: z.string().default("memory"),
  path: z.string().nullable().optional(),
  url: z.string().nullable().optional(),
});

export const GraphConfigDtoSchema = z.object({
  maxDepth: z.number().default(3),
  maxTokens: z.number().default(8000),
  strategy: z.string().default("WEIGHTED_DFS"),
});

export const PipelineStepDtoSchema = z.object({
  depth: z.number(),
  type: z.string(),
  maxSelect: z.number().default(5),
  sectionKey: z.string().nullable().optional(),
  fetchFull: z.boolean().default(false),
  threshold: z.number().default(0),
});

export const KontextConfigSchema = z.object({
  llm: LLMConfigSchema,
  mcp: z.array(MCPConfigSchema).default([]),
  ontology: z.array(OntologyNodeConfigSchema).default([]),
  storage: StorageConfigSchema.default({ type: "memory" }),
  graph: GraphConfigDtoSchema.default({}),
  pipeline: z.array(PipelineStepDtoSchema).nullable().optional(),
  language: z.string().default("en"),
});

export type KontextConfig = z.infer<typeof KontextConfigSchema>;
