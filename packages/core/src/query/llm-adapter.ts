/**
 * LLM completion port. Provider-specific implementations live in `@kontext-brain/llm`.
 */
export interface LLMAdapter {
  complete(systemPrompt: string, context: string, query: string): Promise<string>;
}

export enum LLMRole {
  TRAVERSAL = "TRAVERSAL",
  REASONING = "REASONING",
}

/**
 * Routes LLM calls to different adapters by role.
 * Typical usage: cheap model for traversal, expensive model for final reasoning.
 */
export class RouterLLMAdapter {
  constructor(
    public readonly traversalAdapter: LLMAdapter,
    public readonly reasoningAdapter: LLMAdapter,
  ) {}

  async complete(
    role: LLMRole,
    systemPrompt: string,
    context: string,
    query: string,
  ): Promise<string> {
    switch (role) {
      case LLMRole.TRAVERSAL:
        return this.traversalAdapter.complete(systemPrompt, context, query);
      case LLMRole.REASONING:
        return this.reasoningAdapter.complete(systemPrompt, context, query);
    }
  }
}
