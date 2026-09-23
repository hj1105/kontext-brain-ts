const WHOLE_JSON_FENCE = /^```(?:json)?\r?\n([\s\S]*?)\r?\n```$/;

/**
 * Parses a model's structured JSON reply. Claude may wrap the whole object in one Markdown code
 * fence despite the prompt; only that exact shape is unwrapped. Prose around the fence, several
 * fences or an embedded `{...}` are not extracted, so they still fail JSON.parse. Callers validate
 * the parsed value; this decides only which text is JSON.
 */
export function parseModelJsonOutput(output: string): unknown {
  const trimmed = output.trim();
  return JSON.parse(WHOLE_JSON_FENCE.exec(trimmed)?.[1] ?? trimmed);
}
