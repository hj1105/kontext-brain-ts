import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DefaultPromptTemplates, type LLMAdapter, type PromptTemplates } from "@kontext-brain/core";

/**
 * Routes prompts through the user's logged-in Codex CLI instead of an API key.
 * The chat providers bill per token against an Anthropic or OpenAI key, which a
 * ChatGPT or Claude subscription does not cover; this covers ontology building
 * with the subscription the user already has.
 *
 * It reads the final assistant message from `--output-last-message` rather than
 * scraping the transcript, which also carries hook lines and token counts.
 */

export interface CodexCliOptions {
  /** Executable to run. Defaults to `codex` on PATH. */
  readonly command?: string;
  /** Working directory for the CLI. Defaults to the current one. */
  readonly cwd?: string;
  readonly timeoutMs?: number;
  readonly templates?: PromptTemplates;
}

const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000;

export class CodexCliLLMAdapter implements LLMAdapter {
  constructor(private readonly options: CodexCliOptions = {}) {}

  async complete(systemPrompt: string, context: string, query: string): Promise<string> {
    const templates = this.options.templates ?? DefaultPromptTemplates;
    const prompt = [systemPrompt, templates.formatUserMessage(context, query)].join("\n\n");
    const root = mkdtempSync(join(tmpdir(), "kontext-codex-"));
    const lastMessagePath = join(root, "last-message.txt");
    try {
      await this.run(prompt, lastMessagePath, root);
      return readFileSync(lastMessagePath, "utf8").trim();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }

  private run(prompt: string, lastMessagePath: string, cwd: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const child = spawn(
        this.options.command ?? "codex",
        ["exec", "--skip-git-repo-check", "--output-last-message", lastMessagePath, "-"],
        { cwd: this.options.cwd ?? cwd, stdio: ["pipe", "pipe", "pipe"] },
      );
      let stderr = "";
      child.stderr?.on("data", (chunk) => {
        stderr += String(chunk);
      });
      // Why: a hung CLI would otherwise hold the whole build open indefinitely.
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        reject(
          new Error(
            `codex exec did not answer within ${this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS}ms`,
          ),
        );
      }, this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
      child.once("error", (error) => {
        clearTimeout(timer);
        reject(error);
      });
      child.once("close", (code) => {
        clearTimeout(timer);
        if (code === 0) {
          resolve();
          return;
        }
        reject(new Error(`codex exec exited ${code}: ${stderr.trim().slice(0, 400)}`));
      });
      child.stdin?.end(prompt);
    });
  }
}
