import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DefaultPromptTemplates, type LLMAdapter, type PromptTemplates } from "@kontext-brain/core";

/**
 * The Claude Code counterpart of {@link CodexCliLLMAdapter}: routes prompts
 * through the user's logged-in Claude CLI so ontology building runs on a Claude
 * subscription instead of a billed API key.
 *
 * `--print` writes only the final assistant message to stdout, so there is no
 * transcript to scrape and no `--output-last-message` equivalent to read back.
 * Do not add `--bare`; it skips the session setup that resolves the stored
 * credentials and the CLI then reports itself as logged out.
 */

export interface ClaudeCliOptions {
  /** Executable to run. Defaults to `claude` on PATH. */
  readonly command?: string;
  /** Working directory for the CLI. Defaults to a throwaway directory. */
  readonly cwd?: string;
  readonly timeoutMs?: number;
  readonly templates?: PromptTemplates;
}

const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000;

export class ClaudeCliLLMAdapter implements LLMAdapter {
  constructor(private readonly options: ClaudeCliOptions = {}) {}

  async complete(systemPrompt: string, context: string, query: string): Promise<string> {
    const templates = this.options.templates ?? DefaultPromptTemplates;
    const prompt = [systemPrompt, templates.formatUserMessage(context, query)].join("\n\n");
    // Why: run outside any repository, or the CLI loads that project's
    // CLAUDE.md and settings into a prompt that is supposed to see only the
    // documents the caller passed.
    const root = mkdtempSync(join(tmpdir(), "kontext-claude-"));
    try {
      return (await this.run(prompt, root)).trim();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  }

  private run(prompt: string, cwd: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const child = spawn(this.options.command ?? "claude", ["--print"], {
        cwd: this.options.cwd ?? cwd,
        stdio: ["pipe", "pipe", "pipe"],
      });
      let stdout = "";
      let stderr = "";
      child.stdout?.on("data", (chunk) => {
        stdout += String(chunk);
      });
      child.stderr?.on("data", (chunk) => {
        stderr += String(chunk);
      });
      // Why: a hung CLI would otherwise hold the whole build open indefinitely.
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        reject(
          new Error(
            `claude --print did not answer within ${this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS}ms`,
          ),
        );
      }, this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
      child.once("error", (error) => {
        clearTimeout(timer);
        reject(error);
      });
      child.once("close", (code) => {
        clearTimeout(timer);
        if (code !== 0) {
          reject(new Error(`claude --print exited ${code}: ${stderr.trim().slice(0, 400)}`));
          return;
        }
        // The CLI exits 0 on an auth failure and says so on stdout instead.
        if (/^not logged in/i.test(stdout.trim())) {
          reject(new Error("claude --print is not authenticated; run `claude auth login`"));
          return;
        }
        resolve(stdout);
      });
      child.stdin?.end(prompt);
    });
  }
}
