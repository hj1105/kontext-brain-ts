import { createHash } from "node:crypto";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { OntologyBuildProgressEvent, OntologyBuildProgressSink } from "@kontext-brain/core";

/**
 * The CLI's stdout is the JSON result, read once at the end, so progress goes to
 * a small file the host polls: `<data>/ontology-progress/<digest of config path>.json`.
 * The host derives the same name from the workspace it asked to build.
 */

export interface OntologyBuildProgressFile extends OntologyBuildProgressEvent {
  readonly configPath: string;
  readonly startedAt: string;
  readonly updatedAt: string;
  readonly finished: boolean;
  readonly ok?: boolean;
  readonly error?: string;
}

export function ontologyProgressPath(dataDirectory: string, configPath: string): string {
  const digest = createHash("sha256").update(path.resolve(configPath)).digest("hex").slice(0, 32);
  return path.join(dataDirectory, "ontology-progress", `${digest}.json`);
}

export class OntologyBuildProgressWriter {
  private readonly file: string;
  private readonly startedAt = new Date().toISOString();
  private last: OntologyBuildProgressEvent = { phase: "collect", done: 0, total: 0 };

  constructor(
    dataDirectory: string,
    private readonly configPath: string,
  ) {
    this.file = ontologyProgressPath(dataDirectory, configPath);
    mkdirSync(path.dirname(this.file), { recursive: true });
  }

  get sink(): OntologyBuildProgressSink {
    return (event) => {
      this.last = event;
      this.write({ finished: false });
    };
  }

  finish(outcome: { ok: boolean; error?: string }): void {
    this.write({ finished: true, ...outcome });
  }

  private write(state: { finished: boolean; ok?: boolean; error?: string }): void {
    const record: OntologyBuildProgressFile = {
      ...this.last,
      configPath: path.resolve(this.configPath),
      startedAt: this.startedAt,
      updatedAt: new Date().toISOString(),
      ...state,
    };
    // Why: a reader must never see a half-written file; rename is atomic on one volume.
    const temporary = `${this.file}.${process.pid}.tmp`;
    try {
      writeFileSync(temporary, JSON.stringify(record));
      renameSync(temporary, this.file);
    } catch {
      // Progress is a courtesy; a build must not fail because it could not be shown.
    }
  }
}
