import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { BenchmarkQuery, CorpusDocument, DatasetBundle } from "./contracts.js";
import { writePreparedDataset } from "./datasets.js";
import { readJsonLines } from "./jsonl.js";

const FRAMES_TSV_URL =
  "https://huggingface.co/datasets/google/frames-benchmark/resolve/main/test.tsv";
const FRAMES_TSV_REVISION = "cea20270ebb661d0ee1cdb15598c2c8fcba31025";

interface WikipediaPage {
  readonly pageid?: number;
  readonly title?: string;
  readonly extract?: string;
  readonly contentFormat?: "wikitext";
  readonly missing?: boolean;
  readonly revisions?: readonly {
    readonly slots?: { readonly main?: { readonly content?: string } };
  }[];
}

interface WikipediaResponse {
  readonly query?: {
    readonly normalized?: readonly { readonly from: string; readonly to: string }[];
    readonly redirects?: readonly { readonly from: string; readonly to: string }[];
    readonly pages?: readonly WikipediaPage[];
  };
}

interface CachedWikipediaPage {
  readonly requestedTitle: string;
  readonly page: WikipediaPage;
  readonly fetchedAt: string;
}

class HttpStatusError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfterMs?: number,
  ) {
    super(message);
  }
}

export async function prepareFramesDataset(outputDirectory: string): Promise<DatasetBundle> {
  mkdirSync(outputDirectory, { recursive: true });
  const tsvPath = join(outputDirectory, "upstream-test.tsv");
  const tsv = await downloadText(FRAMES_TSV_URL);
  writeFileSync(tsvPath, tsv, "utf8");
  const rows = parseDelimited(tsv, "\t");
  if (rows.length !== 825)
    throw new Error(`Expected FRAMES header + 824 rows, found ${rows.length}`);
  const [header, ...dataRows] = rows;
  if (header === undefined) throw new Error("FRAMES header is missing");
  const columns = new Map(header.map((name, index) => [name, index]));
  const linkColumns = header
    .map((name, index) => ({ name, index }))
    .filter(({ name }) => name.startsWith("wikipedia_link_"));
  const linksByRow = dataRows.map((row) =>
    linkColumns
      .flatMap(({ index }) => splitWikipediaLinks(row[index]?.trim() ?? ""))
      .map(normalizeWikipediaLink),
  );
  const uniqueLinks = [...new Set(linksByRow.flat())].sort();
  const titleByLink = new Map(
    await Promise.all(uniqueLinks.map(async (link) => [link, await wikipediaTitle(link)] as const)),
  );
  const pagesByTitle = await fetchWikipediaPages(
    [...new Set(titleByLink.values())].sort(),
    join(outputDirectory, "wikipedia-pages.jsonl"),
  );
  const idByLink = new Map<string, string>();
  const documentById = new Map<string, CorpusDocument>();
  for (const link of uniqueLinks) {
    const requestedTitle = titleByLink.get(link);
    if (requestedTitle === undefined) {
      throw new Error(`Wikipedia title missing for ${link}`);
    }
    const cachedPage = pagesByTitle.get(requestedTitle);
    const text = cachedPage?.page.extract?.trim();
    if (!cachedPage || !text) continue;
    const page = cachedPage.page;
    const canonicalKey =
      page.pageid === undefined ? (page.title ?? requestedTitle) : String(page.pageid);
    const id = `wikipedia:${createHash("sha256").update(canonicalKey).digest("hex").slice(0, 24)}`;
    idByLink.set(link, id);
    if (!documentById.has(id)) {
      documentById.set(id, {
        id,
        sourceId: id,
        title: page.title ?? requestedTitle,
        text,
        metadata: { url: link, pageId: page.pageid ?? null, fetchedAt: cachedPage.fetchedAt },
      });
    }
  }
  const documents = [...documentById.values()].sort((left, right) =>
    left.id.localeCompare(right.id),
  );
  const promptColumn = requiredColumn(columns, "Prompt");
  const answerColumn = requiredColumn(columns, "Answer");
  const reasoningColumn = requiredColumn(columns, "reasoning_types");
  const queries: BenchmarkQuery[] = dataRows.map((row, index) => {
    const links = linksByRow[index];
    if (links === undefined) {
      throw new Error(`Wikipedia links missing for FRAMES row ${index}`);
    }
    const goldEvidenceIds = links.flatMap((link) => {
      const id = idByLink.get(link);
      return id ? [id] : [];
    });
    return {
      id: `frames-${String(index).padStart(4, "0")}`,
      text: row[promptColumn]?.trim() ?? "",
      referenceAnswer: row[answerColumn]?.trim() || null,
      goldEvidenceIds,
      goldEvidenceText: [],
      answerable: true,
      category: row[reasoningColumn]?.trim() || "unspecified",
      metadata: {
        wikipediaLinks: JSON.stringify(links),
        missingSources: links.length - goldEvidenceIds.length,
      },
    };
  });
  if (queries.some((query) => !query.text)) throw new Error("FRAMES contains an empty question");
  const bundle: DatasetBundle = {
    id: "frames",
    track: "static-kb",
    documents,
    queries,
    provenance: {
      source: "https://huggingface.co/datasets/google/frames-benchmark",
      version: `test.tsv@${FRAMES_TSV_REVISION}+wikipedia@${new Date().toISOString()}`,
      license: "FRAMES dataset terms plus Wikipedia CC BY-SA; verify before redistribution",
    },
  };
  writePreparedDataset(outputDirectory, bundle);
  return bundle;
}

export function parseDelimited(input: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < input.length; index += 1) {
    const character = input.charAt(index);
    if (character === '"') {
      if (quoted && input[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
      continue;
    }
    if (!quoted && character === delimiter) {
      row.push(field);
      field = "";
      continue;
    }
    if (!quoted && (character === "\n" || character === "\r")) {
      if (character === "\r" && input[index + 1] === "\n") index += 1;
      row.push(field);
      if (row.some((value) => value.length > 0)) rows.push(row);
      row = [];
      field = "";
      continue;
    }
    field += character;
  }
  if (field || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  if (quoted) throw new Error("Unterminated quoted field");
  return rows;
}

async function downloadText(url: string): Promise<string> {
  const response = await fetch(url, { headers: { "User-Agent": "kontext-brain-rag-eval/2.0" } });
  if (!response.ok) throw new Error(`Download failed ${response.status} ${url}`);
  return await response.text();
}

async function wikipediaTitle(link: string): Promise<string> {
  let url = new URL(link);
  if (url.hostname === "w.wiki") {
    const response = await fetch(url, {
      method: "HEAD",
      redirect: "follow",
      headers: { "User-Agent": "kontext-brain-rag-eval/2.0" },
    });
    if (!response.ok) throw new Error(`Failed to resolve Wikipedia short link ${link}`);
    url = new URL(response.url);
  }
  const marker = "/wiki/";
  const index = url.pathname.indexOf(marker);
  if (index >= 0) {
    return repeatedlyDecode(url.pathname.slice(index + marker.length)).replace(/_/g, " ");
  }
  const title = url.searchParams.get("title");
  if (title && title !== "Special:Search") return title.replace(/_/g, " ");
  const search = url.searchParams.get("search");
  if (search) return search.replace(/\+/g, " ");
  throw new Error(`Not a Wikipedia article URL: ${link}`);
}

function normalizeWikipediaLink(link: string): string {
  const normalizedLink = link
    .replace(/\s+\(NOT REQUIRED, BUT HELPFUL\)\s*$/i, "")
    .replace(/,\s*$/, "")
    .trim();
  if (/^https?:\/\//i.test(normalizedLink)) return normalizedLink;
  if (/^(?:en\.)?wikipedia\.org\//i.test(normalizedLink)) return `https://${normalizedLink}`;
  throw new Error(`Invalid Wikipedia link: ${normalizedLink}`);
}

export function splitWikipediaLinks(value: string): string[] {
  return value
    .split(/,\s+(?=https?:\/\/)/i)
    .map((link) => link.replace(/,\s*$/, "").trim())
    .filter(Boolean);
}

function repeatedlyDecode(value: string): string {
  let output = value;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const decoded = decodeURIComponent(output);
      if (decoded === output) break;
      output = decoded;
    } catch {
      break;
    }
  }
  return output;
}

async function fetchWikipediaPages(
  titles: readonly string[],
  cachePath: string,
): Promise<Map<string, CachedWikipediaPage>> {
  const cached = existsSync(cachePath) ? readJsonLines<CachedWikipediaPage>(cachePath) : [];
  const requestedTitles = new Set(titles);
  const output = new Map(
    cached.flatMap((record) =>
      requestedTitles.has(record.requestedTitle) && record.page.contentFormat === "wikitext"
        ? [[record.requestedTitle, record] as const]
        : [],
    ),
  );
  const missingTitles = titles.filter((title) => !output.has(title));
  if (missingTitles.length > 0) {
    process.stderr.write(
      `FRAMES Wikipedia corpus: ${output.size}/${titles.length} cached; fetching ${missingTitles.length}\n`,
    );
  }
  // MediaWiki's anonymous title limit is 50. Revision content supports the
  // whole batch, unlike full-article `extracts`, which is limited to one page.
  const batchSize = 50;
  for (let offset = 0; offset < missingTitles.length; offset += batchSize) {
    const batch = missingTitles.slice(offset, offset + batchSize);
    const endpoint = new URL("https://en.wikipedia.org/w/api.php");
    endpoint.searchParams.set("action", "query");
    endpoint.searchParams.set("prop", "revisions");
    endpoint.searchParams.set("rvprop", "content");
    endpoint.searchParams.set("rvslots", "main");
    endpoint.searchParams.set("redirects", "1");
    endpoint.searchParams.set("format", "json");
    endpoint.searchParams.set("formatversion", "2");
    endpoint.searchParams.set("titles", batch.join("|"));
    const payload = await retry(async () => {
      const response = await fetch(endpoint, {
        headers: {
          "User-Agent": "kontext-brain-rag-eval/2.0 (https://github.com/hj1105/kontext-brain-ts)",
        },
      });
      if (!response.ok) {
        throw new HttpStatusError(
          `Wikipedia API ${response.status}`,
          response.status,
          parseRetryAfter(response.headers.get("retry-after")),
        );
      }
      return (await response.json()) as WikipediaResponse;
    }, 8);
    const aliases = new Map<string, string>();
    for (const mapping of payload.query?.normalized ?? []) aliases.set(mapping.from, mapping.to);
    for (const mapping of payload.query?.redirects ?? []) aliases.set(mapping.from, mapping.to);
    const pages = new Map(
      (payload.query?.pages ?? []).flatMap((page) => {
        if (!page.title) return [];
        const content = page.revisions?.[0]?.slots?.main?.content;
        return [
          [
            page.title,
            {
              pageid: page.pageid,
              title: page.title,
              extract: content,
              contentFormat: "wikitext" as const,
              missing: page.missing || !content,
            },
          ] as const,
        ];
      }),
    );
    const records: CachedWikipediaPage[] = [];
    for (const requested of batch) {
      let resolved = requested;
      for (let hop = 0; hop < 3; hop += 1) {
        const alias = aliases.get(resolved);
        if (alias === undefined) break;
        resolved = alias;
      }
      const page = pages.get(resolved) ??
        pages.get(requested) ?? {
          title: resolved,
          contentFormat: "wikitext" as const,
          missing: true,
        };
      const record = { requestedTitle: requested, page, fetchedAt: new Date().toISOString() };
      output.set(requested, record);
      records.push(record);
    }
    appendFileSync(
      cachePath,
      `${records.map((record) => JSON.stringify(record)).join("\n")}\n`,
      "utf8",
    );
    const completed = Math.min(offset + batch.length, missingTitles.length);
    if (completed === missingTitles.length || completed % 50 === 0) {
      process.stderr.write(
        `FRAMES Wikipedia corpus: ${output.size}/${titles.length} cached (${completed}/${missingTitles.length} this run)\n`,
      );
    }
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
  return output;
}

async function retry<T>(operation: () => Promise<T>, maxRetries: number): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt < maxRetries) {
        const exponentialDelay = Math.min(30_000, 1_000 * 2 ** attempt);
        const retryAfter = error instanceof HttpStatusError ? (error.retryAfterMs ?? 0) : 0;
        const delayMs = Math.min(55_000, Math.max(exponentialDelay, retryAfter));
        process.stderr.write(
          `${(error as Error).message}; retrying in ${Math.ceil(delayMs / 1_000)}s (${attempt + 1}/${maxRetries})\n`,
        );
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
  }
  throw lastError;
}

function parseRetryAfter(value: string | null): number | undefined {
  if (!value) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1_000;
  const date = Date.parse(value);
  return Number.isFinite(date) ? Math.max(0, date - Date.now()) : undefined;
}

function requiredColumn(columns: ReadonlyMap<string, number>, name: string): number {
  const index = columns.get(name);
  if (index === undefined) throw new Error(`FRAMES column ${name} is missing`);
  return index;
}

export function countPreparedFramesRows(path: string): number {
  return parseDelimited(readFileSync(path, "utf8"), "\t").length - 1;
}
