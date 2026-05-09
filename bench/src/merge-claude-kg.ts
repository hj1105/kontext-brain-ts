/**
 * Merge Claude-extracted entity+triple JSONL batches into a KG cache
 * compatible with kg-retriever.ts.
 *
 * Reads:  bench/data/claude-kg-{domain}-batch-NNN.jsonl
 * Writes: bench/data/gb-{domain}-kg.json (overwrites heuristic KG)
 *
 * For chunks not extracted (e.g., novel batches 024-035 didn't finish
 * due to quota limits), falls back to no entities for those chunks.
 * The KG retriever will simply return no results for queries whose
 * gold chunks aren't extracted, which is honest.
 */
import { readFileSync, writeFileSync, existsSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import type { KGSerialized, KGStore, KGEntity, KGEdge } from "./kg-builder.js";
import { normalize } from "./kg-builder.js";

interface Extraction {
  id: string;
  entities: string[];
  triples: Array<[string, string, string]>;
}

function mergeBatches(domain: "medical" | "novel"): KGStore {
  const dataDir = resolve(fileURLToPath(import.meta.url), "../../data");
  const files = readdirSync(dataDir)
    .filter((f) => f.startsWith(`claude-kg-${domain}-batch-`) && f.endsWith(".jsonl"))
    .sort();

  console.log(`[${domain}] merging ${files.length} batch files`);

  const entities = new Map<string, KGEntity>();
  const edges: KGEdge[] = [];
  const chunkToEntities = new Map<string, string[]>();

  let totalLines = 0, totalEntStrs = 0, totalTrips = 0;
  for (const f of files) {
    const lines = readFileSync(`${dataDir}/${f}`, "utf-8").split("\n").filter((l) => l.trim());
    for (const ln of lines) {
      let r: Extraction;
      try { r = JSON.parse(ln); } catch { continue; }
      totalLines++;
      const chunkEnts: string[] = [];
      for (const e of r.entities ?? []) {
        if (typeof e !== "string") continue;
        const id = normalize(e);
        if (!id || id.length < 2 || id.length > 60) continue;
        const ent = entities.get(id);
        if (ent) {
          if (!ent.chunkIds.includes(r.id)) {
            ent.chunkIds.push(r.id);
            ent.freq++;
          }
        } else {
          entities.set(id, { id, surface: e.trim(), chunkIds: [r.id], freq: 1 });
        }
        chunkEnts.push(id);
        totalEntStrs++;
      }
      chunkToEntities.set(r.id, Array.from(new Set(chunkEnts)));
      for (const t of r.triples ?? []) {
        if (!Array.isArray(t) || t.length !== 3) continue;
        const [s, p, o] = t.map((x) => (typeof x === "string" ? x : ""));
        if (!s || !p || !o) continue;
        const sid = normalize(s), oid = normalize(o);
        if (!sid || !oid || sid === oid) continue;
        edges.push({ src: sid, predicate: p.toLowerCase().replace(/\s+/g, "_"), dst: oid, chunkId: r.id });
        // Ensure both endpoints exist as entities
        if (!entities.has(sid)) entities.set(sid, { id: sid, surface: s.trim(), chunkIds: [r.id], freq: 1 });
        if (!entities.has(oid)) entities.set(oid, { id: oid, surface: o.trim(), chunkIds: [r.id], freq: 1 });
        totalTrips++;
      }
    }
  }
  console.log(`[${domain}] processed ${totalLines} chunks, ${totalEntStrs} entity-mentions, ${totalTrips} triples`);
  console.log(`[${domain}] KG: ${entities.size} unique entities, ${edges.length} edges, ${chunkToEntities.size} chunks`);

  return { entities, edges, chunkToEntities };
}

function saveKG(kg: KGStore, path: string): void {
  const ser: KGSerialized = {
    entities: Array.from(kg.entities.values()),
    edges: kg.edges,
    chunkToEntities: Array.from(kg.chunkToEntities.entries()),
  };
  writeFileSync(path, JSON.stringify(ser));
}

async function main(): Promise<void> {
  const dataDir = resolve(fileURLToPath(import.meta.url), "../../data");
  for (const dom of ["medical", "novel"] as const) {
    const kg = mergeBatches(dom);
    const out = `${dataDir}/gb-${dom}-kg.json`;
    saveKG(kg, out);
    console.log(`[${dom}] saved → ${out}\n`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
