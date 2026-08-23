import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { DatasetId, FrameworkId, RetrievalResult } from "./contracts.js";
import { defaultDatasetPaths, loadDataset } from "./datasets.js";
import { readJsonLines, writeJsonAtomic } from "./jsonl.js";
import { scoreDatasetFramework } from "./metrics.js";

const [datasetId, frameworkId, workDirectory] = process.argv.slice(2) as [
  DatasetId | undefined,
  FrameworkId | undefined,
  string | undefined,
];
if (!datasetId || !frameworkId || !workDirectory) {
  throw new Error("Usage: score-retrieval.ts <dataset-id> <framework-id> <work-directory>");
}

const repositoryRoot = resolve(fileURLToPath(import.meta.url), "../../../..");
const frameworkDirectory = resolve(workDirectory, datasetId, frameworkId);
const bundle = loadDataset(datasetId, defaultDatasetPaths(repositoryRoot));
const retrievals = readJsonLines<RetrievalResult>(resolve(frameworkDirectory, "retrieval.jsonl"));
const score = scoreDatasetFramework(bundle, frameworkId, retrievals, [], []);
writeJsonAtomic(resolve(frameworkDirectory, "retrieval-score.json"), score);
process.stdout.write(`${JSON.stringify(score, null, 2)}\n`);
