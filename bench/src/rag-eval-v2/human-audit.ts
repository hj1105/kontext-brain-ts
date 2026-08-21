import { createHash } from "node:crypto";
import type {
  AnswerResult,
  DatasetBundle,
  FrameworkId,
  RetrievalResult,
} from "./contracts.js";

export interface HumanAuditLabels {
  readonly correctness: 0 | 1 | null;
  readonly faithfulness: 0 | 1 | null;
  readonly citationPrecision: 0 | 1 | null;
  readonly citationRecall: 0 | 1 | null;
  readonly acceptableAbstention: 0 | 1 | null;
  readonly notes: string;
}

export interface BlindAuditRow {
  readonly auditId: string;
  readonly datasetId: DatasetBundle["id"];
  readonly queryId: string;
  readonly category: string;
  readonly question: string;
  readonly answerable: boolean;
  readonly referenceAnswer: string | null;
  readonly goldEvidenceText: readonly string[];
  readonly candidateAnswer: AnswerResult["output"];
  readonly retrievedEvidence: RetrievalResult["evidence"];
  readonly labels: HumanAuditLabels;
}

export interface BlindAuditMapping {
  readonly auditId: string;
  readonly frameworkId: FrameworkId;
}

export interface HumanAuditSample {
  readonly rows: readonly BlindAuditRow[];
  readonly mapping: readonly BlindAuditMapping[];
}

interface Candidate {
  readonly frameworkId: FrameworkId;
  readonly answer: AnswerResult;
  readonly retrieval: RetrievalResult;
  readonly query: DatasetBundle["queries"][number];
}

export function createBlindHumanAuditSample(
  bundle: DatasetBundle,
  retrievals: readonly RetrievalResult[],
  answers: readonly AnswerResult[],
  requested: number,
  seed = "kontext-rag-eval-v2",
): HumanAuditSample {
  const retrievalByKey = new Map(
    retrievals.map((result) => [`${result.frameworkId}\0${result.queryId}`, result]),
  );
  const queryById = new Map(bundle.queries.map((query) => [query.id, query]));
  const candidates: Candidate[] = answers.flatMap((answer) => {
    if (answer.status !== "ok") return [];
    const retrieval = retrievalByKey.get(`${answer.frameworkId}\0${answer.queryId}`);
    const query = queryById.get(answer.queryId);
    return retrieval?.status === "ok" && query
      ? [{ frameworkId: answer.frameworkId, answer, retrieval, query }]
      : [];
  });
  const frameworkIds = [...new Set(candidates.map((candidate) => candidate.frameworkId))].sort();
  if (frameworkIds.length === 0 || requested <= 0) return { rows: [], mapping: [] };

  const selected: Candidate[] = [];
  const baseQuota = Math.floor(requested / frameworkIds.length);
  let remainder = requested % frameworkIds.length;
  for (const frameworkId of frameworkIds) {
    const quota = baseQuota + (remainder > 0 ? 1 : 0);
    remainder = Math.max(0, remainder - 1);
    const frameworkCandidates = candidates.filter((candidate) => candidate.frameworkId === frameworkId);
    selected.push(...stratifiedTake(frameworkCandidates, quota, seed));
  }

  const rows: BlindAuditRow[] = [];
  const mapping: BlindAuditMapping[] = [];
  for (const candidate of selected.sort((left, right) =>
    stableKey(seed, `${left.frameworkId}:${left.query.id}`).localeCompare(
      stableKey(seed, `${right.frameworkId}:${right.query.id}`),
    ),
  )) {
    const auditId = `audit-${stableKey(seed, `${candidate.frameworkId}:${candidate.query.id}`).slice(0, 16)}`;
    rows.push({
      auditId,
      datasetId: bundle.id,
      queryId: candidate.query.id,
      category: candidate.query.category,
      question: candidate.query.text,
      answerable: candidate.query.answerable,
      referenceAnswer: candidate.query.referenceAnswer,
      goldEvidenceText: candidate.query.goldEvidenceText,
      candidateAnswer: candidate.answer.output,
      retrievedEvidence: candidate.retrieval.evidence,
      labels: {
        correctness: null,
        faithfulness: null,
        citationPrecision: null,
        citationRecall: null,
        acceptableAbstention: null,
        notes: "",
      },
    });
    mapping.push({ auditId, frameworkId: candidate.frameworkId });
  }
  return { rows, mapping };
}

function stratifiedTake(candidates: readonly Candidate[], quota: number, seed: string): Candidate[] {
  const byCategory = new Map<string, Candidate[]>();
  for (const candidate of candidates) {
    const values = byCategory.get(candidate.query.category) ?? [];
    values.push(candidate);
    byCategory.set(candidate.query.category, values);
  }
  for (const values of byCategory.values()) {
    values.sort((left, right) =>
      stableKey(seed, left.query.id).localeCompare(stableKey(seed, right.query.id)),
    );
  }
  const categories = [...byCategory.keys()].sort();
  const output: Candidate[] = [];
  while (output.length < quota) {
    let added = false;
    for (const category of categories) {
      const candidate = byCategory.get(category)?.shift();
      if (!candidate) continue;
      output.push(candidate);
      added = true;
      if (output.length === quota) break;
    }
    if (!added) break;
  }
  return output;
}

function stableKey(seed: string, value: string): string {
  return createHash("sha256").update(seed).update("\0").update(value).digest("hex");
}
