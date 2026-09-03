# DeepSWE-based paired evaluation

This adapter measures whether Kontext's provenance-governed context improves functional software-engineering outcomes on the official DeepSWE v1.1 tasks. It uses Pier's task images and separate verifier environments, but adds one identical offline context command to every arm. Results are therefore **DeepSWE-based paired A/B results**, not official leaderboard scores.

## Arms

- `baseline`: `kontext-context` is installed, but returns no supplemental context.
- `rag`: the command retrieves raw Evidence text from the frozen Resource/Chunk provenance.
- `kontext`: the command retrieves current Decisions, Domain Terms, and Invariants plus their exact evidence closure.

The model, task instruction, image, timeout, agent implementation, command surface, and rollout count are fixed across arms. Only the context projection changes.

## Corpus ownership

The benchmark never invents or extracts organizational decisions from the DeepSWE task. A corpus is an immutable export from Kontext's already-collected provenance and effective local/managed normative state. Create it before examining DeepSWE `solution/`, `verifier/`, result, or trajectory artifacts. One file is required for every selected task, at `<corpus-root>/<task-id>.json` or `<corpus-root>/<task-id>/corpus.json`.

```json
{
  "schemaVersion": 1,
  "taskId": "python-statemachine-state-data-scoping",
  "organizationId": "organization:personal",
  "baseCodeRevision": "<DeepSWE task base commit>",
  "contextDigest": "sha256:<Task Context Snapshot digest>",
  "sourceFreshnessDigest": "sha256:<source freshness digest>",
  "snapshotAt": "2026-09-03T00:00:00.000Z",
  "generator": {
    "name": "kontext-brain",
    "revision": "<Kontext export revision>"
  },
  "evidence": [
    {
      "evidenceId": "evidence:state-data-design",
      "resourceId": "resource:state-data-design",
      "chunkId": "chunk:state-data-scope",
      "title": "State data design",
      "text": "<verbatim Evidence text>",
      "sourceUri": "file:///independent/specs/state-data.md",
      "observedAt": "2026-09-02T00:00:00.000Z",
      "contentSha256": "<SHA-256 of text>",
      "ontologyNodeIds": ["resource:state-data-design"]
    }
  ],
  "normativeRecords": [
    {
      "revision": {
        "kind": "decision",
        "organizationId": "organization:personal",
        "recordId": "decision:state-data-scope",
        "revisionId": "revision:1",
        "scope": { "kind": "codebase", "codebaseId": "python-statemachine" },
        "evidence": [{ "evidenceId": "evidence:state-data-design" }],
        "egress": {
          "dataClassification": "public",
          "allowedRuntimeProviders": ["openai"]
        },
        "authoredBy": "user:local",
        "authoredAt": "2026-09-02T00:00:00.000Z",
        "statement": "State data is owned by one state and resets on exit."
      },
      "symbolSelectors": [
        {
          "relativePath": "statemachine/statemachine.py",
          "qualifiedName": "StateMachine._activate"
        }
      ]
    }
  ]
}
```

An empty `evidence`/`normativeRecords` corpus is valid for infrastructure tests. It does not test a Kontext treatment effect.

The loader rejects future-dated evidence, hash mismatches, missing evidence closure, duplicate IDs, corpus files inside the benchmark tree, and provenance paths containing task tests, verifier, solution, trajectory, result, or agent artifacts. These structural checks do not prove authorship independence; preregistered corpora still require external review.

## Reproducible run

Pin all three repositories/tools. The paper's mini-swe-agent commit `adfe2023` is release `2.3.0`; the current adapter was checked against Pier `0.3.1` and DeepSWE revision `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`.

```bash
pnpm --filter @kontext-brain/bench code-quality:deepswe -- \
  --dataset /absolute/path/to/deep-swe/tasks \
  --corpus /absolute/path/to/frozen-corpora \
  --mini-swe-version 2.3.0 \
  --pier-revision 0.3.1 \
  --deepswe-revision 0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea \
  --model openai/gpt-5.5 \
  --sample-seed 0 \
  --task-limit 10 \
  --attempts 4 \
  --environment docker \
  --env-file .env.local
```

Add `--dry-run` to validate the corpus and write private Pier/context manifests without starting containers or model calls. Scored runs refuse a dirty Kontext checkout. Pier credentials are passed by environment file path and are never copied into the generated manifest.

On macOS with Docker Desktop, keep `--run-dir` under a Docker-shared path such as `/Users/...` (the default repository-local directory already satisfies this). A run directory under `/tmp` resolves through `/private/tmp` and may prevent Pier's agent/verifier bind-mounted logs from reaching the host.

The report includes task-macro pass@1, pass@4, paired deltas with task-cluster bootstrap intervals, exclusions, token/cost/duration/step metrics, patch hashes, context-call telemetry, and paths plus hashes for full trajectories. Pier's ATIF trajectory is preferred; the original mini-swe-agent trajectory is archived and used as the lossless fallback.
