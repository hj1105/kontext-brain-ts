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
  "runtimeProvider": "codex",
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
      "sourceSpan": "State data scope",
      "source": {
        "connectorId": "filesystem",
        "externalId": "file:///independent/specs/state-data.md",
        "type": "markdown"
      },
      "observedAt": "2026-09-02T00:00:00.000Z",
      "contentSha256": "<SHA-256 of text>",
      "ontologyNodeIds": ["resource:state-data-design"],
      "allowedRuntimeProviders": ["codex"]
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
          "allowedRuntimeProviders": ["codex"]
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

The loader rejects future-dated evidence, hash mismatches, missing evidence closure, runtime-provider egress mismatches, duplicate IDs, corpus files inside the benchmark tree, and provenance paths containing task tests, verifier, solution, trajectory, result, or agent artifacts. The preparation step also requires the frozen `runtimeProvider` to match the provider prefix in `--model`. These structural checks do not prove authorship independence; preregistered corpora still require external review.

## Export from the sidecar

Prepare the benchmark Task with the normal `kontext_prepare_task` flow, then export its immutable Task Context Snapshot. The sidecar Evidence must carry its original Resource, Chunk, connector/external source identity, observation time, content hash, and Ontology Node IDs. Export fails if the snapshot is stale, conflicted, inaccessible to the selected runtime provider, missing exact Evidence closure, or lacks provenance. Empty exports require the explicit infrastructure-only flag.

```bash
pnpm --filter @kontext-brain/bench code-quality:deepswe:export -- \
  --task-id python-statemachine-state-data-scoping \
  --organization-id organization:personal \
  --runtime-provider codex \
  --output /absolute/path/to/frozen-corpora/python-statemachine-state-data-scoping.json
```

The exporter records the clean Kontext Git revision automatically. `--generator-revision` is available for an installed, externally pinned build, and `--data-dir` selects a non-default sidecar directory.

## Preregistered Awilix pilot

The `awilix-async-container-initialization` pilot exercises the complete production-shaped path: the pinned public Awilix checkout is synchronized as Code and documentation Resources, exact line ranges become Chunks and Evidence, shared Ontology Nodes derive Planned Symbol governance links, the sidecar prepares a Task Context Snapshot, and the exporter freezes it. The source manifest pins every input file hash. It does not read DeepSWE tests, verifier files, solutions, or earlier trajectories.

```bash
git clone https://github.com/jeffijoe/awilix.git /absolute/path/to/awilix
git -C /absolute/path/to/awilix switch --detach 82ac179c1de4c216c4e333093044fac643303f0c

pnpm --filter @kontext-brain/bench code-quality:deepswe:pilot:awilix -- \
  --checkout /absolute/path/to/awilix \
  --runtime-provider codex \
  --output /absolute/path/to/frozen-corpora/awilix-async-container-initialization.json
```

This pilot intentionally uses facts already present in the base checkout. It measures whether ontology-linked, provenance-governed compression helps an agent apply existing project contracts; it does not claim access to otherwise unavailable information. A one-rollout run is infrastructure and directional evidence only. Use the default four rollouts per arm for the preregistered comparison.

## Reproducible subscription run

The default runtime is the open-source Codex CLI authenticated through the user's ChatGPT subscription. The runner refuses any API credential file, strips provider API-key environment variables before starting Pier, verifies that `codex login status` reports ChatGPT login, uploads the local auth cache only to the ephemeral agent container, and removes it before verification. Pin Codex, Pier, and DeepSWE revisions for replayability.

For Docker runs, the runner derives one fixed worker image identity from the task base image, Codex version, Pier version, and build-recipe version. It builds a labeled image only on a cache miss and otherwise requires every identity label to match before reusing the exact local image ID. Tasks with different base images are split into separate Pier jobs, while every arm for the same base uses the same pinned worker image. This removes repeated Codex installation without sharing task workspaces, auth caches, context bundles, or verifier state.

```bash
pnpm --filter @kontext-brain/bench code-quality:deepswe -- \
  --dataset /absolute/path/to/deep-swe/tasks \
  --corpus /absolute/path/to/frozen-corpora \
  --runtime codex-subscription \
  --codex-version 0.144.6 \
  --pier-revision 0.3.1 \
  --deepswe-revision 0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea \
  --model gpt-5.5 \
  --sample-seed 0 \
  --task-limit 10 \
  --attempts 4 \
  --environment docker
```

Add `--dry-run` to validate the corpus and write private Pier/context manifests without starting containers or model calls. Scored runs refuse a dirty Kontext checkout. Subscription reports retain token telemetry but omit API-equivalent dollar estimates because those values are not usage-based API charges.

The legacy mini-swe-agent route is intentionally opt-in and cannot silently load `.env.local`. It requires all of `--runtime mini-swe-api`, `--mini-swe-version <version>`, and `--allow-api-billing`; `--env-file` is accepted only on that route.

On macOS with Docker Desktop, keep `--run-dir` under a Docker-shared path such as `/Users/...` (the default repository-local directory already satisfies this). A run directory under `/tmp` resolves through `/private/tmp` and may prevent Pier's agent/verifier bind-mounted logs from reaching the host.

The report includes task-macro pass@1, pass@4, paired deltas with task-cluster bootstrap intervals, exclusions, token/duration/step metrics, patch hashes, context-call telemetry, and paths plus hashes for full trajectories. API runs additionally include cost; subscription runs do not. Pier's ATIF trajectory is preferred; the original mini-swe-agent trajectory is used only as the API-runner fallback.
