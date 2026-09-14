# DeepSWE methodology review for Kontext evaluation

Date: 2026-09-03

## Decision

Adopt the DeepSWE v1.1 task, isolation, verifier, sampling, and reporting protocol as the next external validity layer for Kontext. Do not present a Kontext experiment as an official DeepSWE leaderboard score unless it uses the official unmodified agent configuration. The intended experiment changes the context treatment, so it should be reported as a **DeepSWE-based paired A/B evaluation**.

“DeepSWE” here means Datacurve's 2026 benchmark, not the 2025 DeepSWE-Preview model from Agentica and Together AI.

## Why this is a better test

DeepSWE contains 113 original, long-horizon tasks from 91 active open-source repositories in TypeScript, Go, Python, JavaScript, and Rust. Tasks were authored rather than mined from merged fixes and were not merged upstream, reducing the public-answer contamination present in SWE-bench-style tasks. Each task pins an immutable base commit. ([paper §§3.2–3.3](https://arxiv.org/html/2607.07946v1#S3.SS2), [official repository](https://github.com/datacurve-ai/deep-swe))

Its hand-written functional verifiers exercise public APIs and observable behavior and are intended to accept any correct implementation, rather than require the reference patch's internal shape. The reference solution is used for review, not grading. Verifiers also run selected existing regression tests. ([paper §§3.4, 4.2](https://arxiv.org/html/2607.07946v1#S3.SS4), [official README](https://github.com/datacurve-ai/deep-swe/blob/main/README.md))

Each verifier is run three times during task authoring to detect flakiness. Tasks receive LLM-assisted analysis and independent human review for prompt-verifier bijection, acceptance breadth, realism, and environment cleanliness. Multiple frontier agents are also used diagnostically during review. ([paper §§4.2–4.3](https://arxiv.org/html/2607.07946v1#S4.SS2))

Version 1.1 uses a separate verifier environment: the agent commits its changes, a patch is collected, and the patch is applied and graded in a pristine container. The agent must not see the verifier or reference solution. ([official README](https://github.com/datacurve-ai/deep-swe/blob/main/README.md))

These properties directly address the two largest limitations of the current Django smoke evaluation: its target came from a historical public fix, and the Kontext target-symbol plan was derived from the merged change shape.

## Protocol to copy exactly

1. Pin the DeepSWE repository revision, Pier version, task definition, base commit, container digest, model snapshot, reasoning effort, agent configuration, and Kontext revision for every run. DeepSWE itself pins repository commits and its paper pinned mini-swe-agent to commit `adfe2023`, released as v2.3.0. ([paper §§4.1, 5.2](https://arxiv.org/html/2607.07946v1#S5.SS2), [mini-swe-agent v2.3.0 release](https://github.com/SWE-agent/mini-swe-agent/releases/tag/v2.3.0))
2. Use Pier's isolated agent and separate verifier environments. Do not mount or index `tests/`, `solution/`, the DeepSWE checkout, hidden grader output, or upstream Git history in an agent workspace. DeepSWE uses a shallow clone at the base commit to avoid history leakage. ([paper §§3.3, 5.2](https://arxiv.org/html/2607.07946v1#S3.SS3), [sample v1.1 task](https://github.com/datacurve-ai/deep-swe/blob/main/tasks/arktype-json-schema-refs-dependencies/task.toml))
3. Keep the model, reasoning effort, seed policy, task prompt, tool surface, timeout, and arm order policy fixed. Only the context treatment may differ.
4. Run approximately four independent rollouts per task and arm. DeepSWE reports task-macro-averaged pass@1 and the fraction solved at least once in up to four rollouts as pass@4. ([paper §§5.3–5.4](https://arxiv.org/html/2607.07946v1#S5.SS3))
5. Count context-window exhaustion and agent timeout as failures. Exclude provider, transient network, and verifier infrastructure errors from numerator and denominator and publish every exclusion; do not silently retry until success. ([paper §5.6](https://arxiv.org/html/2607.07946v1#S5.SS6))
6. Report pass@1, pass@4, 95% uncertainty, output tokens, wall-clock duration, dollar cost, agent steps, and per-task trajectories. DeepSWE reports cost-shaped measures alongside accuracy and releases trajectories. ([paper §§5.5, 6.2](https://arxiv.org/html/2607.07946v1#S5.SS5), [trajectory site](https://deepswe.datacurve.ai/))
7. Before admitting a task to the Kontext comparison, verify that its reference solution passes, its base state fails the new behavior, and the verifier is deterministic across at least three clean runs.

The current repository task files are authoritative for operational limits. For example, the sampled v1.1 task declares a 10,800-second agent timeout and a 1,800-second separate verifier timeout, while the May 2026 paper describes a 9,000-second rollout limit. Pinning the exact task revision prevents silently mixing protocols. ([sample v1.1 task](https://github.com/datacurve-ai/deep-swe/blob/main/tasks/arktype-json-schema-refs-dependencies/task.toml), [paper §5.2](https://arxiv.org/html/2607.07946v1#S5.SS2))

## Kontext experiment design

Run three matched arms for the same task, model, and rollout index:

| Arm | Available context service | Purpose |
| --- | --- | --- |
| baseline | identical command/tool surface returning no supplemental organizational context | Native repository reasoning control |
| raw RAG | same pre-task corpus, retrieved without ontology governance | Retrieval control |
| Kontext | same pre-task corpus, governed by provenance, recency, decisions, domain terms, and symbol bindings | Treatment |

All arms should receive the same task prompt and see the same repository files. The raw RAG and Kontext arms must use the exact same source corpus and source snapshot. Kontext may rank, reconcile, and structure that corpus but must not receive information unavailable to raw RAG.

The baseline should still receive the same tool declaration and instruction budget, with the service returning an explicit empty result. This reduces prompt and tool-surface confounding.

Do not predeclare behavior-bearing targets from the hidden reference patch. Kontext must discover the relevant symbols from the prompt, permitted pre-task context, and base repository. A `begin_logic` binding should be created when the agent selects or creates a behavior-bearing symbol, not from a gold-derived manifest.

The allowed Kontext corpus is limited to artifacts that could genuinely exist before implementation:

- base-repository code and documentation;
- pre-task local/session decisions with timestamps and provenance;
- public upstream information dated no later than the pinned task snapshot, if the experiment explicitly allows internet-derived organizational context;
- task-author-provided decision history created without viewing the reference implementation or verifier internals.

Every corpus item should be hashed and recorded. The solution patch, verifier implementation, verifier results, diagnostic trajectories, later public discussions, and any ontology records derived from them are prohibited.

## Two scores, not one

### Scientific context-effect score

Use the official fixed mini-swe-agent scaffold and expose all three arms through an identical bash-callable context adapter. This isolates the effect of Kontext's context treatment. Because the scaffold is modified by adding a context service, label the result “DeepSWE-based”, not an official leaderboard score. DeepSWE deliberately fixes mini-swe-agent because scaffold choice is a confound. ([paper §5.2](https://arxiv.org/html/2607.07946v1#S5.SS2))

### Product-effect score

Run baseline Codex CLI and Codex CLI + Kontext plugin/MCP under otherwise pinned settings. This measures the actual product users receive, including native editing tools and orchestration. It must be reported separately because DeepSWE explicitly says its fixed-harness leaderboard is not a ranking of native coding products. ([paper scope and §5.2](https://arxiv.org/html/2607.07946v1#S1))

## Statistics

The primary estimand is paired improvement in per-task pass@1:

`delta = macro_pass_at_1(Kontext) - macro_pass_at_1(baseline)`

Also report `Kontext - raw RAG`, pass@4 deltas, regression-failure rate, and efficiency deltas. Use a paired cluster bootstrap over tasks, keeping all arms and rollouts for a sampled task together. Also publish the simple DeepSWE run-to-run interval for comparability, but do not rely on it alone: the paper notes that four-run intervals omit task-sampling uncertainty and can be too narrow. ([paper §5.5](https://arxiv.org/html/2607.07946v1#S5.SS5))

Pre-register the task subset, sample seed, arm order, model configuration, exclusion rules, and analysis code before running. Report per-task paired outcomes so gains cannot be hidden by aggregate scores.

## Recommended rollout

1. **Infrastructure proof:** two tasks, one rollout, all three arms. Validate isolation, patch transfer, hidden-verifier exclusion, telemetry, and failure classification.
2. **Preregistered pilot:** deterministic 10-task subset (`sample-seed = 0`), four rollouts, three arms: 120 total rollouts. Use the result to validate variance and estimate full-run cost, not to claim superiority.
3. **Confirmatory run:** all 113 tasks, four rollouts, three arms: 1,356 nominal rollouts. Freeze code and analysis before starting.
4. **Native product replication:** only after the fixed-scaffold result, repeat a smaller preregistered slice with Codex CLI and Codex CLI + Kontext.

## Acceptance gate for the next implementation PR

- A pinned DeepSWE/Pier adapter can run one task end-to-end without exposing verifier or solution files.
- Baseline, raw RAG, and Kontext share the same task image, prompt, model configuration, timeout, and context-tool surface.
- Raw RAG and Kontext corpora are byte-identical before transformation and are provenance-hashed.
- Gold-derived symbols or decisions cannot enter the Kontext index.
- Results include pass/fail, exclusion reason, tokens, cost, duration, steps, patch hash, corpus hash, tool calls, and a full trajectory (Pier ATIF when available, original mini-swe-agent trajectory as fallback).
- Analysis computes task-macro pass@1, pass@4, paired deltas, and task-clustered intervals.
- A failed public regression or hidden verifier makes the task fail; context consultation compliance remains a diagnostic, not a substitute for functional correctness.

## Limitations retained from DeepSWE

DeepSWE uses binary task reward and does not measure code quality, maintainability, security, or partial progress when the verifier does not encode them. It also under-represents short edits, bug localization, ambiguous user requests, and non-coding work. A fixed harness improves internal validity while reducing similarity to tuned production agents. The benchmark authors explicitly scope leaderboard positions to this measurement rather than overall model or product quality. ([paper §§1, 8](https://arxiv.org/html/2607.07946v1#S8))

Kontext should therefore retain secondary review metrics for evidence correctness, terminology consistency, stale-decision avoidance, and unnecessary-change surface, but these must be reported separately from the DeepSWE functional score.
