# Autoresearch Loop Analysis

This analysis uses the generated history files from the local run:

- `history/results-summary.csv`: 101 benchmark result rows
- `history/iterations.csv`: 91 autonomous loop iterations

## Public Comparison Boundary

The public-facing comparison should use the BEIR-derived FiQA and SciFact rows only. Internal smoke-test presets are useful for debugging the controller, but they should not be shown as market comparisons.

The closest public benchmark family for this repo is [BEIR](https://arxiv.org/abs/2104.08663):

- `fast-fiqa-dev`: `beir/fiqa/dev`
- `promotion-scifact`: `beir/scifact/test`
- `report-fiqa-test`: `beir/fiqa/test`

[MTEB](https://huggingface.co/spaces/mteb/leaderboard) is useful context, but it is not a direct apples-to-apples comparison:

- MTEB Retrieval includes `FiQA2018` and `SciFact`, but those are retriever leaderboard scores, not this repo's BM25-plus-cross-encoder reranking setup.
- MTEB Reranking uses `AskUbuntuDupQuestions`, `MindSmallReranking`, `SciDocsRR`, and `StackOverflowDupQuestions`, which are not the datasets used by this harness.
- BGE's own [BEIR reranker tutorial](https://bge-model.com/tutorial/5_Reranking/5.3.html) is closer because it evaluates rerankers on BEIR. It reports `bge-reranker-v2-m3` at `0.44828` `ndcg@10` on `fiqa-test` when reranking candidates from `bge-large-en-v1.5`, versus this repo's kept `0.252578` on `report-fiqa-test` with BM25 candidates and `BAAI/bge-reranker-base`.

That gap should not be claimed as a direct loss because the first-stage retriever and rerank depth differ, but it is a strong signal that the harness needs to test newer rerankers and stronger candidate generation before claiming competitive progress.

## Did It Improve?

Accepted progress did not beat the public BGE-base checkpoints.

| Benchmark | Best kept `ndcg@10` | Best raw `ndcg@10` | Raw lift | Accepted? |
|---|---:|---:|---:|---|
| `fast-fiqa-dev` | 0.262157 | 0.278969 | +6.4% | No |
| `promotion-scifact` | 0.661389 | 0.679487 | +2.7% | No |
| `report-fiqa-test` | 0.252578 | 0.252578 | 0.0% | Baseline only |

The raw movement was real enough to keep investigating, but it was not stable enough to count as a kept improvement.

## Where The Loop Wasted Budget

Out of 91 autonomous iterations:

| Outcome | Count | Meaning |
|---|---:|---|
| `discard` | 43 | Strategy ran but failed keep criteria |
| `brain_error` | 41 | Controller output was invalid or blocked before evaluation |
| `crash` | 7 | Evaluation failed |
| `keep` | 0 | No autonomous accepted improvement |

The biggest waste was not simply "too few iterations." It was low effective iteration quality:

- 41 of 91 iterations never produced a valid evaluated strategy.
- 41 of 101 result rows were non-public or smoke-test rows.
- Only 24 autonomous iterations actually evaluated on `fast-fiqa-dev`.
- The strongest families were identified, but too many loops repeated nearby shapes or failed proposal formatting.

## What Stopped Raw Wins From Being Kept

The strongest public raw result was `shorter-truncation-policy`:

- `fast-fiqa-dev`: `0.278969`, up from `0.262157`
- `promotion-scifact`: `0.679487`, up from `0.661389`

It still failed the keep policy because resource guardrails matter, not just `ndcg@10`. From the summary rows, this candidate raised peak memory far beyond the 20% memory guardrail:

- FiQA dev memory: baseline `1639.094 MB`, candidate `2728.484 MB`
- SciFact memory: baseline `1380.891 MB`, candidate `2519.828 MB`

Other raw winners showed a similar pattern: useful `ndcg@10` movement, then rejection by memory, latency, repeat, or promotion checks.

## Were The Parameters Right?

Partly.

Good signal:

- `candidate_k` produced repeated FiQA dev gains near `0.276227`.
- `truncation_policy` produced the best public raw gains.
- `query_type_heuristic` had smaller but interpretable movement.

Weak or overused signal:

- `fusion_weight` and `score_normalization` mostly became score-massage loops.
- `metadata_boost` repeated the same candidate-depth-shaped result rather than proving metadata helped.
- `dedup_filter` and `pre_rerank_filter` crashed often enough that they were disabled for autonomous search.

Missing signal:

- The loop could not safely test `model_name`, so it could not compare the fixed `BAAI/bge-reranker-base` checkpoint against current open rerankers.
- The loop mostly tuned reranking around BM25 candidates instead of testing whether candidate generation or rerank depth was the real ceiling.

## Did The Playbook Limit Creativity?

Yes, in a specific way.

The playbook helped prevent uncontrolled edits to the evaluator, datasets, and metrics. That part should stay. But it made the search surface too narrow: the controller could only tune small strategy knobs around one fixed reranker. That encouraged shallow changes such as score normalization, fusion weights, and small candidate-depth changes.

The fix is not to remove the harness. The fix is to separate creativity from execution:

- Keep the frozen evaluator and guardrails.
- Let the idea stage propose broader families.
- Only execute ideas that map to an allowlisted, reproducible strategy surface.
- Add `reranker_model` as an explicit family so checkpoint swaps are visible, comparable, and validated.
- Keep model swaps one-at-a-time so a win can be attributed to the model, not hidden mixed changes.

## Harness Changes Made

- Added `reranker_model` as a proposal family.
- Added bounded strategy keys: `model_name`, `model_batch_size`, and `model_max_length`.
- Added an allowlist for public rerankers that can be tested under the same frozen artifacts.
- Removed internal smoke presets from the public prepare preset list.
- Updated public plots to exclude internal smoke-test benchmark rows.

## Next Valid Experiment

Run the same public artifacts with one model change at a time:

1. Baseline: `BAAI/bge-reranker-base`
2. Candidate: `BAAI/bge-reranker-v2-m3`
3. Optional candidate: `Qwen/Qwen3-Reranker-0.6B`

Keep `candidate_k`, datasets, qrels, and metric policy fixed for that comparison. If model swap improves `ndcg@10` but fails memory or latency, record it as a raw quality win and then tune batch size or rerank depth in a separate run.
