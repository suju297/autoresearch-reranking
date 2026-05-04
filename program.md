# Autoresearch for Reranking

This repo follows the small-repo pattern from `karpathy/autoresearch`, adapted to offline reranking experiments and grounded in the local clone under `upstream-autoresearch/`.

## Setup

1. Read the in-scope files:
   - `README.md`
   - `docs/reranking_playbook.md`
   - `docs/metric-policy.md`
   - `local_brain.py`
   - `prepare.py`
   - `rerank_strategy.py`
   - `train.py`
   - `run.py`
2. Verify that the fast benchmark exists under `artifacts/fast/fiqa-dev/`. If not, prepare it explicitly.
3. Verify that `results.tsv` is either absent or untracked by git.
4. The first experiment must always be the baseline.
5. Prefer using `autoresearch_driver.py` for branch setup, prompt packaging, one-shot run handling, and local autonomous loops.

## Trust Boundary

You may edit only `rerank_strategy.py`.

You may not edit:

- `prepare.py`
- `eval.py`
- `train.py`
- `run.py`
- `local_brain.py`
- `autoresearch_driver.py`
- dataset files
- metrics
- dependency definitions

## Goal

Improve `ndcg@10` without violating guardrails on the frozen dev split:

- keep `recall@100` within policy bounds
- keep `latency_p95_ms` within policy bounds
- keep `peak_memory_mb` within policy bounds
- keep `cost_usd` within policy bounds
- use `mrr@10` for diagnosis only
- prefer simpler code when gains are marginal

## Experiment Loop

1. Inspect the current git state if the repo has been initialized.
2. Make one small change in `rerank_strategy.py`, either manually or through `uv run autoresearch-reranking loop 1`.
3. Commit the change if git is available.
4. Run `uv run autoresearch-reranking once`, or let `uv run autoresearch-reranking loop <n>` do it.
5. Review the summary block and `results.tsv`.
6. Open `history/index.html` if you want a graphable view of loop outcomes, recent iterations, and family usage.
6. A qualifying keep candidate must survive two more repeated dev runs before final acceptance.
7. Run a strategic review after 5 completed loops, then every 10 loops after that. Use the review to steer the next search region, not to bypass the frozen evaluator.
8. Use exactly one change family per proposal. The current family taxonomy is:
   - `candidate_k`
   - `score_normalization`
   - `fusion_weight`
   - `metadata_boost`
   - `dedup_filter`
   - `truncation_policy`
   - `pre_rerank_filter`
   - `query_type_heuristic`
9. Use staged search. First explore each active family at least once while underexplored families remain. After first-pass coverage, downselect to the best 1 to 2 families by measured fast and promotion signal and spend subsequent loops tuning only those families. The same family may not run back-to-back, and a family with 3 discards in the last 5 attempts enters cooldown for 5 runs.
10. Every proposal must declare:
   - `family`
   - `hypothesis`
   - `changed_keys`
   - `why_not_duplicate`
   - expected directions for `ndcg@10`, `recall@100`, and `latency_p95_ms`
   - `promotion_risk`
11. The controller must use a proposer/reviewer split:
   - proposer: generate 3 to 4 typed candidate ideas from different families
   - reviewer: select one idea based on novelty, plausibility, and promotion-benchmark fit
12. Reject a proposal before execution if it repeats a recent failed family with the same mechanism, changed keys, rationale fingerprint, or expected tradeoff profile.
13. The fast benchmark is frozen `beir/fiqa/dev` under `artifacts/fast/fiqa-dev/`.
14. If the promotion benchmark is configured, treat full `beir/scifact/test` as the validation gate after repeated dev evaluation.
15. Use `beir/fiqa/test` only for held-out reporting, never for tuning.
16. Use a fixed 10-minute fast-loop budget on `beir/fiqa/dev` and a fixed 25-minute budget on promotion or report checks.
17. The reranker checkpoint is fixed to `BAAI/bge-reranker-base`. Do not propose model swaps in this phase.
18. Keep the change only if the final summary row is `keep`.
19. If the run crashes or regresses, revert to the last kept commit when git is available.
20. Use the local strategy library in `docs/reranking_playbook.md` plus recent run history; do not use unrestricted live web access inside the inner loop.

## Autonomous Brain Rule

When using `autoresearch_driver.py auto-loop`, the local controller may read the frozen harness state, but it must still propose only a new `rerank_strategy.py`. The controller is the local `llama_cpp` GGUF backend. The loop may keep a lightweight controller warm across iterations when configured, but it must still avoid overlapping large controller and evaluator residency on the accelerator. Evaluation runs the strategy inside a sandboxed worker with offline cache access only.

Every completed `run-once` and `auto-loop` call must refresh the generated history bundle under `history/`. Treat `history/` as untracked generated output, not part of the editable search surface.

## Baseline Rule

The first run is always the current unmodified strategy in `rerank_strategy.py`. Log it as `baseline`.

## Simplicity Rule

If two strategies are effectively tied, keep the simpler one.
