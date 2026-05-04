# Metric Policy

## Goal

Improve reranking quality through controlled experiments on a frozen benchmark and frozen evaluation harness. Optimize one primary metric only. Use guardrails to reject misleading wins.

The active fast-loop benchmark is frozen `beir/fiqa/dev`. It is the main tuning surface for the current BGE-base experiment.

## Primary Metric

- `ndcg@10` on frozen `beir/fiqa/dev`

## Guardrails

- `recall@100`
- `latency_p95_ms`
- `peak_memory_mb`
- `cost_usd`
- run validity
- `mrr@10` is diagnostic only

## Keep Rule

A candidate qualifies for keep only if all of the following are true:

- `ndcg@10` improves by at least `0.002`
- `recall@100` does not regress by more than `1%` relative
- `latency_p95_ms` does not regress by more than `20%` relative
- `peak_memory_mb` does not regress by more than `20%` relative
- `cost_usd` does not increase
- the run completes without crash, timeout, malformed ranking output, or missing metrics

## Recheck Rule

If a run qualifies for keep on the fast benchmark, rerun it two more times before final acceptance.

Final acceptance requires:

- mean `ndcg@10` across the repeated runs still clears the `0.002` gain threshold
- no repeated run violates the recall guardrail
- no repeated run violates the latency guardrail
- no repeated run violates the peak-memory guardrail
- no repeated run violates the cost guardrail
- no repeated run crashes

If repeated evaluation fails, the candidate is downgraded to `discard`.

## Dataset Policy

Use two evaluation tiers:

1. Fast iteration benchmark
   - default: `beir/fiqa/dev`
   - optimize only on this frozen dev benchmark
   - fixed wall-clock budget: 10 minutes per run
2. Promotion benchmark
   - default recommendation: full `beir/scifact/test` in `artifacts/promotion`
   - fixed wall-clock budget: 25 minutes per run
   - run only after a candidate survives repeated evaluation on the fast benchmark
3. Report benchmark
   - default recommendation: `beir/fiqa/test` in `artifacts/report`
   - fixed wall-clock budget: 25 minutes per run
   - use only for held-out reporting, never for tuning
   - do not optimize against it

Do not optimize against `test`. Use `test` only for milestone checks and reporting.

## Frozen Boundary

The autoresearch agent may edit only:

- `rerank_strategy.py`

The agent may not edit:

- `prepare.py`
- `eval.py`
- `train.py`
- `run.py`
- dataset artifacts
- benchmark manifests
- `results.tsv`
- decision thresholds unless a human explicitly changes policy

## Logging

Every accepted trial should leave:

- attempt-level run artifacts under `runs/<run-id>/`
- a summary row in `results.tsv`
- benchmark identity fields so results are only compared within the same frozen benchmark version
- harness version fields so sandboxed runs are not compared against legacy in-process runs

## Operating Principle

Optimize `ndcg@10` on frozen dev, reject wins that hurt recall, latency, peak memory, or cost, and do not trust a change until it survives repeated runs and, when configured, a larger frozen benchmark check.
