from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import resource
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Mapping

from eval import evaluate_rankings
from prepare import ARTIFACT_DIR, TIME_BUDGET, benchmark_metadata, load_qrels_by_query, load_split_artifacts
from strategy_runner import DEFAULT_STRATEGY_PATH, StrategyRuntime


def peak_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def write_metrics(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_ranked_doc_ids(ranked_doc_ids: List[str], candidates: List[Dict[str, object]]) -> List[str]:
    candidate_doc_ids = [str(candidate["doc_id"]) for candidate in candidates]
    candidate_set = set(candidate_doc_ids)
    seen = set()
    normalized: List[str] = []

    for doc_id in ranked_doc_ids:
        if doc_id not in candidate_set:
            raise ValueError(f"Strategy returned unknown doc_id: {doc_id}")
        if doc_id in seen:
            raise ValueError(f"Strategy returned duplicate doc_id: {doc_id}")
        normalized.append(doc_id)
        seen.add(doc_id)

    if seen != candidate_set:
        missing = [doc_id for doc_id in candidate_doc_ids if doc_id not in seen]
        raise ValueError(f"Strategy omitted candidate doc_ids: {missing[:5]}")
    return normalized


def configured_parallel_devices() -> List[str]:
    raw = os.environ.get("AUTORESEARCH_RERANK_PARALLEL_DEVICES", "").strip()
    if not raw:
        return []
    devices: List[str] = []
    for token in raw.replace(";", ",").split(","):
        value = token.strip()
        if not value:
            continue
        if value.startswith("cuda"):
            devices.append(value)
        elif value.isdigit():
            devices.append(f"cuda:{value}")
        else:
            devices.append(value)
    deduped: List[str] = []
    for device in devices:
        if device not in deduped:
            deduped.append(device)
    return deduped


def strategy_runtime_envs() -> List[Dict[str, str]]:
    devices = configured_parallel_devices()
    if not devices:
        return []
    envs: List[Dict[str, str]] = []
    for index, device in enumerate(devices):
        env: Dict[str, str] = {
            "AUTORESEARCH_DISABLE_SANDBOX": "1",
            "AUTORESEARCH_RERANK_WORKER_ORDINAL": str(index),
        }
        if device.startswith("cuda:"):
            ordinal = device.split(":", 1)[1]
            env["CUDA_VISIBLE_DEVICES"] = ordinal
            env["AUTORESEARCH_RERANK_DEVICE"] = "cuda"
        else:
            env["AUTORESEARCH_RERANK_DEVICE"] = device
        envs.append(env)
    return envs


def enrich_candidates(
    *,
    query_id: str,
    candidate_map: Mapping[str, List[Dict[str, object]]],
    docs_by_id: Mapping[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    candidates = []
    for candidate in candidate_map.get(query_id, []):
        doc = docs_by_id[candidate["doc_id"]]
        enriched = dict(doc)
        enriched["retrieval_score"] = candidate["retrieval_score"]
        candidates.append(enriched)
    return candidates


def execute_query_batch(
    *,
    strategy_runtime: StrategyRuntime,
    queries: List[Dict[str, object]],
    candidate_map: Mapping[str, List[Dict[str, object]]],
    docs_by_id: Mapping[str, Dict[str, object]],
    deadline: float,
) -> Dict[str, object]:
    rankings: Dict[str, List[str]] = {}
    latencies_ms: List[float] = []
    evaluator_peak_mb = 0.0

    for query in queries:
        if time.monotonic() > deadline:
            raise TimeoutError("run exceeded budget before finishing the assigned query shard")
        query_id = str(query["query_id"])
        candidates = enrich_candidates(
            query_id=query_id,
            candidate_map=candidate_map,
            docs_by_id=docs_by_id,
        )
        started = time.perf_counter()
        ranked_doc_ids = strategy_runtime.rerank(
            str(query["text"]),
            candidates,
            ctx={"query_id": query_id},
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        rankings[query_id] = validate_ranked_doc_ids(ranked_doc_ids, candidates)
        evaluator_peak_mb = max(evaluator_peak_mb, peak_memory_mb())

    return {
        "rankings": rankings,
        "latencies_ms": latencies_ms,
        "evaluator_peak_memory_mb": evaluator_peak_mb,
        "strategy_peak_memory_mb": strategy_runtime.strategy_peak_memory_mb,
        "runtime_info": strategy_runtime.runtime_info(),
    }


def run_experiment(
    artifact_dir: Path = ARTIFACT_DIR,
    split: str = "dev",
    time_budget_seconds: float = TIME_BUDGET,
) -> Dict[str, object]:
    docs, queries, candidate_map = load_split_artifacts(artifact_dir=artifact_dir, split=split)
    qrels_by_query = load_qrels_by_query(artifact_dir=artifact_dir, split=split)
    docs_by_id = {doc["doc_id"]: doc for doc in docs}
    rankings: Dict[str, List[str]] = {}
    latencies_ms: List[float] = []

    total_started = time.perf_counter()
    deadline = time.monotonic() + time_budget_seconds
    evaluator_peak_mb = 0.0

    runtime_envs = strategy_runtime_envs()
    if len(runtime_envs) <= 1:
        runtime_env = runtime_envs[0] if runtime_envs else None
        with StrategyRuntime(strategy_path=DEFAULT_STRATEGY_PATH, env_overrides=runtime_env) as strategy_runtime:
            strategy_runtime.warmup()
            strategy = strategy_runtime.strategy_config()

            for query in queries:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"run exceeded budget of {time_budget_seconds:.1f} seconds")

                query_id = str(query["query_id"])
                candidates = enrich_candidates(
                    query_id=query_id,
                    candidate_map=candidate_map,
                    docs_by_id=docs_by_id,
                )
                started = time.perf_counter()
                ranked_doc_ids = strategy_runtime.rerank(
                    str(query["text"]),
                    candidates,
                    ctx={"query_id": query_id},
                )
                latencies_ms.append((time.perf_counter() - started) * 1000.0)
                rankings[query_id] = validate_ranked_doc_ids(ranked_doc_ids, candidates)
                evaluator_peak_mb = max(evaluator_peak_mb, peak_memory_mb())

            runtime_metadata = {
                "worker_count": 1,
                "devices": [strategy.get("device")],
                "workers": [strategy_runtime.runtime_info()],
                "sandboxed": strategy_runtime.use_sandbox,
            }
    else:
        query_shards = [queries[index::len(runtime_envs)] for index in range(len(runtime_envs))]
        with ExitStack() as stack:
            runtimes = [
                stack.enter_context(
                    StrategyRuntime(
                        strategy_path=DEFAULT_STRATEGY_PATH,
                        env_overrides=runtime_env,
                    )
                )
                for runtime_env in runtime_envs
            ]
            for runtime in runtimes:
                runtime.warmup()
            strategy = runtimes[0].strategy_config()
            worker_configs = [runtime.strategy_config() for runtime in runtimes]
            runtime_metadata = {
                "worker_count": len(runtimes),
                "devices": [config.get("device") for config in worker_configs],
                "workers": [runtime.runtime_info() for runtime in runtimes],
                "parallel_devices": configured_parallel_devices(),
                "sandboxed": any(runtime.use_sandbox for runtime in runtimes),
            }

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(runtimes)) as executor:
                futures = [
                    executor.submit(
                        execute_query_batch,
                        strategy_runtime=runtime,
                        queries=query_shard,
                        candidate_map=candidate_map,
                        docs_by_id=docs_by_id,
                        deadline=deadline,
                    )
                    for runtime, query_shard in zip(runtimes, query_shards)
                    if query_shard
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    rankings.update(result["rankings"])
                    latencies_ms.extend(result["latencies_ms"])
                    evaluator_peak_mb = max(evaluator_peak_mb, float(result["evaluator_peak_memory_mb"]))

            total_strategy_peak_mb = sum(runtime.strategy_peak_memory_mb for runtime in runtimes)
            runtime_metadata["workers"] = [runtime.runtime_info() for runtime in runtimes]

    metrics = evaluate_rankings(rankings=rankings, qrels_by_query=qrels_by_query, latencies_ms=latencies_ms)
    metrics.update(benchmark_metadata(artifact_dir=artifact_dir, split=split))
    metrics["total_seconds"] = time.perf_counter() - total_started
    metrics["num_queries"] = float(len(queries))
    if len(runtime_envs) <= 1:
        metrics["strategy_peak_memory_mb"] = strategy_runtime.strategy_peak_memory_mb
    else:
        metrics["strategy_peak_memory_mb"] = total_strategy_peak_mb
    metrics["evaluator_peak_memory_mb"] = max(evaluator_peak_mb, peak_memory_mb())
    metrics["peak_memory_mb"] = metrics["strategy_peak_memory_mb"] + metrics["evaluator_peak_memory_mb"]
    metrics["cost_usd"] = 0.0
    metrics["split"] = split
    metrics["strategy"] = strategy
    metrics["strategy_runtime"] = {
        "sandboxed": bool(runtime_metadata.get("sandboxed", False)),
        "worker": "strategy_worker.py",
        **runtime_metadata,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen evaluator against one split.")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--split", default="dev", help="Dataset split to evaluate: dev or test.")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional output path for machine-readable metrics.")
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=TIME_BUDGET,
        help="Maximum wall-clock budget for the evaluator itself.",
    )
    args = parser.parse_args()

    metrics = run_experiment(
        artifact_dir=args.artifact_dir,
        split=args.split,
        time_budget_seconds=args.time_budget_seconds,
    )
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        write_metrics(args.metrics_json, metrics)

    print("---")
    print(f"ndcg@10:         {metrics['ndcg@10']:.6f}")
    print(f"recall@100:      {metrics['recall@100']:.6f}")
    print(f"latency_p95_ms:  {metrics['latency_p95_ms']:.3f}")
    print(f"mrr@10:          {metrics['mrr@10']:.6f}")
    print(f"peak_memory_mb:  {metrics['peak_memory_mb']:.3f}")
    print(f"total_seconds:   {metrics['total_seconds']:.3f}")
    print(f"num_queries:     {int(metrics['num_queries'])}")
    print(f"candidate_k:     {int(metrics['strategy']['candidate_k'])}")
    print(f"split:           {metrics['split']}")


if __name__ == "__main__":
    main()
