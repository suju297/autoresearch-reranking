from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping

from prepare import TIME_BUDGET as DEFAULT_BUDGET_SECONDS
from prepare import HARNESS_VERSION as CURRENT_HARNESS_VERSION
from prepare import benchmark_metadata

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "current"
DEFAULT_RESULTS_PATH = ROOT / "results.tsv"
DEFAULT_RUNS_DIR = ROOT / "runs"
LEGACY_HARNESS_VERSION = "inprocess-strategy-v1"
RESULTS_COLUMNS = [
    "timestamp",
    "record_type",
    "run_id",
    "parent_commit",
    "commit",
    "benchmark_name",
    "benchmark_manifest_hash",
    "candidate_generation_version",
    "harness_version",
    "split",
    "ndcg@10",
    "recall@100",
    "latency_p95_ms",
    "mrr@10",
    "peak_memory_mb",
    "cost_usd",
    "wall_clock_s",
    "status",
    "decision_reason",
    "description",
]
RESULTS_HEADER = "\t".join(RESULTS_COLUMNS) + "\n"
METRIC_FIELDS = {
    "ndcg@10",
    "recall@100",
    "latency_p95_ms",
    "mrr@10",
    "peak_memory_mb",
    "total_seconds",
}


def current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "nogit"
    return result.stdout.strip() or "nogit"


def infer_harness_version(metrics: Mapping[str, object] | None) -> str:
    if metrics is None:
        return LEGACY_HARNESS_VERSION
    strategy_runtime = metrics.get("strategy_runtime")
    if isinstance(strategy_runtime, Mapping) and strategy_runtime.get("sandboxed") is True:
        return CURRENT_HARNESS_VERSION
    value = metrics.get("harness_version")
    if value:
        return str(value)
    return LEGACY_HARNESS_VERSION


def metrics_path_for_run(run_id: str) -> Path:
    return DEFAULT_RUNS_DIR / run_id / "metrics.json"


def load_run_metrics(run_id: str) -> Dict[str, object] | None:
    if not run_id:
        return None
    metrics_path = metrics_path_for_run(run_id)
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def migrate_results_schema(results_path: Path) -> None:
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n")
        if header == RESULTS_HEADER.rstrip("\n"):
            return
        handle.seek(0)
        rows = list(csv.DictReader(handle, delimiter="\t"))

    migrated_rows: List[Dict[str, object]] = []
    for row in rows:
        metrics = load_run_metrics(row.get("run_id", ""))
        migrated_rows.append(
            {
                "timestamp": row.get("timestamp", ""),
                "record_type": row.get("record_type", "summary"),
                "run_id": row.get("run_id", ""),
                "parent_commit": row.get("parent_commit", ""),
                "commit": row.get("commit", ""),
                "benchmark_name": row.get("benchmark_name", ""),
                "benchmark_manifest_hash": row.get("benchmark_manifest_hash", ""),
                "candidate_generation_version": row.get("candidate_generation_version", ""),
                "harness_version": row.get("harness_version", "") or infer_harness_version(metrics),
                "split": row.get("split", ""),
                "ndcg@10": row.get("ndcg@10", ""),
                "recall@100": row.get("recall@100", ""),
                "latency_p95_ms": row.get("latency_p95_ms", ""),
                "mrr@10": row.get("mrr@10", ""),
                "peak_memory_mb": row.get("peak_memory_mb", "")
                or f"{float((metrics or {}).get('peak_memory_mb', 0.0)):.3f}",
                "cost_usd": row.get("cost_usd", "") or f"{float((metrics or {}).get('cost_usd', 0.0)):.6f}",
                "wall_clock_s": row.get("wall_clock_s", ""),
                "status": row.get("status", ""),
                "decision_reason": row.get("decision_reason", ""),
                "description": row.get("description", ""),
            }
        )

    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(RESULTS_COLUMNS)
        for row in migrated_rows:
            writer.writerow([row.get(column, "") for column in RESULTS_COLUMNS])


def ensure_results_schema(results_path: Path) -> None:
    if not results_path.exists():
        results_path.write_text(RESULTS_HEADER, encoding="utf-8")
        return
    migrate_results_schema(results_path)


def read_results(results_path: Path) -> List[Dict[str, str]]:
    ensure_results_schema(results_path)
    if not results_path.exists():
        return []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_result(results_path: Path, row: Mapping[str, object]) -> None:
    ensure_results_schema(results_path)
    with results_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([row.get(column, "") for column in RESULTS_COLUMNS])


def row_metric(row: Mapping[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def benchmark_identity(artifact_dir: Path, split: str) -> Dict[str, str]:
    metadata = benchmark_metadata(artifact_dir=artifact_dir, split=split)
    return {
        "benchmark_name": str(metadata["benchmark_name"]),
        "benchmark_manifest_hash": str(metadata["benchmark_manifest_hash"]),
        "candidate_generation_version": str(metadata["candidate_generation_version"]),
        "harness_version": str(metadata["harness_version"]),
        "split": str(metadata["split"]),
    }


def last_keep_result(
    results_path: Path,
    *,
    benchmark_name: str | None = None,
    benchmark_manifest_hash: str | None = None,
    harness_version: str | None = None,
    split: str | None = None,
) -> Dict[str, str] | None:
    keeps = []
    for row in read_results(results_path):
        if row.get("status") != "keep":
            continue
        if row.get("record_type", "summary") != "summary":
            continue
        if benchmark_name is not None and row.get("benchmark_name") != benchmark_name:
            continue
        if benchmark_manifest_hash is not None and row.get("benchmark_manifest_hash") != benchmark_manifest_hash:
            continue
        if harness_version is not None and row.get("harness_version") != harness_version:
            continue
        if split is not None and row.get("split") != split:
            continue
        keeps.append(row)
    if not keeps:
        return None
    return keeps[-1]


def parse_summary(log_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in METRIC_FIELDS:
            metrics[key] = float(value)
        elif key in {"num_queries", "candidate_k"}:
            metrics[key] = float(value)
    return metrics


def load_metrics(metrics_path: Path, log_path: Path) -> Dict[str, object]:
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return parse_summary(log_path)


def build_decision(
    metrics: Mapping[str, object],
    previous_keep: Mapping[str, str] | None,
    min_ndcg_gain: float,
    max_recall_rel_drop: float,
    max_latency_rel_increase: float,
    max_peak_memory_rel_increase: float,
    max_cost_usd_increase: float,
) -> Dict[str, object]:
    current_ndcg = float(metrics["ndcg@10"])
    current_recall = float(metrics["recall@100"])
    current_latency = float(metrics["latency_p95_ms"])
    current_peak_memory = float(metrics.get("peak_memory_mb", 0.0))
    current_cost = float(metrics.get("cost_usd", 0.0))

    if previous_keep is None:
        return {
            "status": "keep",
            "reason": "baseline run",
            "guardrails": {},
            "deltas": {},
        }

    previous_ndcg = row_metric(previous_keep, "ndcg@10") or 0.0
    previous_recall = row_metric(previous_keep, "recall@100") or 0.0
    previous_latency = row_metric(previous_keep, "latency_p95_ms") or 0.0
    previous_peak_memory = row_metric(previous_keep, "peak_memory_mb") or 0.0
    previous_cost = row_metric(previous_keep, "cost_usd") or 0.0
    ndcg_gain = current_ndcg - previous_ndcg

    recall_rel_drop = 0.0
    if previous_recall > 0.0 and current_recall < previous_recall:
        recall_rel_drop = (previous_recall - current_recall) / previous_recall

    latency_rel_increase = 0.0
    if previous_latency > 0.0 and current_latency > previous_latency:
        latency_rel_increase = (current_latency - previous_latency) / previous_latency

    peak_memory_rel_increase = 0.0
    if previous_peak_memory > 0.0 and current_peak_memory > previous_peak_memory:
        peak_memory_rel_increase = (current_peak_memory - previous_peak_memory) / previous_peak_memory

    cost_usd_increase = max(0.0, current_cost - previous_cost)

    guardrails = {
        "ndcg_gain": {
            "passed": ndcg_gain >= min_ndcg_gain,
            "current": current_ndcg,
            "previous_keep": previous_ndcg,
            "delta": ndcg_gain,
            "minimum_required": min_ndcg_gain,
        },
        "recall": {
            "passed": recall_rel_drop <= max_recall_rel_drop,
            "current": current_recall,
            "previous_keep": previous_recall,
            "relative_drop": recall_rel_drop,
            "max_relative_drop": max_recall_rel_drop,
        },
        "latency": {
            "passed": latency_rel_increase <= max_latency_rel_increase,
            "current": current_latency,
            "previous_keep": previous_latency,
            "relative_increase": latency_rel_increase,
            "max_relative_increase": max_latency_rel_increase,
        },
        "peak_memory": {
            "passed": peak_memory_rel_increase <= max_peak_memory_rel_increase,
            "current": current_peak_memory,
            "previous_keep": previous_peak_memory,
            "relative_increase": peak_memory_rel_increase,
            "max_relative_increase": max_peak_memory_rel_increase,
        },
        "cost_usd": {
            "passed": cost_usd_increase <= max_cost_usd_increase,
            "current": current_cost,
            "previous_keep": previous_cost,
            "increase": cost_usd_increase,
            "max_increase": max_cost_usd_increase,
        },
    }
    status = "keep" if all(item["passed"] for item in guardrails.values()) else "discard"
    return {
        "status": status,
        "reason": "all guardrails passed" if status == "keep" else "guardrail regression",
        "guardrails": guardrails,
        "deltas": {
            "ndcg@10": ndcg_gain,
            "recall@100": current_recall - previous_recall,
            "latency_p95_ms": current_latency - previous_latency,
            "peak_memory_mb": current_peak_memory - previous_peak_memory,
            "cost_usd": current_cost - previous_cost,
        },
    }


def result_row_from_payload(
    payload: Mapping[str, object],
    *,
    record_type: str = "summary",
    status: str | None = None,
    reason: str | None = None,
    metrics: Mapping[str, object] | None = None,
    description: str | None = None,
    run_id: str | None = None,
) -> Dict[str, object]:
    metrics_payload = payload["metrics"] if metrics is None else metrics
    decision = payload["decision"]
    return {
        "timestamp": payload["timestamp"],
        "record_type": record_type,
        "run_id": run_id or payload["run_id"],
        "parent_commit": payload.get("parent_commit") or "",
        "commit": payload.get("commit") or "",
        "benchmark_name": payload.get("benchmark_name") or "",
        "benchmark_manifest_hash": payload.get("benchmark_manifest_hash") or "",
        "candidate_generation_version": payload.get("candidate_generation_version") or "",
        "harness_version": payload.get("harness_version") or CURRENT_HARNESS_VERSION,
        "split": payload.get("split") or "",
        "ndcg@10": f"{float(metrics_payload.get('ndcg@10', 0.0)):.6f}",
        "recall@100": f"{float(metrics_payload.get('recall@100', 0.0)):.6f}",
        "latency_p95_ms": f"{float(metrics_payload.get('latency_p95_ms', 0.0)):.3f}",
        "mrr@10": f"{float(metrics_payload.get('mrr@10', 0.0)):.6f}",
        "peak_memory_mb": f"{float(metrics_payload.get('peak_memory_mb', 0.0)):.3f}",
        "cost_usd": f"{float(metrics_payload.get('cost_usd', 0.0)):.6f}",
        "wall_clock_s": f"{float(metrics_payload.get('total_seconds', metrics_payload.get('wall_clock_s', 0.0))):.3f}",
        "status": status or str(decision["status"]),
        "decision_reason": reason or str(decision["reason"]),
        "description": description or str(payload.get("label") or ""),
    }


def print_summary(payload: Mapping[str, object]) -> None:
    metrics = payload["metrics"]
    decision = payload["decision"]
    print("---")
    print(f"ndcg@10: {float(metrics.get('ndcg@10', 0.0)):.6f}")
    print(f"recall@100: {float(metrics.get('recall@100', 0.0)):.6f}")
    print(f"latency_p95_ms: {float(metrics.get('latency_p95_ms', 0.0)):.3f}")
    print(f"mrr@10: {float(metrics.get('mrr@10', 0.0)):.6f}")
    print(f"peak_memory_mb: {float(metrics.get('peak_memory_mb', 0.0)):.3f}")
    print(f"cost_usd: {float(metrics.get('cost_usd', 0.0)):.6f}")
    print(f"status: {decision['status']}")
    print(f"reason: {decision['reason']}")
    print(f"run_dir: {payload['run_dir']}")


def execute_run(
    *,
    artifact_dir: Path,
    results_path: Path,
    runs_dir: Path,
    run_dir: Path | None,
    label: str,
    run_id: str,
    split: str,
    budget_seconds: float,
    min_ndcg_gain: float,
    max_recall_rel_drop: float,
    max_latency_rel_increase: float,
    max_peak_memory_rel_increase: float,
    max_cost_usd_increase: float,
    parent_commit: str | None,
    commit: str | None,
    append_results_row: bool,
    reference_keep: Mapping[str, str] | None = None,
) -> Dict[str, object]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    resolved_run_dir = run_dir if run_dir is not None else runs_dir / run_id
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "run.log"
    archived_log_path = resolved_run_dir / "train.log"
    metrics_path = resolved_run_dir / "metrics.json"
    decision_path = resolved_run_dir / "decision.json"
    identity = benchmark_identity(artifact_dir=artifact_dir, split=split)
    resolved_commit = commit or current_commit()

    if reference_keep is None:
        reference_keep = last_keep_result(
            results_path,
            benchmark_name=identity["benchmark_name"],
            benchmark_manifest_hash=identity["benchmark_manifest_hash"],
            harness_version=identity["harness_version"],
            split=split,
        )

    metrics_path.unlink(missing_ok=True)

    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--artifact-dir",
                    str(artifact_dir),
                    "--split",
                    split,
                    "--time-budget-seconds",
                    str(budget_seconds),
                    "--metrics-json",
                    str(metrics_path),
                ],
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=max(budget_seconds * 2.0, 10.0),
                check=False,
            )

        archived_log_path.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        metrics = load_metrics(metrics_path=metrics_path, log_path=log_path)
        metrics.update(identity)
        if completed.returncode != 0 or "ndcg@10" not in metrics:
            raise RuntimeError("train.py failed or did not emit the expected summary metrics")

        decision = build_decision(
            metrics=metrics,
            previous_keep=reference_keep,
            min_ndcg_gain=min_ndcg_gain,
            max_recall_rel_drop=max_recall_rel_drop,
            max_latency_rel_increase=max_latency_rel_increase,
            max_peak_memory_rel_increase=max_peak_memory_rel_increase,
            max_cost_usd_increase=max_cost_usd_increase,
        )
    except Exception as exc:
        if log_path.exists():
            archived_log_path.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        metrics = {
            **identity,
            "ndcg@10": 0.0,
            "recall@100": 0.0,
            "latency_p95_ms": 0.0,
            "mrr@10": 0.0,
            "peak_memory_mb": 0.0,
            "total_seconds": 0.0,
        }
        decision = {
            "status": "crash",
            "reason": str(exc),
            "guardrails": {},
            "deltas": {},
        }

    payload = {
        "timestamp": timestamp,
        "run_id": run_id,
        "parent_commit": parent_commit,
        "commit": resolved_commit,
        "label": label,
        "split": split,
        "run_dir": str(resolved_run_dir),
        "artifact_dir": str(artifact_dir),
        "reference_keep": reference_keep,
        "metrics": metrics,
        "decision": decision,
        **identity,
    }
    decision_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if append_results_row:
        append_result(results_path, result_row_from_payload(payload))

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one reranking experiment.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--min-ndcg-gain", type=float, default=0.002)
    parser.add_argument("--max-recall-rel-drop", type=float, default=0.01)
    parser.add_argument("--max-latency-rel-increase", type=float, default=0.20)
    parser.add_argument("--max-peak-memory-rel-increase", type=float, default=0.20)
    parser.add_argument("--max-cost-usd-increase", type=float, default=0.0)
    args = parser.parse_args()

    if not args.artifact_dir.exists():
        print(f"Missing artifact dir: {args.artifact_dir}. Run `python3 prepare.py` first.", file=sys.stderr)
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = args.label.replace(" ", "-")
    run_id = args.run_id or f"{timestamp}-{safe_label}"
    run_dir = args.run_dir if args.run_dir is not None else args.runs_dir / run_id
    payload = execute_run(
        artifact_dir=args.artifact_dir,
        results_path=args.results,
        runs_dir=args.runs_dir,
        run_dir=run_dir,
        label=args.label,
        run_id=run_id,
        split=args.split,
        budget_seconds=args.budget_seconds,
        min_ndcg_gain=args.min_ndcg_gain,
        max_recall_rel_drop=args.max_recall_rel_drop,
        max_latency_rel_increase=args.max_latency_rel_increase,
        max_peak_memory_rel_increase=args.max_peak_memory_rel_increase,
        max_cost_usd_increase=args.max_cost_usd_increase,
        parent_commit=None,
        commit=current_commit(),
        append_results_row=True,
    )
    print_summary(payload)
    return 0 if payload["decision"]["status"] != "crash" else 1


if __name__ == "__main__":
    raise SystemExit(main())
