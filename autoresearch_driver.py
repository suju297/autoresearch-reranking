from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import pprint
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping

from history_report import refresh_history_exports
from local_brain import (
    DEFAULT_BRAIN_BACKEND,
    DEFAULT_BRAIN_KEEP_LOADED,
    DEFAULT_BRAIN_MAX_TOKENS,
    DEFAULT_BRAIN_MODEL,
    DEFAULT_BRAIN_REASONING_MODE,
    DEFAULT_BRAIN_TEMPERATURE,
    DEFAULT_BRAIN_TOP_P,
    DEFAULT_LLAMA_N_CTX,
    DEFAULT_LLAMA_N_GPU_LAYERS,
    DEFAULT_LLAMA_TYPE_K,
    DEFAULT_LLAMA_TYPE_V,
    DEFAULT_REVIEW_REASONING_MODE,
    DEFAULT_BRAIN_WARM_START,
    ProposalFormatError,
    build_brain,
    parse_proposal,
    strip_reasoning,
)
from prepare import TIME_BUDGET, load_manifest, manifest_hash
from proposal_validator import (
    FAMILY_ORDER,
    FAMILY_SPECS,
    ProposalIdea,
    diversity_metrics,
    family_catalog_markdown,
    family_cooldowns,
    idea_from_mapping,
    infer_family,
    loop_result_changed_keys,
    loop_result_family,
    novelty_rejections,
    normalize_family,
    preferred_families,
    recent_rejected_labels,
    sanitize_slug,
    summarize_family_feedback,
)
from run import (
    CURRENT_HARNESS_VERSION,
    DEFAULT_ARTIFACT_DIR,
    RESULTS_HEADER,
    append_result,
    build_decision,
    current_commit,
    execute_run,
    last_keep_result,
    read_results,
    result_row_from_payload,
    row_metric,
)


ROOT = Path(__file__).resolve().parent
PROGRAM_PATH = ROOT / "program.md"
PLAYBOOK_PATH = ROOT / "docs" / "reranking_playbook.md"
STRATEGY_PATH = ROOT / "rerank_strategy.py"
RESULTS_PATH = ROOT / "results.tsv"
RUNS_DIR = ROOT / "runs"
HISTORY_DIR = ROOT / "history"
FAST_ARTIFACT_DIR = ROOT / "artifacts" / "fast"
FAST_GROUP_NAME = "fast-fiqa-dev"
FAST_BENCHMARK_SPECS = (
    {
        "slug": "fiqa-dev",
        "dataset_id": "fast-fiqa-dev",
        "artifact_dir": FAST_ARTIFACT_DIR / "fiqa-dev",
    },
)
DEFAULT_PROMOTION_ARTIFACT_DIR = ROOT / "artifacts" / "promotion"
DEFAULT_MILESTONE_ARTIFACT_DIR = ROOT / "artifacts" / "milestone"
DEFAULT_REPORT_ARTIFACT_DIR = ROOT / "artifacts" / "report"
DEFAULT_PROMOTION_BUDGET_SECONDS = 1500.0
RESULTS_SUMMARY_LIMIT = 8
ALLOWED_MUTABLE_PATHS = {"rerank_strategy.py"}
IGNORED_STATUS_PREFIXES = ("artifacts/", "runs/")
IGNORED_STATUS_PATHS = {"results.tsv", "run.log"}
DEFAULT_AUTO_ITERATIONS = 3
DEFAULT_MAX_DRAFT_ATTEMPTS = 3
DEFAULT_LABEL_PREFIX = "auto-loop"
DEFAULT_REVIEW_FIRST_INTERVAL = 5
DEFAULT_REVIEW_REPEAT_INTERVAL = 10
DEFAULT_REVIEW_RESULT_LIMIT = 8
DEFAULT_REVIEW_MAX_TOKENS = 320
DEFAULT_EXPLORE_TRIALS_PER_FAMILY = 1
DEFAULT_EXPLOIT_TOP_FAMILIES = 2
MAX_REVIEW_ATTEMPTS = 2
REVIEW_REQUIRED_SECTIONS = (
    "Direction",
    "Failure Pattern",
    "One-Line Guidance",
    "Next Search Region",
    "Next 3 Trial Ideas",
)
ALLOWED_THIRD_PARTY_IMPORTS = {"torch", "sentence_transformers"}
ALLOWED_CANDIDATE_FIELDS = ("doc_id", "text", "retrieval_score", "source_dataset")
DEFAULT_IDEA_CANDIDATES = 4
PROGRESS_PHASE_FRACTIONS = {
    "starting": 0.05,
    "draft": 0.18,
    "reviewer": 0.38,
    "proposal": 0.55,
    "evaluation": 0.80,
    "strategic-review": 0.92,
    "done": 1.00,
}
PROMPT_GENERAL_RULES_CHARS = 1800
PROMPT_FAMILY_CARDS_CHARS = 1100
SUPPORTED_SCORE_NORMALIZATION = {"none", "minmax", "softsign"}
SUPPORTED_TRUNCATION_MODES = {"head", "head_tail"}
SUPPORTED_DEDUP_MODES = {"doc_id", "doc_id_text"}
ALLOWED_STRATEGY_KEYS = {
    "family",
    "candidate_k",
    "score_normalization",
    "fusion_weight",
    "metadata_boost",
    "dedup_filter",
    "truncation_policy",
    "pre_rerank_filter",
    "query_type_heuristic",
}
PROMPT_STRATEGY_LINES = 40
PROMPT_RESULTS_LIMIT = 4
PROMPT_VALIDATION_ERROR_LIMIT = 3
PROMPT_VALIDATION_ERROR_CHARS = 120
AUTONOMOUS_DISABLED_FAMILIES = {"dedup_filter", "pre_rerank_filter"}
AUTONOMOUS_FAMILY_ORDER = tuple(family for family in FAMILY_ORDER if family not in AUTONOMOUS_DISABLED_FAMILIES)
AUTONOMOUS_PRIORITY_FAMILY_ORDER = tuple(
    family
    for family in (
        "candidate_k",
        "truncation_policy",
        "query_type_heuristic",
        "metadata_boost",
        "score_normalization",
        "fusion_weight",
    )
    if family in AUTONOMOUS_FAMILY_ORDER
)


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git(args: List[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def has_git() -> bool:
    try:
        git(["rev-parse", "--is-inside-work-tree"])
        return True
    except Exception:
        return False


def current_branch() -> str:
    if not has_git():
        return "nogit"
    return git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def branch_exists(branch_name: str) -> bool:
    if not has_git():
        return False
    result = git(["show-ref", "--verify", f"refs/heads/{branch_name}"], check=False)
    return result.returncode == 0


def load_manifest_if_present(artifact_dir: Path) -> Dict[str, object]:
    if not artifact_dir.exists():
        return {}
    return load_manifest(artifact_dir=artifact_dir)


def fast_benchmark_specs(artifact_root: Path = FAST_ARTIFACT_DIR) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for spec in FAST_BENCHMARK_SPECS:
        artifact_dir = artifact_root / spec["slug"]
        specs.append({**spec, "artifact_dir": artifact_dir})
    return specs


def configured_fast_benchmark_specs(artifact_root: Path = FAST_ARTIFACT_DIR) -> List[Dict[str, Any]]:
    configured: List[Dict[str, Any]] = []
    for spec in fast_benchmark_specs(artifact_root):
        manifest = load_manifest_if_present(spec["artifact_dir"])
        if manifest:
            configured.append({**spec, "manifest": manifest})
    return configured


def fast_group_manifest(artifact_root: Path = FAST_ARTIFACT_DIR) -> Dict[str, Any]:
    specs = configured_fast_benchmark_specs(artifact_root)
    if not specs:
        return {}
    source_manifests = []
    candidate_versions = set()
    total_queries = 0
    total_dev_queries = 0
    total_test_queries = 0
    total_docs = 0
    candidate_top_ks = set()
    for spec in specs:
        manifest = spec["manifest"]
        source_manifests.append(
            {
                "slug": spec["slug"],
                "artifact_dir": str(spec["artifact_dir"]),
                "dataset": manifest.get("dataset"),
                "query_count": manifest.get("query_count"),
                "dev_query_count": manifest.get("dev_query_count"),
                "test_query_count": manifest.get("test_query_count"),
                "doc_count": manifest.get("doc_count"),
                "manifest_hash": manifest_hash(spec["artifact_dir"]),
            }
        )
        total_queries += int(manifest.get("query_count", 0))
        total_dev_queries += int(manifest.get("dev_query_count", 0))
        total_test_queries += int(manifest.get("test_query_count", 0))
        total_docs += int(manifest.get("doc_count", 0))
        candidate_top_ks.add(int(manifest.get("candidate_top_k", 0)))
        candidate_versions.add(str(manifest.get("candidate_generation_version", "")))
    return {
        "dataset": FAST_GROUP_NAME,
        "source_datasets": source_manifests,
        "query_count": total_queries,
        "dev_query_count": total_dev_queries,
        "test_query_count": total_test_queries,
        "doc_count": total_docs,
        "candidate_top_k": candidate_top_ks.pop() if len(candidate_top_ks) == 1 else sorted(candidate_top_ks),
        "candidate_generation_version": sorted(candidate_versions),
    }


def fast_group_manifest_hash(artifact_root: Path = FAST_ARTIFACT_DIR) -> str:
    specs = configured_fast_benchmark_specs(artifact_root)
    if not specs:
        return "missing"
    payload = [
        {
            "slug": spec["slug"],
            "hash": manifest_hash(spec["artifact_dir"]),
        }
        for spec in specs
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def fast_group_identity(artifact_root: Path = FAST_ARTIFACT_DIR, split: str = "dev") -> Dict[str, str]:
    manifest = fast_group_manifest(artifact_root)
    if not manifest:
        return {
            "benchmark_name": FAST_GROUP_NAME,
            "benchmark_manifest_hash": "missing",
            "candidate_generation_version": "missing",
            "harness_version": CURRENT_HARNESS_VERSION,
            "split": split,
        }
    candidate_generation = manifest.get("candidate_generation_version", [])
    if isinstance(candidate_generation, list):
        candidate_generation_text = ",".join(str(item) for item in candidate_generation)
    else:
        candidate_generation_text = str(candidate_generation)
    return {
        "benchmark_name": FAST_GROUP_NAME,
        "benchmark_manifest_hash": fast_group_manifest_hash(artifact_root),
        "candidate_generation_version": candidate_generation_text,
        "harness_version": CURRENT_HARNESS_VERSION,
        "split": split,
    }


def fast_group_keep_reference(artifact_root: Path = FAST_ARTIFACT_DIR, split: str = "dev") -> Dict[str, str] | None:
    identity = fast_group_identity(artifact_root=artifact_root, split=split)
    return last_keep_result(
        RESULTS_PATH,
        benchmark_name=identity["benchmark_name"],
        benchmark_manifest_hash=identity["benchmark_manifest_hash"],
        harness_version=identity["harness_version"],
        split=split,
    )


def ensure_results_header() -> None:
    if RESULTS_PATH.exists():
        return
    RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")


def parse_git_status() -> Dict[str, List[str]]:
    if not has_git():
        return {"tracked": [], "untracked": []}
    result = git(["status", "--porcelain"], check=True).stdout.splitlines()
    tracked: List[str] = []
    untracked: List[str] = []
    for line in result:
        if not line:
            continue
        status = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if status == "??":
            untracked.append(path)
        else:
            tracked.append(path)
    return {"tracked": tracked, "untracked": untracked}


def classify_dirty_paths() -> Dict[str, List[str]]:
    status = parse_git_status()
    tracked_relevant = []
    tracked_blocked = []
    for path in status["tracked"]:
        if path in ALLOWED_MUTABLE_PATHS:
            tracked_relevant.append(path)
        elif path.startswith(IGNORED_STATUS_PREFIXES) or path in IGNORED_STATUS_PATHS:
            continue
        else:
            tracked_blocked.append(path)

    untracked_relevant = []
    untracked_blocked = []
    for path in status["untracked"]:
        if path in ALLOWED_MUTABLE_PATHS:
            untracked_relevant.append(path)
        elif path.startswith(IGNORED_STATUS_PREFIXES) or path in IGNORED_STATUS_PATHS:
            continue
        else:
            untracked_blocked.append(path)

    return {
        "tracked_relevant": tracked_relevant,
        "tracked_blocked": tracked_blocked,
        "untracked_relevant": untracked_relevant,
        "untracked_blocked": untracked_blocked,
    }


def require_safe_worktree() -> None:
    dirty = classify_dirty_paths()
    blocked = dirty["tracked_blocked"] + dirty["untracked_blocked"]
    if blocked:
        raise RuntimeError(
            "Unsafe worktree for git-native autoresearch. Commit or stash these paths first: "
            + ", ".join(blocked)
        )


def recent_results_markdown(limit: int = RESULTS_SUMMARY_LIMIT) -> str:
    rows = [row for row in read_results(RESULTS_PATH) if row.get("record_type", "summary") == "summary"][-limit:]
    if not rows:
        return "_No experiments recorded yet._"
    header = "| timestamp | commit | benchmark | split | ndcg@10 | recall@100 | latency_p95_ms | mrr@10 | peak_memory_mb | cost_usd | status | reason | description |"
    divider = "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|"
    body = [
        (
            f"| {row['timestamp']} | {row['commit']} | {row.get('benchmark_name', '')} | {row.get('split', '')} | "
            f"{row.get('ndcg@10', '')} | {row.get('recall@100', '')} | {row.get('latency_p95_ms', '')} | "
            f"{row.get('mrr@10', '')} | {row.get('peak_memory_mb', '')} | {row.get('cost_usd', '')} | {row.get('status', '')} | {row.get('decision_reason', '')} | "
            f"{row.get('description', '')} |"
        )
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def compact_results_summary(limit: int = 6) -> str:
    rows = [row for row in read_results(RESULTS_PATH) if row.get("record_type", "summary") == "summary"][-limit:]
    if not rows:
        return "_No experiments recorded yet._"
    lines = []
    for row in rows:
        lines.append(
            "- "
            + f"{row.get('benchmark_name', '')}/{row.get('split', '')}: "
            + f"{row.get('description', '')} -> {row.get('status', '')} "
            + f"(ndcg@10 {row.get('ndcg@10', '')}, latency {row.get('latency_p95_ms', '')}, reason: {row.get('decision_reason', '')})"
        )
    return "\n".join(lines)


def compact_manifest_summary(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    if not manifest:
        return {"configured": False}
    return {
        "dataset": manifest.get("dataset"),
        "query_count": manifest.get("query_count"),
        "doc_count": manifest.get("doc_count"),
        "dev_queries": manifest.get("dev_queries", manifest.get("dev_query_count")),
        "test_queries": manifest.get("test_queries", manifest.get("test_query_count")),
        "candidate_top_k": manifest.get("candidate_top_k"),
        "candidate_generation": manifest.get("candidate_generation", manifest.get("candidate_generation_version")),
        "source_datasets": manifest.get("source_datasets"),
    }


def compact_keep_summary(row: Mapping[str, Any] | None) -> Dict[str, Any]:
    if row is None:
        return {"status": "none"}
    return {
        "benchmark_name": row.get("benchmark_name"),
        "split": row.get("split"),
        "ndcg@10": row.get("ndcg@10"),
        "recall@100": row.get("recall@100"),
        "latency_p95_ms": row.get("latency_p95_ms"),
        "peak_memory_mb": row.get("peak_memory_mb"),
        "description": row.get("description"),
    }


def compact_metric_snapshot(metrics: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not metrics:
        return {"status": "none"}
    return {
        "benchmark_name": metrics.get("benchmark_name"),
        "ndcg@10": metrics.get("ndcg@10"),
        "recall@100": metrics.get("recall@100"),
        "latency_p95_ms": metrics.get("latency_p95_ms"),
        "peak_memory_mb": metrics.get("peak_memory_mb"),
        "cost_usd": metrics.get("cost_usd"),
    }


def compact_iteration_summary(result: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if result is None:
        return None
    summary = {
        "iteration": result.get("iteration"),
        "family": result.get("family"),
        "label": result.get("label"),
        "summary": result.get("summary"),
        "status": result.get("overall", {}).get("status"),
        "reason": result.get("overall", {}).get("reason"),
        "fast_metrics": compact_metric_snapshot(result.get("fast_metrics")),
    }
    if result.get("fast_decision"):
        summary["fast_decision"] = result.get("fast_decision")
    if result.get("promotion_metrics"):
        summary["promotion_metrics"] = compact_metric_snapshot(result.get("promotion_metrics"))
    if result.get("promotion_decision"):
        summary["promotion_decision"] = result.get("promotion_decision")
    return summary


def compact_validation_errors(
    validation_errors: List[str],
    *,
    limit: int = PROMPT_VALIDATION_ERROR_LIMIT,
    max_chars: int = PROMPT_VALIDATION_ERROR_CHARS,
) -> List[str]:
    compacted: List[str] = []
    for error in validation_errors:
        normalized = " ".join(str(error).split())
        if len(normalized) > max_chars:
            normalized = normalized[: max_chars - 3].rstrip() + "..."
        if normalized not in compacted:
            compacted.append(normalized)
    return compacted[-limit:]


def recent_loop_results_markdown(results: List[Mapping[str, Any]], limit: int = DEFAULT_REVIEW_RESULT_LIMIT) -> str:
    if not results:
        return "_No loop iterations recorded yet._"
    rows = results[-limit:]
    header = "| iteration | family | label | status | ndcg@10 | latency_p95_ms | peak_memory_mb | reason |"
    divider = "|---:|---|---|---|---:|---:|---:|---|"
    body = []
    for row in rows:
        metrics = row.get("fast_metrics", {})
        overall = row.get("overall", {})
        ndcg = metrics.get("ndcg@10")
        latency = metrics.get("latency_p95_ms")
        peak_memory = metrics.get("peak_memory_mb")
        ndcg_text = f"{float(ndcg):.6f}" if ndcg is not None else "-"
        latency_text = f"{float(latency):.3f}" if latency is not None else "-"
        peak_memory_text = f"{float(peak_memory):.3f}" if peak_memory is not None else "-"
        body.append(
            f"| {row.get('iteration', '')} | {row.get('family', '')} | {row.get('label', '')} | {overall.get('status', row.get('status', ''))} | "
            f"{ndcg_text} | {latency_text} | {peak_memory_text} | {overall.get('reason', row.get('reason', ''))} |"
        )
    return "\n".join([header, divider, *body])


def proposal_family_name(proposal) -> str | None:
    explicit = normalize_family(getattr(proposal, "family", None))
    if explicit is not None:
        return explicit
    return infer_family(getattr(proposal, "label", ""), getattr(proposal, "summary", ""))


def tried_search_families(loop_results: List[Mapping[str, Any]]) -> List[str]:
    families: List[str] = []
    for row in loop_results:
        family = loop_result_family(row)
        if family is not None and family not in families:
            families.append(family)
    return families


def preferred_search_families(loop_results: List[Mapping[str, Any]]) -> List[str]:
    return [family for family in preferred_families(loop_results) if family in AUTONOMOUS_FAMILY_ORDER]


def recent_failed_families(loop_results: List[Mapping[str, Any]]) -> List[str]:
    return list(family_cooldowns(loop_results).keys())


def decision_ndcg_delta(decision: Mapping[str, Any] | None) -> float | None:
    if not isinstance(decision, Mapping):
        return None
    compact_delta = decision.get("ndcg_delta")
    if isinstance(compact_delta, (int, float)):
        return float(compact_delta)
    guardrails = decision.get("guardrails", {})
    if isinstance(guardrails, Mapping):
        ndcg_gain = guardrails.get("ndcg_gain", {})
        if isinstance(ndcg_gain, Mapping):
            delta = ndcg_gain.get("delta")
            if isinstance(delta, (int, float)):
                return float(delta)
    deltas = decision.get("deltas", {})
    if isinstance(deltas, Mapping):
        delta = deltas.get("ndcg@10")
        if isinstance(delta, (int, float)):
            return float(delta)
    return None


def failed_guardrail_names(decision: Mapping[str, Any] | None) -> List[str]:
    if not isinstance(decision, Mapping):
        return []
    compact_failed = decision.get("failed_guardrails")
    if isinstance(compact_failed, list):
        return [str(name) for name in compact_failed]
    failed: List[str] = []
    guardrails = decision.get("guardrails", {})
    if not isinstance(guardrails, Mapping):
        return failed
    for name, payload in guardrails.items():
        if not isinstance(payload, Mapping):
            continue
        if payload.get("passed") is False:
            failed.append(str(name))
    return failed


def compact_decision_summary(decision: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(decision, Mapping):
        return None
    summary = {
        "status": decision.get("status"),
        "reason": decision.get("reason"),
    }
    ndcg_delta = decision_ndcg_delta(decision)
    if ndcg_delta is not None:
        summary["ndcg_delta"] = ndcg_delta
    failed = failed_guardrail_names(decision)
    if failed:
        summary["failed_guardrails"] = failed
    return summary


def family_signal_stats(loop_results: List[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {
        family: {
            "family": family,
            "attempts": 0,
            "keeps": 0,
            "discards": 0,
            "crashes": 0,
            "brain_errors": 0,
            "promotion_attempts": 0,
            "promotion_passes": 0,
            "best_fast_ndcg_delta": None,
            "best_promotion_ndcg_delta": None,
            "last_label": None,
            "last_status": None,
            "failed_guardrails": {},
            "score": 0.0,
        }
        for family in AUTONOMOUS_FAMILY_ORDER
    }
    for row in loop_results:
        family = loop_result_family(row)
        if family not in stats:
            continue
        family_stats = stats[family]
        family_stats["attempts"] += 1
        family_stats["last_label"] = row.get("label")
        overall = row.get("overall", {})
        status = str(overall.get("status", row.get("status", "")))
        family_stats["last_status"] = status
        if status == "keep":
            family_stats["keeps"] += 1
        elif status == "discard":
            family_stats["discards"] += 1
        elif status == "crash":
            family_stats["crashes"] += 1
        elif status == "brain_error":
            family_stats["brain_errors"] += 1

        for name in failed_guardrail_names(row.get("fast_decision")):
            failed = family_stats["failed_guardrails"]
            failed[name] = failed.get(name, 0) + 1
        for name in failed_guardrail_names(row.get("promotion_decision")):
            failed = family_stats["failed_guardrails"]
            failed[name] = failed.get(name, 0) + 1

        fast_delta = decision_ndcg_delta(row.get("fast_decision"))
        if fast_delta is not None:
            best_fast = family_stats["best_fast_ndcg_delta"]
            if best_fast is None or fast_delta > best_fast:
                family_stats["best_fast_ndcg_delta"] = fast_delta

        promotion_decision = row.get("promotion_decision")
        if promotion_decision is not None:
            family_stats["promotion_attempts"] += 1
            if str(row.get("overall", {}).get("status", "")) == "keep":
                family_stats["promotion_passes"] += 1
            promotion_delta = decision_ndcg_delta(promotion_decision)
            if promotion_delta is not None:
                best_promotion = family_stats["best_promotion_ndcg_delta"]
                if best_promotion is None or promotion_delta > best_promotion:
                    family_stats["best_promotion_ndcg_delta"] = promotion_delta

    for family_stats in stats.values():
        best_fast = max(float(family_stats["best_fast_ndcg_delta"] or 0.0), 0.0)
        best_promotion = max(float(family_stats["best_promotion_ndcg_delta"] or 0.0), 0.0)
        family_stats["score"] = (
            100.0 * family_stats["keeps"]
            + 25.0 * family_stats["promotion_passes"]
            + 5000.0 * best_promotion
            + 2500.0 * best_fast
            - 6.0 * family_stats["crashes"]
            - 4.0 * family_stats["brain_errors"]
            - 2.0 * family_stats["discards"]
        )
    return stats


def family_signal_sort_key(family_stats: Mapping[str, Any]) -> tuple[float, float, float, int, int, int, int]:
    return (
        float(family_stats.get("score", 0.0)),
        float(family_stats.get("best_promotion_ndcg_delta") or 0.0),
        float(family_stats.get("best_fast_ndcg_delta") or 0.0),
        int(family_stats.get("keeps", 0)),
        -int(family_stats.get("brain_errors", 0)),
        -int(family_stats.get("crashes", 0)),
        -int(family_stats.get("discards", 0)),
    )


def build_search_plan(
    loop_results: List[Mapping[str, Any]],
    *,
    next_iteration: int,
    total_iterations: int,
    explore_trials_per_family: int = DEFAULT_EXPLORE_TRIALS_PER_FAMILY,
    exploit_top_families: int = DEFAULT_EXPLOIT_TOP_FAMILIES,
) -> Dict[str, Any]:
    preferred = preferred_search_families(loop_results)
    prioritized_preferred = [
        family for family in AUTONOMOUS_PRIORITY_FAMILY_ORDER if family in preferred
    ] + [
        family for family in preferred if family not in AUTONOMOUS_PRIORITY_FAMILY_ORDER
    ]
    cooldowns = family_cooldowns(loop_results)
    stats = family_signal_stats(loop_results)
    ranked = sorted(
        [stats[family] for family in AUTONOMOUS_FAMILY_ORDER if stats[family]["attempts"] > 0],
        key=family_signal_sort_key,
        reverse=True,
    )

    exploration_remaining = [
        family
        for family in prioritized_preferred
        if stats[family]["attempts"] < explore_trials_per_family
    ]
    phase = "explore" if exploration_remaining else "exploit"

    if phase == "explore":
        target_families = list(exploration_remaining)
        backup_families = [family for family in prioritized_preferred if family not in target_families]
        reason = "Underexplored safe families remain, so spend this phase gathering one clean read on each family."
        target_idea_quota = min(3, len(target_families)) if target_families else 1
        ranked_families = [family_stats["family"] for family_stats in ranked[:exploit_top_families]]
    else:
        ranked_families = [
            family_stats["family"]
            for family_stats in ranked
            if family_stats["family"] not in cooldowns
        ] or [family_stats["family"] for family_stats in ranked]
        target_families = ranked_families[:exploit_top_families] or prioritized_preferred[:exploit_top_families] or list(AUTONOMOUS_PRIORITY_FAMILY_ORDER[:exploit_top_families])
        backup_families = [family for family in prioritized_preferred if family not in target_families]
        reason = "Exploration has baseline coverage, so exploit the families with the best measured signal and fewest controller failures."
        target_idea_quota = min(2, len(target_families)) if target_families else 1

    return {
        "phase": phase,
        "next_iteration": next_iteration,
        "total_iterations": total_iterations,
        "explore_trials_per_family": explore_trials_per_family,
        "exploit_top_families": exploit_top_families,
        "target_families": target_families,
        "backup_families": backup_families,
        "exploration_remaining": exploration_remaining,
        "ranked_families": ranked_families,
        "target_idea_quota": target_idea_quota,
        "reason": reason,
        "family_stats": [stats[family] for family in AUTONOMOUS_FAMILY_ORDER],
    }


def search_phase_markdown(search_plan: Mapping[str, Any]) -> str:
    target_families = list(search_plan.get("target_families", []))
    backup_families = list(search_plan.get("backup_families", []))
    exploration_remaining = list(search_plan.get("exploration_remaining", []))
    ranked_families = list(search_plan.get("ranked_families", []))
    lines = [
        f"- Phase: `{search_plan.get('phase', 'explore')}`",
        f"- Reason: {search_plan.get('reason', '')}",
        f"- Target families this iteration: {', '.join(f'`{family}`' for family in target_families) if target_families else '_none_'}",
        f"- Backup families: {', '.join(f'`{family}`' for family in backup_families[:3]) if backup_families else '_none_'}",
        f"- Ideas that should come from target families: `{search_plan.get('target_idea_quota', 1)}`",
    ]
    if exploration_remaining:
        lines.append(
            f"- Families still needing first-pass coverage: {', '.join(f'`{family}`' for family in exploration_remaining)}"
        )
    if ranked_families:
        lines.append(
            f"- Current family ranking: {', '.join(f'`{family}`' for family in ranked_families[:4])}"
        )
    return "\n".join(lines)


def strategic_review_due(iteration: int, *, first_interval: int, repeat_interval: int) -> bool:
    if first_interval <= 0:
        return False
    if iteration == first_interval:
        return True
    if repeat_interval <= 0 or iteration < first_interval:
        return False
    return iteration > first_interval and (iteration - first_interval) % repeat_interval == 0


def strategy_excerpt(max_lines: int = 160) -> str:
    lines = STRATEGY_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[:max_lines])


def strategy_prompt_excerpt(*, max_lines: int = PROMPT_STRATEGY_LINES) -> str:
    try:
        strategy = load_strategy_mapping(STRATEGY_PATH.read_text(encoding="utf-8"))
        visible_strategy = {
            key: value for key, value in strategy.items() if key not in AUTONOMOUS_DISABLED_FAMILIES
        }
        rendered = pprint.pformat(visible_strategy, sort_dicts=False, width=88)
        return "\n".join(rendered.splitlines()[:max_lines])
    except Exception:
        return strategy_excerpt(max_lines=max_lines)


def validate_strategy_code(code: str) -> None:
    compile(code, str(STRATEGY_PATH), "exec")
    tree = ast.parse(code, filename=str(STRATEGY_PATH))
    required_markers = ("def strategy_config(", "def rerank(")
    missing = [marker for marker in required_markers if marker not in code]
    if missing:
        raise ValueError(f"strategy file is missing required definitions: {', '.join(missing)}")
    stdlib_modules = getattr(sys, "stdlib_module_names", set())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module_name = (node.module or "").split(".", 1)[0]
            module_names = [module_name] if module_name else []
        else:
            continue
        for module_name in module_names:
            if module_name in {"__future__", ""}:
                continue
            if module_name in stdlib_modules:
                continue
            if module_name in ALLOWED_THIRD_PARTY_IMPORTS:
                continue
            raise ValueError(f"proposal introduced unsupported dependency import: {module_name}")


def strategy_assignment_span(code: str) -> tuple[int, int]:
    tree = ast.parse(code, filename=str(STRATEGY_PATH))
    lines = code.splitlines(keepends=True)
    for node in tree.body:
        target_name = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
        if target_name != "STRATEGY":
            continue
        start = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
        end = sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
        return start, end
    raise ValueError("could not locate STRATEGY assignment")


def load_strategy_mapping(code: str) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {}
    exec(compile(code, str(STRATEGY_PATH), "exec"), namespace)
    strategy = namespace.get("STRATEGY")
    if not isinstance(strategy, dict):
        raise ValueError("STRATEGY must evaluate to a dict")
    return copy.deepcopy(strategy)


def replace_strategy_mapping(code: str, strategy: Mapping[str, Any]) -> str:
    start, end = strategy_assignment_span(code)
    replacement = "STRATEGY: Dict[str, object] = " + pprint.pformat(dict(strategy), sort_dicts=False, width=88)
    return code[:start] + replacement + code[end:]


def strategy_mapping_literal(code: str) -> Dict[str, Any] | None:
    stripped = code.strip()
    try:
        parsed = ast.literal_eval(stripped)
    except Exception:
        try:
            parsed = json.loads(stripped)
        except Exception:
            return None
    if not isinstance(parsed, Mapping):
        return None
    return copy.deepcopy(dict(parsed))


def extract_braced_mapping(text: str, *, start_index: int) -> Dict[str, Any] | None:
    # Proposal responses can truncate after emitting a complete STRATEGY literal.
    if start_index < 0 or start_index >= len(text) or text[start_index] != "{":
        return None
    depth = 0
    string_delim: str | None = None
    escape = False
    index = start_index
    while index < len(text):
        char = text[index]
        if string_delim is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == string_delim:
                string_delim = None
            index += 1
            continue
        if char in {"'", '"'}:
            string_delim = char
            index += 1
            continue
        if char == "#":
            newline = text.find("\n", index)
            if newline == -1:
                break
            index = newline + 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = ast.literal_eval(text[start_index : index + 1])
                except Exception:
                    return None
                if not isinstance(parsed, Mapping):
                    return None
                return copy.deepcopy(dict(parsed))
        index += 1
    return None


def strategy_mapping_assignment(code: str) -> Dict[str, Any] | None:
    match = re.search(r"\bSTRATEGY\b\s*(?::[^\n=]+)?=\s*", code)
    if match is None:
        return None
    brace_index = code.find("{", match.end())
    if brace_index == -1:
        return None
    return extract_braced_mapping(code, start_index=brace_index)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"`{field_name}` must be a boolean")


def _coerce_int(value: Any, *, field_name: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"`{field_name}` must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        parsed = int(value.strip())
    else:
        raise ValueError(f"`{field_name}` must be an integer")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"`{field_name}` must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"`{field_name}` must be <= {maximum}")
    return parsed


def _coerce_float(value: Any, *, field_name: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"`{field_name}` must be a number")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"`{field_name}` must be a number") from exc
    else:
        raise ValueError(f"`{field_name}` must be a number")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"`{field_name}` must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"`{field_name}` must be <= {maximum}")
    return parsed


def _coerce_enum(value: Any, *, field_name: str, allowed: set[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"`{field_name}` must be one of: {', '.join(sorted(allowed))}")
    parsed = value.strip()
    if parsed not in allowed:
        raise ValueError(f"`{field_name}` must be one of: {', '.join(sorted(allowed))}")
    return parsed


def _require_mapping(value: Any, *, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"`{field_name}` must be an object")
    return dict(value)


def normalize_strategy_mapping(strategy: Mapping[str, Any]) -> Dict[str, Any]:
    unknown_keys = sorted(set(strategy) - ALLOWED_STRATEGY_KEYS)
    if unknown_keys:
        raise ValueError("strategy contains unsupported top-level keys: " + ", ".join(unknown_keys))

    normalized = copy.deepcopy(dict(strategy))

    raw_family = str(normalized.get("family", "baseline")).strip()
    if raw_family == "baseline":
        normalized["family"] = "baseline"
    else:
        family = normalize_family(raw_family)
        if family is None:
            raise ValueError("`family` must be `baseline` or one of: " + ", ".join(FAMILY_ORDER))
        normalized["family"] = family

    normalized["candidate_k"] = _coerce_int(
        normalized.get("candidate_k", 10),
        field_name="candidate_k",
        minimum=1,
        maximum=100,
    )
    normalized["score_normalization"] = _coerce_enum(
        normalized.get("score_normalization", "none"),
        field_name="score_normalization",
        allowed=SUPPORTED_SCORE_NORMALIZATION,
    )
    normalized["fusion_weight"] = _coerce_float(
        normalized.get("fusion_weight", 1.0),
        field_name="fusion_weight",
        minimum=0.0,
        maximum=1.0,
    )

    metadata_cfg = _require_mapping(normalized.get("metadata_boost", {}), field_name="metadata_boost")
    unknown_metadata = sorted(set(metadata_cfg) - {"enabled", "title_match_weight", "source_match_weight"})
    if unknown_metadata:
        raise ValueError("`metadata_boost` contains unsupported keys: " + ", ".join(unknown_metadata))
    metadata_cfg["enabled"] = _coerce_bool(metadata_cfg.get("enabled", False), field_name="metadata_boost.enabled")
    metadata_cfg["title_match_weight"] = _coerce_float(
        metadata_cfg.get("title_match_weight", 0.05),
        field_name="metadata_boost.title_match_weight",
        minimum=0.0,
        maximum=1.0,
    )
    metadata_cfg["source_match_weight"] = _coerce_float(
        metadata_cfg.get("source_match_weight", 0.02),
        field_name="metadata_boost.source_match_weight",
        minimum=0.0,
        maximum=1.0,
    )
    normalized["metadata_boost"] = metadata_cfg

    dedup_cfg = _require_mapping(normalized.get("dedup_filter", {}), field_name="dedup_filter")
    unknown_dedup = sorted(set(dedup_cfg) - {"enabled", "mode"})
    if unknown_dedup:
        raise ValueError("`dedup_filter` contains unsupported keys: " + ", ".join(unknown_dedup))
    dedup_cfg["enabled"] = _coerce_bool(dedup_cfg.get("enabled", False), field_name="dedup_filter.enabled")
    dedup_cfg["mode"] = _coerce_enum(
        dedup_cfg.get("mode", "doc_id"),
        field_name="dedup_filter.mode",
        allowed=SUPPORTED_DEDUP_MODES,
    )
    normalized["dedup_filter"] = dedup_cfg

    truncation_cfg = _require_mapping(normalized.get("truncation_policy", {}), field_name="truncation_policy")
    unknown_truncation = sorted(set(truncation_cfg) - {"mode", "max_chars"})
    if unknown_truncation:
        raise ValueError("`truncation_policy` contains unsupported keys: " + ", ".join(unknown_truncation))
    truncation_cfg["mode"] = _coerce_enum(
        truncation_cfg.get("mode", "head"),
        field_name="truncation_policy.mode",
        allowed=SUPPORTED_TRUNCATION_MODES,
    )
    truncation_cfg["max_chars"] = _coerce_int(
        truncation_cfg.get("max_chars", 2000),
        field_name="truncation_policy.max_chars",
        minimum=64,
        maximum=20000,
    )
    normalized["truncation_policy"] = truncation_cfg

    prefilter_cfg = _require_mapping(normalized.get("pre_rerank_filter", {}), field_name="pre_rerank_filter")
    unknown_prefilter = sorted(set(prefilter_cfg) - {"enabled", "min_query_term_matches"})
    if unknown_prefilter:
        raise ValueError("`pre_rerank_filter` contains unsupported keys: " + ", ".join(unknown_prefilter))
    prefilter_cfg["enabled"] = _coerce_bool(
        prefilter_cfg.get("enabled", False),
        field_name="pre_rerank_filter.enabled",
    )
    prefilter_cfg["min_query_term_matches"] = _coerce_int(
        prefilter_cfg.get("min_query_term_matches", 1),
        field_name="pre_rerank_filter.min_query_term_matches",
        minimum=1,
        maximum=20,
    )
    normalized["pre_rerank_filter"] = prefilter_cfg

    heuristic_cfg = _require_mapping(normalized.get("query_type_heuristic", {}), field_name="query_type_heuristic")
    unknown_heuristic = sorted(
        set(heuristic_cfg)
        - {
            "enabled",
            "entity_like_max_terms",
            "entity_like_candidate_k",
            "long_query_min_terms",
            "long_query_candidate_k",
        }
    )
    if unknown_heuristic:
        raise ValueError("`query_type_heuristic` contains unsupported keys: " + ", ".join(unknown_heuristic))
    heuristic_cfg["enabled"] = _coerce_bool(
        heuristic_cfg.get("enabled", False),
        field_name="query_type_heuristic.enabled",
    )
    heuristic_cfg["entity_like_max_terms"] = _coerce_int(
        heuristic_cfg.get("entity_like_max_terms", 3),
        field_name="query_type_heuristic.entity_like_max_terms",
        minimum=1,
        maximum=20,
    )
    heuristic_cfg["entity_like_candidate_k"] = _coerce_int(
        heuristic_cfg.get("entity_like_candidate_k", normalized["candidate_k"]),
        field_name="query_type_heuristic.entity_like_candidate_k",
        minimum=1,
        maximum=100,
    )
    heuristic_cfg["long_query_min_terms"] = _coerce_int(
        heuristic_cfg.get("long_query_min_terms", 8),
        field_name="query_type_heuristic.long_query_min_terms",
        minimum=1,
        maximum=50,
    )
    heuristic_cfg["long_query_candidate_k"] = _coerce_int(
        heuristic_cfg.get("long_query_candidate_k", normalized["candidate_k"]),
        field_name="query_type_heuristic.long_query_candidate_k",
        minimum=1,
        maximum=100,
    )
    normalized["query_type_heuristic"] = heuristic_cfg
    return normalized


def normalize_strategy_code(code: str) -> str:
    validate_strategy_code(code)
    strategy = normalize_strategy_mapping(load_strategy_mapping(code))
    return replace_strategy_mapping(code, strategy)


def validate_strategy_change_set(
    *,
    original_code: str,
    updated_code: str,
    declared_changed_keys: List[str],
) -> None:
    original_strategy = normalize_strategy_mapping(load_strategy_mapping(original_code))
    updated_strategy = normalize_strategy_mapping(load_strategy_mapping(updated_code))
    changed_keys = sorted(
        key
        for key in ALLOWED_STRATEGY_KEYS
        if key != "family" and original_strategy.get(key) != updated_strategy.get(key)
    )
    if not changed_keys:
        raise ValueError("proposal did not change any strategy keys")

    declared = {key for key in declared_changed_keys if key in ALLOWED_STRATEGY_KEYS and key != "family"}
    undeclared = [key for key in changed_keys if key not in declared]
    if undeclared:
        raise ValueError("proposal changed undeclared strategy keys: " + ", ".join(undeclared))

    missing = [key for key in declared if key not in changed_keys]
    if missing:
        raise ValueError("proposal declared changed_keys that did not actually change: " + ", ".join(sorted(missing)))


def extract_integer_hint(*parts: object) -> int | None:
    text = " ".join(str(part) for part in parts if part)
    matches = re.findall(r"(?<!\d)(\d{1,3})(?!\d)", text)
    if not matches:
        return None
    return int(matches[-1])


def synthesize_strategy_code_from_idea(code: str, idea: ProposalIdea) -> str:
    strategy = load_strategy_mapping(code)
    family = idea.family
    strategy["family"] = family
    if family == "candidate_k":
        current = int(strategy.get("candidate_k", 10))
        target = extract_integer_hint(idea.label, idea.hypothesis, idea.primary_mechanism)
        if target is None or target == current:
            target = current + 5 if current < 15 else current + 2
        strategy["candidate_k"] = max(1, min(50, target))
    elif family == "score_normalization":
        current = str(strategy.get("score_normalization", "none"))
        strategy["score_normalization"] = "minmax" if current == "none" else "softsign"
    elif family == "fusion_weight":
        current = float(strategy.get("fusion_weight", 1.0))
        strategy["fusion_weight"] = round(0.85 if current >= 0.95 else max(0.5, current - 0.15), 2)
    elif family == "metadata_boost":
        cfg = dict(strategy.get("metadata_boost", {}))
        cfg.update({"enabled": True, "title_match_weight": 0.08, "source_match_weight": 0.03})
        strategy["metadata_boost"] = cfg
    elif family == "dedup_filter":
        cfg = dict(strategy.get("dedup_filter", {}))
        cfg.update({"enabled": True, "mode": "doc_id_text"})
        strategy["dedup_filter"] = cfg
    elif family == "truncation_policy":
        cfg = dict(strategy.get("truncation_policy", {}))
        cfg.update({"mode": "head_tail", "max_chars": 1600})
        strategy["truncation_policy"] = cfg
    elif family == "pre_rerank_filter":
        cfg = dict(strategy.get("pre_rerank_filter", {}))
        cfg.update({"enabled": True, "min_query_term_matches": 2})
        strategy["pre_rerank_filter"] = cfg
    elif family == "query_type_heuristic":
        cfg = dict(strategy.get("query_type_heuristic", {}))
        base_k = int(strategy.get("candidate_k", 10))
        cfg.update(
            {
                "enabled": True,
                "entity_like_max_terms": 3,
                "entity_like_candidate_k": max(base_k + 2, int(cfg.get("entity_like_candidate_k", base_k + 2))),
                "long_query_min_terms": 8,
                "long_query_candidate_k": min(base_k, int(cfg.get("long_query_candidate_k", max(6, base_k - 2)))),
            }
        )
        strategy["query_type_heuristic"] = cfg
    else:
        raise ValueError(f"unsupported synthesized family: {family}")
    updated = replace_strategy_mapping(code, strategy)
    return normalize_strategy_code(updated)


def materialize_strategy_code(
    proposed_code: str,
    *,
    base_code: str,
    selected_idea: ProposalIdea,
) -> tuple[str, bool]:
    mapping = strategy_mapping_literal(proposed_code)
    if mapping is not None:
        strategy = load_strategy_mapping(base_code)
        strategy.update(mapping)
        strategy["family"] = selected_idea.family
        updated = normalize_strategy_code(replace_strategy_mapping(base_code, strategy))
        return updated, False
    try:
        normalized_code = normalize_strategy_code(proposed_code)
    except (SyntaxError, ValueError) as exc:
        mapping = strategy_mapping_assignment(proposed_code)
        if mapping is None:
            raise exc
        strategy = load_strategy_mapping(base_code)
        strategy.update(mapping)
        strategy["family"] = selected_idea.family
        updated = normalize_strategy_code(replace_strategy_mapping(base_code, strategy))
        return updated, True
    return normalized_code, False


def validate_proposal_novelty(proposal, loop_results: List[Mapping[str, Any]]) -> str:
    family = proposal_family_name(proposal)
    if family is None:
        raise ValueError(
            "proposal must include a valid FAMILY line. Allowed families: "
            + ", ".join(FAMILY_ORDER)
        )
    if family not in AUTONOMOUS_FAMILY_ORDER:
        raise ValueError(
            "proposal family is disabled for autonomous search: "
            + family
        )
    if not proposal.changed_keys:
        raise ValueError("proposal must declare changed_keys in the metadata json block")
    if not proposal.why_not_duplicate:
        raise ValueError("proposal must explain why it is not a duplicate")
    idea = ProposalIdea(
        family=family,
        label=proposal.label,
        hypothesis=proposal.hypothesis or proposal.summary,
        changed_keys=proposal.changed_keys,
        why_not_duplicate=proposal.why_not_duplicate,
        expected_ndcg_direction=proposal.expected_ndcg_direction or "up",
        expected_recall_direction=proposal.expected_recall_direction or "flat",
        expected_latency_direction=proposal.expected_latency_direction or "flat",
        promotion_risk=proposal.promotion_risk or "medium",
        primary_mechanism=proposal.primary_mechanism or proposal.summary,
        why_recent_attempts_failed=proposal.why_recent_attempts_failed,
    )
    reasons = novelty_rejections(idea, loop_results)
    if reasons:
        raise ValueError("; ".join(reasons))
    return family


def validate_review_content(content: str) -> None:
    stripped = content.strip()
    if not stripped:
        raise ValueError("review content was empty")
    missing = [section for section in REVIEW_REQUIRED_SECTIONS if f"## {section}" not in stripped]
    if missing:
        raise ValueError("review missing required sections: " + ", ".join(missing))
    ideas_section = stripped.split("## Next 3 Trial Ideas", 1)[1]
    idea_lines = [line for line in ideas_section.splitlines() if line.strip().startswith(("1.", "2.", "3."))]
    if len(idea_lines) < 3:
        raise ValueError("review must include three numbered trial ideas")


def fallback_trial_ideas(loop_results: List[Mapping[str, Any]], *, search_plan: Mapping[str, Any] | None = None) -> List[str]:
    templates = {
        "candidate_k": "Test a nearby rerank depth such as candidate_k = 12 or 13 to keep most of the gain from deeper reranking without repeating the full head-size-15 jump.",
        "score_normalization": "Normalize reranker and retrieval scores before fusion so the blend is less sensitive to scale drift across datasets.",
        "fusion_weight": "Blend retrieval_score into the final ranking only as a tie-break or small residual weight instead of a dominant factor.",
        "metadata_boost": "Use available source or title-like metadata only as a small conditional boost for entity-style queries.",
        "dedup_filter": "Remove obvious duplicate candidates before cross-encoder scoring so the reranker budget is spent on distinct content.",
        "truncation_policy": "Try a shorter focused excerpt or head-tail truncation that preserves the query-bearing span without inflating latency.",
        "pre_rerank_filter": "Add a cheap pre-rerank term-match filter that removes empty or clearly irrelevant candidates before scoring.",
        "query_type_heuristic": "Use a query-conditional rule such as longer candidate depth for short entity-like queries and the baseline otherwise.",
    }
    ideas: List[str] = []
    family_order = list((search_plan or {}).get("target_families", [])) + list((search_plan or {}).get("backup_families", []))
    if not family_order:
        family_order = preferred_search_families(loop_results)
    for family in family_order:
        template = templates.get(family)
        if template is None:
            continue
        ideas.append(family + ": " + template)
        if len(ideas) == 3:
            break
    if len(ideas) < 3:
        for family in AUTONOMOUS_FAMILY_ORDER:
            template = templates.get(family)
            if template is None:
                continue
            candidate = family + ": " + template
            if candidate in ideas:
                continue
            ideas.append(candidate)
            if len(ideas) == 3:
                break
    return ideas


def fallback_review_content(
    *,
    loop_results: List[Mapping[str, Any]],
    artifact_dir: Path,
    promotion_artifact_dir: Path | None,
) -> str:
    recent = loop_results[-DEFAULT_REVIEW_RESULT_LIMIT:]
    status_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    best_fast = None
    best_ndcg = float("-inf")
    for row in recent:
        overall = row.get("overall", {})
        status = str(overall.get("status", row.get("status", "unknown")))
        status_counts[status] = status_counts.get(status, 0) + 1
        family = loop_result_family(row)
        if family is not None:
            family_counts[family] = family_counts.get(family, 0) + 1
        metrics = row.get("fast_metrics", {})
        ndcg = metrics.get("ndcg@10")
        if ndcg is not None and float(ndcg) > best_ndcg:
            best_ndcg = float(ndcg)
            best_fast = row

    dominant_failure = max(status_counts.items(), key=lambda item: item[1])[0] if status_counts else "unknown"
    repeated_family = max(family_counts.items(), key=lambda item: item[1])[0] if family_counts else "none"
    search_plan = build_search_plan(
        loop_results,
        next_iteration=len(loop_results) + 1,
        total_iterations=len(loop_results) + 1,
    )
    next_families = list(search_plan.get("target_families", []))[:3] or preferred_search_families(loop_results)[:3]
    fast_manifest = fast_group_manifest(artifact_dir)
    promotion_manifest = load_manifest_if_present(promotion_artifact_dir) if promotion_artifact_dir is not None else {}

    direction = "The loop is exploring multiple families, but only one fast-loop candidate has shown real ndcg upside and it failed promotion."
    if best_fast is None:
        direction = "The loop is not yet producing a reliable fast-loop winner, so search quality is still the limiting factor."
    failure_pattern = (
        f"Recent results are dominated by `{dominant_failure}` outcomes"
        + (f", with `{repeated_family}` appearing most often." if repeated_family != "none" else ".")
    )
    guidance = "Bias future trials toward changes that can survive the promotion benchmark, not just improve one nano benchmark."
    next_region = ", ".join(f"`{family}`" for family in next_families) if next_families else "`candidate_k`"
    if best_fast is not None and best_fast.get("label") == "rerank-head-size-15":
        next_region = "`candidate_k` near the successful head-size-15 result, then `score_normalization` or `fusion_weight` variants that preserve promotion stability"

    content = [
        "# Strategic Review",
        "",
        "## Direction",
        direction,
        "",
        "## Failure Pattern",
        failure_pattern,
        "",
        "## One-Line Guidance",
        guidance,
        "",
        "## Next Search Region",
        f"Fast benchmark group `{fast_manifest.get('dataset', 'unknown')}` is finding upside, but promotion `{promotion_manifest.get('dataset', 'unconfigured')}` is filtering it out. Explore {next_region}.",
        "",
        "## Next 3 Trial Ideas",
    ]
    for index, idea in enumerate(fallback_trial_ideas(loop_results, search_plan=search_plan), start=1):
        content.append(f"{index}. {idea}")
    return "\n".join(content).strip()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def refresh_history() -> Dict[str, str]:
    return refresh_history_exports(output_dir=HISTORY_DIR, runs_dir=RUNS_DIR, results_path=RESULTS_PATH)


def render_loop_progress(
    *,
    iteration: int,
    total_iterations: int,
    phase: str,
    keeps: int,
    latest_status: str = "",
    width: int = 28,
) -> str:
    if total_iterations <= 0:
        total_iterations = 1
    phase_fraction = PROGRESS_PHASE_FRACTIONS.get(phase, 0.0)
    completed_fraction = min(
        max(((iteration - 1) + phase_fraction) / total_iterations, 0.0),
        1.0,
    )
    filled = min(width, max(0, round(width * completed_fraction)))
    bar = "#" * filled + "-" * (width - filled)
    status_suffix = f" status={latest_status}" if latest_status else ""
    return (
        f"[{bar}] iter {iteration}/{total_iterations} "
        f"phase={phase} keeps={keeps}{status_suffix}"
    )


def emit_loop_progress(
    *,
    iteration: int,
    total_iterations: int,
    phase: str,
    keeps: int,
    latest_status: str = "",
    done: bool = False,
) -> None:
    line = render_loop_progress(
        iteration=iteration,
        total_iterations=total_iterations,
        phase=phase,
        keeps=keeps,
        latest_status=latest_status,
    )
    if sys.stderr.isatty():
        terminator = "\n" if done else "\r"
        sys.stderr.write(line + terminator)
        sys.stderr.flush()
        return
    if done:
        print(line, file=sys.stderr)


def refresh_loop_payload(loop_payload: Dict[str, Any]) -> None:
    results = loop_payload.get("results", [])
    loop_payload["diversity"] = diversity_metrics(results)
    loop_payload["family_feedback"] = summarize_family_feedback(results, limit=9)
    loop_payload["search_policy"] = build_search_plan(
        results,
        next_iteration=len(results) + 1,
        total_iterations=int(loop_payload.get("iterations_requested", 0) or 0),
        explore_trials_per_family=int(
            loop_payload.get("policy", {}).get("explore_trials_per_family", DEFAULT_EXPLORE_TRIALS_PER_FAMILY)
        ),
        exploit_top_families=int(
            loop_payload.get("policy", {}).get("exploit_top_families", DEFAULT_EXPLOIT_TOP_FAMILIES)
        ),
    )


def load_playbook_text() -> str:
    if not PLAYBOOK_PATH.exists():
        return ""
    return PLAYBOOK_PATH.read_text(encoding="utf-8")


def _section_slice(text: str, marker: str, *, next_markers: tuple[str, ...]) -> str:
    start = text.find(marker)
    if start == -1:
        return ""
    end = text.find("\n## ", start + len(marker))
    if end == -1:
        end = len(text)
    for next_marker in next_markers:
        candidate = text.find(next_marker, start + len(marker))
        if candidate != -1:
            end = min(end, candidate)
    return text[start:end].strip()


def playbook_general_excerpt(*, max_chars: int = 4200) -> str:
    text = load_playbook_text()
    if not text:
        return "_No local playbook configured._"
    markers = [
        "## 1. Operating rules",
        "## 2. Fast mental model",
        "## 4. Family selection priorities",
        "## 5. Benchmark awareness cards",
        "## 6. Anti-patterns",
        "## 7. Novelty policy",
        "## 12. Good first experiment menu",
    ]
    chunks = []
    for index, marker in enumerate(markers):
        next_markers = tuple(markers[index + 1:] + ["## 3. Proposal families", "## 13. Short glossary"])
        section = _section_slice(text, marker, next_markers=next_markers)
        if section:
            chunks.append(section)
    filtered_chunks = []
    for chunk in chunks:
        kept_lines = [
            line
            for line in chunk.splitlines()
            if not any(family in line for family in AUTONOMOUS_DISABLED_FAMILIES)
        ]
        filtered_chunks.append("\n".join(kept_lines).strip())
    combined = "\n\n".join(chunk for chunk in filtered_chunks if chunk).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars].rsplit("\n", 1)[0].strip()
    return combined or "_No local playbook configured._"


def playbook_family_cards(families: List[str], *, max_chars: int = 3200) -> str:
    text = load_playbook_text()
    if not text:
        return "_No local playbook configured._"
    chunks = []
    seen = set()
    for family in families:
        normalized = normalize_family(family)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        marker = f"`{normalized}`"
        candidate_start = text.find(marker)
        if candidate_start == -1:
            continue
        heading_start = text.rfind("### ", 0, candidate_start)
        if heading_start == -1:
            continue
        next_heading = text.find("\n### ", candidate_start)
        next_section = text.find("\n## ", candidate_start)
        ends = [value for value in (next_heading, next_section) if value != -1]
        end = min(ends) if ends else len(text)
        chunks.append(text[heading_start:end].strip())
    combined = "\n\n".join(chunks).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars].rsplit("\n", 1)[0].strip()
    return combined or "_No family cards available._"


def runtime_schema_excerpt() -> str:
    lines = [
        "- `candidate_k`: integer from 1 to 100.",
        f"- `score_normalization`: one of {', '.join(f'`{value}`' for value in sorted(SUPPORTED_SCORE_NORMALIZATION))}.",
        "- `fusion_weight`: numeric float from 0.0 to 1.0. Do not use strings such as `rank_only`.",
        f"- `truncation_policy.mode`: one of {', '.join(f'`{value}`' for value in sorted(SUPPORTED_TRUNCATION_MODES))}.",
        "- `truncation_policy.max_chars`: integer, not an object or per-dataset map.",
        "- `metadata_boost` weights must be numeric floats.",
        "- `query_type_heuristic` thresholds and candidate_k values must be integers.",
        "- If the strategy violates this runtime schema, the proposal will be rejected before evaluation.",
    ]
    return "\n".join(lines)


def extract_json_payload(text: str) -> Any:
    cleaned = strip_reasoning(text).strip()
    fenced_blocks = []
    marker = "```json"
    start = 0
    while True:
        block_start = cleaned.find(marker, start)
        if block_start == -1:
            break
        content_start = cleaned.find("\n", block_start)
        if content_start == -1:
            break
        block_end = cleaned.find("```", content_start + 1)
        if block_end == -1:
            break
        fenced_blocks.append(cleaned[content_start + 1:block_end].strip())
        start = block_end + 3
    for block in fenced_blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("response did not contain a valid json payload") from exc


def family_feedback_markdown(loop_results: List[Mapping[str, Any]]) -> str:
    lines = [summarize_family_feedback(loop_results, limit=9)]
    cooldowns = family_cooldowns(loop_results)
    if cooldowns:
        lines.append("")
        lines.append("Cooldowns:")
        for family, remaining in cooldowns.items():
            lines.append(f"- `{family}`: {remaining} run(s) remaining")
    return "\n".join(lines)


def idea_to_mapping(idea: ProposalIdea) -> Dict[str, Any]:
    return {
        "family": idea.family,
        "label": idea.label,
        "hypothesis": idea.hypothesis,
        "changed_keys": idea.changed_keys,
        "why_not_duplicate": idea.why_not_duplicate,
        "expected_ndcg_direction": idea.expected_ndcg_direction,
        "expected_recall_direction": idea.expected_recall_direction,
        "expected_latency_direction": idea.expected_latency_direction,
        "promotion_risk": idea.promotion_risk,
        "primary_mechanism": idea.primary_mechanism,
        "why_recent_attempts_failed": idea.why_recent_attempts_failed,
        "rationale_fingerprint": idea.rationale_fingerprint(),
    }


def fallback_idea_for_family(family: str, loop_results: List[Mapping[str, Any]]) -> ProposalIdea:
    if family not in AUTONOMOUS_FAMILY_ORDER:
        raise ValueError(f"family `{family}` is disabled for autonomous search")
    templates = {
        "candidate_k": {
            "hypothesis": "A bounded change to candidate depth may surface more relevant items to the reranker without rewriting the whole policy.",
            "changed_keys": ["candidate_k"],
            "expected_ndcg_direction": "up",
            "expected_recall_direction": "flat",
            "expected_latency_direction": "slightly_up",
            "promotion_risk": "medium",
            "primary_mechanism": "candidate_k",
        },
        "score_normalization": {
            "hypothesis": "Changing score normalization may make reranker and retrieval scores easier to compare without touching retrieval itself.",
            "changed_keys": ["score_normalization"],
            "expected_ndcg_direction": "up",
            "expected_recall_direction": "flat",
            "expected_latency_direction": "flat",
            "promotion_risk": "medium",
            "primary_mechanism": "score_normalization",
        },
        "fusion_weight": {
            "hypothesis": "A mechanism-level fusion change may preserve useful retrieval priors when reranker scores are noisy.",
            "changed_keys": ["fusion_weight"],
            "expected_ndcg_direction": "up",
            "expected_recall_direction": "flat",
            "expected_latency_direction": "flat",
            "promotion_risk": "high",
            "primary_mechanism": "fusion_weight",
        },
        "metadata_boost": {
            "hypothesis": "A small conditional metadata prior may help entity-like or title-oriented queries without large latency cost.",
            "changed_keys": ["metadata_boost"],
            "expected_ndcg_direction": "up",
            "expected_recall_direction": "flat",
            "expected_latency_direction": "slightly_up",
            "promotion_risk": "medium",
            "primary_mechanism": "metadata_boost",
        },
        "truncation_policy": {
            "hypothesis": "A tighter truncation policy may reduce noise and latency while preserving the most relevant text span.",
            "changed_keys": ["truncation_policy"],
            "expected_ndcg_direction": "up",
            "expected_recall_direction": "flat",
            "expected_latency_direction": "slightly_down",
            "promotion_risk": "medium",
            "primary_mechanism": "truncation_policy",
        },
        "query_type_heuristic": {
            "hypothesis": "A simple query-type heuristic may let the strategy use different safe settings for short entity-like vs descriptive queries.",
            "changed_keys": ["query_type_heuristic"],
            "expected_ndcg_direction": "up",
            "expected_recall_direction": "flat",
            "expected_latency_direction": "slightly_up",
            "promotion_risk": "medium",
            "primary_mechanism": "query_type_heuristic",
        },
    }
    template = templates[family]
    feedback = summarize_family_feedback(loop_results, limit=5)
    return ProposalIdea(
        family=family,
        label=f"fallback-{family.replace('_', '-')}",
        hypothesis=template["hypothesis"],
        changed_keys=list(template["changed_keys"]),
        why_not_duplicate=f"Fallback playbook idea for an underexplored family. Recent family feedback: {feedback}",
        expected_ndcg_direction=template["expected_ndcg_direction"],
        expected_recall_direction=template["expected_recall_direction"],
        expected_latency_direction=template["expected_latency_direction"],
        promotion_risk=template["promotion_risk"],
        primary_mechanism=str(template["primary_mechanism"]),
        why_recent_attempts_failed="Use this only because the proposer underfilled the idea list; keep the mechanism simple and interpretable.",
    )


def supplement_candidate_ideas(
    ideas: List[ProposalIdea],
    *,
    loop_results: List[Mapping[str, Any]],
    prioritized_families: List[str] | None = None,
    target_count: int = DEFAULT_IDEA_CANDIDATES,
) -> List[ProposalIdea]:
    supplemented = list(ideas)
    seen_families = {idea.family for idea in supplemented}
    family_order = prioritized_families or preferred_search_families(loop_results)
    for family in family_order:
        if len(supplemented) >= target_count:
            break
        if family in seen_families:
            continue
        fallback = fallback_idea_for_family(family, loop_results)
        if novelty_rejections(fallback, loop_results):
            continue
        supplemented.append(fallback)
        seen_families.add(family)
    for family in preferred_search_families(loop_results):
        if len(supplemented) >= target_count:
            break
        if family in seen_families:
            continue
        fallback = fallback_idea_for_family(family, loop_results)
        if novelty_rejections(fallback, loop_results):
            continue
        supplemented.append(fallback)
        seen_families.add(family)
    for family in AUTONOMOUS_FAMILY_ORDER:
        if len(supplemented) >= target_count:
            break
        if family in seen_families:
            continue
        fallback = fallback_idea_for_family(family, loop_results)
        if novelty_rejections(fallback, loop_results):
            continue
        supplemented.append(fallback)
        seen_families.add(family)
    return supplemented


def _extract_partial_string_field(text: str, field_name: str) -> str:
    cleaned = strip_reasoning(text)
    match = re.search(
        rf'"{re.escape(field_name)}"\s*:\s*"([^"\r\n]*)',
        cleaned,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _extract_partial_string_list(text: str, field_name: str) -> List[str]:
    cleaned = strip_reasoning(text)
    match = re.search(
        rf'"{re.escape(field_name)}"\s*:\s*\[([^\]]*)',
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    return [item.strip() for item in re.findall(r'"([^"]+)"', match.group(1))]


def recover_candidate_ideas(text: str, *, loop_results: List[Mapping[str, Any]]) -> List[ProposalIdea]:
    partial_family = normalize_family(_extract_partial_string_field(text, "family"))
    if partial_family is None:
        partial_family = infer_family(
            _extract_partial_string_field(text, "label"),
            _extract_partial_string_field(text, "hypothesis"),
            strip_reasoning(text)[:400],
        )
    if partial_family is None or partial_family not in AUTONOMOUS_FAMILY_ORDER:
        return []

    fallback = fallback_idea_for_family(partial_family, loop_results)
    payload: Dict[str, Any] = idea_to_mapping(fallback)
    partial_label = _extract_partial_string_field(text, "label")
    partial_hypothesis = _extract_partial_string_field(text, "hypothesis")
    partial_why_not_duplicate = _extract_partial_string_field(text, "why_not_duplicate")
    partial_primary_mechanism = _extract_partial_string_field(text, "primary_mechanism")
    partial_why_recent_attempts_failed = _extract_partial_string_field(text, "why_recent_attempts_failed")
    partial_ndcg = _extract_partial_string_field(text, "expected_ndcg_direction")
    partial_recall = _extract_partial_string_field(text, "expected_recall_direction")
    partial_latency = _extract_partial_string_field(text, "expected_latency_direction")
    partial_promotion_risk = _extract_partial_string_field(text, "promotion_risk")

    if partial_label:
        payload["label"] = sanitize_slug(partial_label, fallback=fallback.label)
    if partial_hypothesis:
        payload["hypothesis"] = partial_hypothesis
    if partial_why_not_duplicate:
        payload["why_not_duplicate"] = partial_why_not_duplicate
    if partial_primary_mechanism:
        payload["primary_mechanism"] = partial_primary_mechanism
    if partial_why_recent_attempts_failed:
        payload["why_recent_attempts_failed"] = partial_why_recent_attempts_failed
    if partial_ndcg:
        payload["expected_ndcg_direction"] = partial_ndcg
    if partial_recall:
        payload["expected_recall_direction"] = partial_recall
    if partial_latency:
        payload["expected_latency_direction"] = partial_latency
    if partial_promotion_risk:
        payload["promotion_risk"] = partial_promotion_risk
    partial_changed_keys = _extract_partial_string_list(text, "changed_keys")
    if partial_changed_keys:
        payload["changed_keys"] = partial_changed_keys

    try:
        return [idea_from_mapping(payload)]
    except ValueError:
        return [fallback]


def parse_candidate_ideas(text: str) -> List[ProposalIdea]:
    payload = extract_json_payload(text)
    if isinstance(payload, Mapping):
        ideas_payload = payload.get("ideas")
        if ideas_payload is None and "family" in payload:
            ideas_payload = [payload]
    else:
        ideas_payload = payload
    if not isinstance(ideas_payload, list):
        raise ValueError("idea proposer must return a json list, a single idea object, or an object with an `ideas` list")
    ideas: List[ProposalIdea] = []
    families_seen = set()
    for item in ideas_payload:
        if not isinstance(item, Mapping):
            raise ValueError("every idea must be a json object")
        idea = idea_from_mapping(item)
        if idea.family in families_seen:
            raise ValueError(f"idea list repeated family `{idea.family}`")
        families_seen.add(idea.family)
        ideas.append(idea)
    if not ideas:
        raise ValueError("idea proposer must return at least 1 candidate idea")
    return ideas


def parse_reviewer_selection(text: str, ideas: List[ProposalIdea]) -> Dict[str, str]:
    payload = extract_json_payload(text)
    if not isinstance(payload, Mapping):
        raise ValueError("reviewer must return a json object")
    family = normalize_family(str(payload.get("family", "")))
    label = str(payload.get("label", "")).strip()
    justification = str(payload.get("justification", "")).strip()
    if not justification:
        synthesized_parts = [
            str(payload.get("hypothesis", "")).strip(),
            str(payload.get("primary_mechanism", "")).strip(),
            str(payload.get("why_not_duplicate", "")).strip(),
        ]
        justification = " ".join(part for part in synthesized_parts if part)
    if family is None or not label:
        raise ValueError("reviewer must return valid `family` and `label` fields")
    if not justification:
        raise ValueError("reviewer must explain why the selected idea is best")
    for idea in ideas:
        if idea.family == family and idea.label == label:
            return {"family": family, "label": label, "justification": justification}
    raise ValueError("reviewer selected an idea that was not in the proposer candidate list")


def fallback_reviewer_selection(
    ideas: List[ProposalIdea],
    *,
    raw_text: str,
    error: str,
) -> Dict[str, str]:
    if not ideas:
        raise ValueError("reviewer fallback requires at least one valid idea")
    payload = None
    try:
        extracted = extract_json_payload(raw_text)
        if isinstance(extracted, Mapping):
            payload = extracted
    except ValueError:
        payload = None

    requested_family = normalize_family(str(payload.get("family", ""))) if payload is not None else None
    requested_label = str(payload.get("label", "")).strip() if payload is not None else ""
    if requested_family is not None:
        family_matches = [idea for idea in ideas if idea.family == requested_family]
        if len(family_matches) == 1:
            selected = family_matches[0]
            justification = (
                f"Reviewer output was invalid ({error}); preserving the requested family "
                f"and selecting the only valid candidate remaining in `{selected.family}`."
            )
            return {
                "family": selected.family,
                "label": selected.label,
                "justification": justification,
                "fallback_reason": error,
            }

    selected = ideas[0]
    requested_summary = ""
    if requested_family is not None or requested_label:
        requested_summary = (
            f" Requested family={requested_family or 'unknown'} label={requested_label or 'unknown'}."
        )
    justification = (
        f"Reviewer output was invalid ({error}); defaulting to the first novelty-safe candidate "
        f"to keep the loop moving.{requested_summary}"
    )
    return {
        "family": selected.family,
        "label": selected.label,
        "justification": justification,
        "fallback_reason": error,
    }


def build_idea_prompt(
    *,
    loop_id: str,
    iteration: int,
    total_iterations: int,
    artifact_dir: Path,
    promotion_artifact_dir: Path | None,
    loop_results: List[Mapping[str, Any]],
    search_plan: Mapping[str, Any],
    last_outcome: Mapping[str, Any] | None,
    latest_review: Mapping[str, Any] | None,
    validation_errors: List[str],
) -> str:
    fast_manifest = fast_group_manifest(artifact_dir)
    promotion_manifest = load_manifest_if_present(promotion_artifact_dir) if promotion_artifact_dir is not None else {}
    last_keep = fast_group_keep_reference(artifact_dir, "dev")
    preferred_family_order = preferred_search_families(loop_results)
    playbook_families = (
        list(search_plan.get("target_families", []))
        + list(search_plan.get("backup_families", []))
    )[:5] or preferred_family_order[:5] or list(AUTONOMOUS_FAMILY_ORDER[:5])
    repeated_labels = recent_rejected_labels(loop_results)
    feedback_sections = [
        "# Autoresearch Idea Proposer",
        f"Loop: `{loop_id}`",
        f"Iteration: `{iteration}` of `{total_iterations}`",
        f"Branch: `{current_branch()}`",
        f"Commit: `{current_commit()}`",
        "",
        "## Rules",
        "- Edit only `rerank_strategy.py`.",
        "- Propose 3 to 4 candidate ideas from different families.",
        f"- Families must come from: {', '.join(f'`{family}`' for family in AUTONOMOUS_FAMILY_ORDER)}.",
        "- Optimize `ndcg@10` on frozen `beir/fiqa/dev`, but do not overfit FiQA-only wins at the expense of promotion `scifact`.",
        "- Prefer mechanism changes over tiny numeric nudges.",
        "- Do not repeat a recent failed family without a materially different mechanism and changed_keys set.",
        f"- Current phase is `{search_plan.get('phase', 'explore')}`. Prioritize target families `{', '.join(search_plan.get('target_families', [])) or 'none'}`.",
        f"- At least {search_plan.get('target_idea_quota', 1)} ideas should come from the target families unless novelty blocks them.",
        f"- Do not add new third-party imports beyond: {', '.join(f'`{name}`' for name in sorted(ALLOWED_THIRD_PARTY_IMPORTS))}.",
        f"- Candidate dictionaries only guarantee these keys: {', '.join(f'`{name}`' for name in ALLOWED_CANDIDATE_FIELDS)}.",
        "",
        "Return exactly one JSON object with an `ideas` list. Each idea must follow this schema:",
        "```json",
        json.dumps(
            {
                "ideas": [
                    {
                        "family": "candidate_k",
                        "label": "short-kebab-case-label",
                        "hypothesis": "why this should improve reranking",
                        "changed_keys": ["candidate_k"],
                        "why_not_duplicate": "why this is materially different",
                        "expected_ndcg_direction": "up",
                        "expected_recall_direction": "flat",
                        "expected_latency_direction": "slightly_up",
                        "promotion_risk": "medium",
                        "primary_mechanism": "the main mechanism being changed",
                        "why_recent_attempts_failed": "why nearby attempts failed",
                    }
                ]
            },
            indent=2,
        ),
        "```",
        "",
        "## Allowed Search Families",
        "\n".join(f"- `{family}`: {FAMILY_SPECS[family]}" for family in AUTONOMOUS_FAMILY_ORDER),
        "",
        "## Local Playbook Rules",
        playbook_general_excerpt(max_chars=PROMPT_GENERAL_RULES_CHARS),
        "",
        "## Local Playbook Family Cards",
        playbook_family_cards(playbook_families, max_chars=PROMPT_FAMILY_CARDS_CHARS),
        "",
        "## Family Feedback",
        family_feedback_markdown(loop_results),
        "",
        "## Search Phase",
        search_phase_markdown(search_plan),
        "",
        "## Current Search State",
        f"- Prefer families in this order: {', '.join(f'`{family}`' for family in preferred_family_order) if preferred_family_order else '_none_'}",
        f"- Recently rejected labels: {', '.join(f'`{label}`' for label in repeated_labels) if repeated_labels else '_none_'}",
        "",
        "## Fast Benchmark",
        "```json",
        json.dumps(compact_manifest_summary(fast_manifest), indent=2),
        "```",
        "",
        "## Promotion Benchmark",
        "```json",
        json.dumps(compact_manifest_summary(promotion_manifest), indent=2),
        "```",
        "",
        "## Last Fast Keep",
        "```json",
        json.dumps(compact_keep_summary(last_keep), indent=2),
        "```",
        "",
        "## Recent Results",
        compact_results_summary(limit=PROMPT_RESULTS_LIMIT),
        "",
        "## Current Strategy",
        "```python",
        strategy_prompt_excerpt(),
        "```",
    ]
    if last_outcome is not None:
        feedback_sections.extend(
            [
                "",
                "## Last Outcome",
                "```json",
                json.dumps(last_outcome, indent=2),
                "```",
            ]
        )
    if latest_review is not None:
        feedback_sections.extend(
            [
                "",
                "## Strategic Review Guidance",
                f"Review trigger: `{latest_review.get('trigger', 'scheduled')}` after iteration `{latest_review.get('iteration', '')}`",
                "```markdown",
                str(latest_review.get("content", "")).strip(),
                "```",
            ]
        )
    if validation_errors:
        feedback_sections.extend(
            [
                "",
                "## Validation Errors To Avoid",
                *[f"- {error}" for error in compact_validation_errors(validation_errors)],
            ]
        )
    return "\n".join(feedback_sections) + "\n"


def build_reviewer_prompt(
    *,
    loop_id: str,
    iteration: int,
    ideas: List[ProposalIdea],
    loop_results: List[Mapping[str, Any]],
    search_plan: Mapping[str, Any],
    latest_review: Mapping[str, Any] | None,
) -> str:
    idea_families = [idea.family for idea in ideas]
    sections = [
        "# Autoresearch Idea Reviewer",
        f"Loop: `{loop_id}`",
        f"Iteration: `{iteration}`",
        "",
        "Review the candidate ideas and pick exactly one.",
        "Pick only from the `Candidate Ideas` list below. Do not invent or reuse a rejected label.",
        "Penalize duplicate mechanisms, vague `fusion_weight` nudges, ideas likely to hurt latency without clear upside, and ideas that only look good on one nano benchmark but are unlikely to survive promotion.",
        f"Prefer target families for this `{search_plan.get('phase', 'explore')}` phase unless an off-target backup idea is clearly more novel and benchmark-aware.",
        "",
        "Return exactly one JSON object:",
        "```json",
        json.dumps(
            {
                "family": "candidate_k",
                "label": "selected-idea-label",
                "justification": "why this idea is the best next experiment",
            },
            indent=2,
        ),
        "```",
        "",
        "## Local Playbook Rules",
        playbook_general_excerpt(max_chars=PROMPT_GENERAL_RULES_CHARS),
        "",
        "## Local Playbook Family Cards",
        playbook_family_cards(idea_families, max_chars=PROMPT_FAMILY_CARDS_CHARS),
        "",
        "## Family Feedback",
        family_feedback_markdown(loop_results),
        "",
        "## Search Phase",
        search_phase_markdown(search_plan),
        "",
        "## Recent Results",
        compact_results_summary(limit=PROMPT_RESULTS_LIMIT),
        "",
        "## Candidate Ideas",
        "```json",
        json.dumps([idea_to_mapping(idea) for idea in ideas], indent=2),
        "```",
    ]
    if latest_review is not None:
        sections.extend(
            [
                "",
                "## Strategic Review Guidance",
                str(latest_review.get("content", "")).strip(),
            ]
        )
    return "\n".join(sections) + "\n"


def build_proposal_prompt(
    *,
    loop_id: str,
    iteration: int,
    total_iterations: int,
    artifact_dir: Path,
    promotion_artifact_dir: Path | None,
    loop_results: List[Mapping[str, Any]],
    search_plan: Mapping[str, Any],
    selected_idea: ProposalIdea,
    reviewer_selection: Mapping[str, str],
    last_outcome: Mapping[str, Any] | None,
    latest_review: Mapping[str, Any] | None,
    validation_errors: List[str],
) -> str:
    fast_manifest = fast_group_manifest(artifact_dir)
    promotion_manifest = load_manifest_if_present(promotion_artifact_dir) if promotion_artifact_dir is not None else {}
    last_keep = fast_group_keep_reference(artifact_dir, "dev")
    feedback_sections = [
        "# Autoresearch Final Proposal",
        f"Loop: `{loop_id}`",
        f"Iteration: `{iteration}` of `{total_iterations}`",
        "",
        "Implement exactly the selected idea below by returning a metadata JSON block and a `python` block containing either a complete `STRATEGY` mapping or the full `rerank_strategy.py` file.",
        "- Return exactly two fenced blocks: one `json` block, then one `python` block.",
        "- Keep the mechanism within the chosen family.",
        "- Do not change unrelated keys.",
        "- Do not return the unchanged file.",
        "- Prefer returning only an updated `STRATEGY` dict in the python block. It must be a complete mapping and the harness will reconstruct the module around it.",
        "- Return the full module only if you must change executable code outside `STRATEGY`.",
        f"- Candidate dictionaries only guarantee these keys: {', '.join(f'`{name}`' for name in ALLOWED_CANDIDATE_FIELDS)}.",
        "",
        "## Selected Idea",
        "```json",
        json.dumps(idea_to_mapping(selected_idea), indent=2),
        "```",
        "",
        "## Local Playbook Rules",
        playbook_general_excerpt(max_chars=PROMPT_GENERAL_RULES_CHARS),
        "",
        "## Local Playbook Family Card",
        playbook_family_cards([selected_idea.family], max_chars=PROMPT_FAMILY_CARDS_CHARS),
        "",
        "## Reviewer Justification",
        "```json",
        json.dumps(dict(reviewer_selection), indent=2),
        "```",
        "",
        "## Runtime Schema Rules",
        runtime_schema_excerpt(),
        "",
        "## Family Feedback",
        family_feedback_markdown(loop_results),
        "",
        "## Search Phase",
        search_phase_markdown(search_plan),
        "",
        "## Fast Benchmark",
        "```json",
        json.dumps(compact_manifest_summary(fast_manifest), indent=2),
        "```",
        "",
        "## Promotion Benchmark",
        "```json",
        json.dumps(compact_manifest_summary(promotion_manifest), indent=2),
        "```",
        "",
        "## Last Fast Keep",
        "```json",
        json.dumps(compact_keep_summary(last_keep), indent=2),
        "```",
        "",
        "## Current Strategy",
        "```python",
        strategy_excerpt(),
        "```",
    ]
    if last_outcome is not None:
        feedback_sections.extend(
            [
                "",
                "## Last Outcome",
                "```json",
                json.dumps(last_outcome, indent=2),
                "```",
            ]
        )
    if latest_review is not None:
        feedback_sections.extend(
            [
                "",
                "## Strategic Review Guidance",
                str(latest_review.get("content", "")).strip(),
            ]
        )
    if validation_errors:
        feedback_sections.extend(
            [
                "",
                "## Validation Errors To Avoid",
                *[f"- {error}" for error in compact_validation_errors(validation_errors)],
            ]
        )
    return "\n".join(feedback_sections) + "\n"


def build_strategy_review_prompt(
    *,
    loop_id: str,
    iteration: int,
    artifact_dir: Path,
    promotion_artifact_dir: Path | None,
    loop_results: List[Mapping[str, Any]],
    search_plan: Mapping[str, Any],
    latest_review: Mapping[str, Any] | None,
    validation_errors: List[str] | None = None,
) -> str:
    fast_manifest = fast_group_manifest(artifact_dir)
    promotion_manifest = load_manifest_if_present(promotion_artifact_dir) if promotion_artifact_dir is not None else {}
    last_keep = fast_group_keep_reference(artifact_dir, "dev")
    sections = [
        "# Strategic Review",
        "",
        f"Loop: `{loop_id}`",
        f"Completed iteration: `{iteration}`",
        f"Branch: `{current_branch()}`",
        f"Commit: `{current_commit()}`",
        "",
        "You are reviewing the search direction of the autoresearch loop.",
        "Do not propose code directly. Diagnose whether the loop is exploring the right region and what it should try next.",
        "",
        "Return concise markdown under 180 words with exactly these sections and headings:",
        "1. `## Direction`",
        "2. `## Failure Pattern`",
        "3. `## One-Line Guidance`",
        "4. `## Next Search Region`",
        "5. `## Next 3 Trial Ideas`",
        "Use one short paragraph for each section and exactly three numbered trial ideas.",
        "Each trial idea must name one search family from the catalog and one concrete bounded change.",
        "Do not use code fences.",
        "",
        "## Fast Benchmark",
        "```json",
        json.dumps(compact_manifest_summary(fast_manifest), indent=2),
        "```",
        "",
        "## Promotion Benchmark",
        "```json",
        json.dumps(compact_manifest_summary(promotion_manifest), indent=2),
        "```",
        "",
        "## Last Fast Keep",
        "```json",
        json.dumps(compact_keep_summary(last_keep), indent=2),
        "```",
        "",
        "## Recent Global Results",
        recent_results_markdown(limit=PROMPT_RESULTS_LIMIT),
        "",
        "## Current Loop Results",
        recent_loop_results_markdown(loop_results, limit=PROMPT_RESULTS_LIMIT),
        "",
        "## Search Family Catalog",
        "\n".join(f"- `{family}`: {FAMILY_SPECS[family]}" for family in AUTONOMOUS_FAMILY_ORDER),
        "",
        "## Local Playbook Rules",
        playbook_general_excerpt(max_chars=PROMPT_GENERAL_RULES_CHARS),
        "",
        "## Local Playbook Family Cards",
        playbook_family_cards(
            list(search_plan.get("target_families", []))[:4]
            or preferred_search_families(loop_results)[:4]
            or list(AUTONOMOUS_FAMILY_ORDER[:4]),
            max_chars=PROMPT_FAMILY_CARDS_CHARS,
        ),
        "",
        "## Search State",
        f"- Tried families so far: {', '.join(f'`{family}`' for family in tried_search_families(loop_results)) if tried_search_families(loop_results) else '_none yet_'}",
        family_feedback_markdown(loop_results),
        "",
        "## Search Phase",
        search_phase_markdown(search_plan),
        "",
        "## Current Strategy",
        "```python",
        strategy_prompt_excerpt(),
        "```",
    ]
    if latest_review is not None:
        sections.extend(
            [
                "",
                "## Prior Strategic Review",
                "```markdown",
                str(latest_review.get("content", "")).strip(),
                "```",
            ]
        )
    if validation_errors:
        sections.extend(
            [
                "",
                "## Validation Errors To Avoid",
                *[f"- {error}" for error in compact_validation_errors(validation_errors)],
            ]
        )
    return "\n".join(sections) + "\n"


def render_prompt(run_id: str, label: str) -> str:
    fast_manifest = fast_group_manifest(FAST_ARTIFACT_DIR)
    promotion_manifest = load_manifest_if_present(DEFAULT_PROMOTION_ARTIFACT_DIR)
    last_keep = fast_group_keep_reference(FAST_ARTIFACT_DIR, "dev")
    branch = current_branch()
    commit = current_commit()
    dirty = classify_dirty_paths()
    keep_summary = "_No keep result yet._"
    if last_keep is not None:
        keep_summary = json.dumps(last_keep, indent=2)

    promotion_block = "No promotion benchmark configured yet."
    if promotion_manifest:
        promotion_block = f"```json\n{json.dumps(promotion_manifest, indent=2)}\n```"

    return f"""# Autoresearch Trial Package

Run id: `{run_id}`
Label: `{label}`
Branch: `{branch}`
Commit: `{commit}`

## Objective

Make exactly one reranking experiment against the frozen paired fast-loop benchmark.
Edit only `rerank_strategy.py`.

## Required Context

Read these files before changing anything:

- `README.md`
- `program.md`
- `docs/reranking_playbook.md`
- `docs/metric-policy.md`
- `prepare.py`
- `train.py`
- `run.py`
- `rerank_strategy.py`

## Fast Benchmark

```json
{json.dumps(fast_manifest, indent=2)}
```

## Promotion Benchmark

{promotion_block}

## Last Keep

```json
{keep_summary}
```

## Recent Results

{recent_results_markdown()}

## Dirty State Check

```json
{json.dumps(dirty, indent=2)}
```

## Strategy Snapshot

```python
{strategy_excerpt()}
```

## Trial Instructions

1. Propose one small, testable change inside `rerank_strategy.py`.
2. Keep the change bounded and simple.
3. After editing, run:
   - `python autoresearch_driver.py run-once --label "{label}" --run-id "{run_id}"`
4. The driver will:
   - run the fast benchmark on `dev`
   - rerun 2 more times if the first result qualifies for keep
   - require the mean `ndcg@10` gain to stay above threshold
   - optionally run the promotion benchmark if one is configured
   - reset the experiment commit on final `discard` or `crash` when git state is safe

## Protocol Source

```markdown
{PROGRAM_PATH.read_text(encoding="utf-8")}
```
"""


def write_prompt_package(run_id: str, label: str) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = run_dir / "prompt.md"
    prompt_path.write_text(render_prompt(run_id=run_id, label=label), encoding="utf-8")
    return prompt_path


def default_run_id(label: str) -> str:
    safe_label = label.replace(" ", "-")
    return f"{now_stamp()}-{safe_label}"


def setup_branch(tag: str, base_ref: str) -> str:
    if not has_git():
        raise RuntimeError("Git is required for setup.")
    require_safe_worktree()
    branch = f"autoresearch/{tag}"
    if branch_exists(branch):
        if current_branch() != branch:
            git(["switch", branch])
    else:
        git(["switch", "-c", branch, base_ref])
    ensure_results_header()
    return branch


def maybe_commit_strategy(label: str) -> Dict[str, str | None]:
    if not has_git():
        return {"starting_commit": None, "experiment_commit": None}

    require_safe_worktree()
    starting_commit = current_commit()
    dirty = classify_dirty_paths()
    has_strategy_changes = bool(dirty["tracked_relevant"] or dirty["untracked_relevant"])
    if not has_strategy_changes:
        return {"starting_commit": starting_commit, "experiment_commit": None}

    git(["add", "--", "rerank_strategy.py"])
    git(["commit", "-m", f"autoresearch: {label}"])
    return {"starting_commit": starting_commit, "experiment_commit": current_commit()}


def reset_to_commit(commit: str) -> None:
    git(["reset", "--hard", commit])


def aggregate_metrics(payloads: List[Mapping[str, object]]) -> Dict[str, object]:
    aggregate = dict(payloads[0]["metrics"])
    for key in ("ndcg@10", "recall@100", "latency_p95_ms", "mrr@10"):
        aggregate[key] = mean(float(payload["metrics"].get(key, 0.0)) for payload in payloads)
    aggregate["peak_memory_mb"] = max(float(payload["metrics"].get("peak_memory_mb", 0.0)) for payload in payloads)
    aggregate["cost_usd"] = sum(float(payload["metrics"].get("cost_usd", 0.0)) for payload in payloads)
    aggregate["total_seconds"] = sum(float(payload["metrics"].get("total_seconds", 0.0)) for payload in payloads)
    aggregate["evaluation_runs"] = len(payloads)
    return aggregate


def aggregate_fast_group_metrics(child_payloads: List[Mapping[str, object]]) -> Dict[str, object]:
    if not child_payloads:
        raise ValueError("fast group must contain at least one child payload")
    aggregate = dict(child_payloads[0]["metrics"])
    for key in ("ndcg@10", "recall@100", "mrr@10"):
        aggregate[key] = mean(float(payload["metrics"].get(key, 0.0)) for payload in child_payloads)
    aggregate["latency_p95_ms"] = max(float(payload["metrics"].get("latency_p95_ms", 0.0)) for payload in child_payloads)
    aggregate["peak_memory_mb"] = max(float(payload["metrics"].get("peak_memory_mb", 0.0)) for payload in child_payloads)
    aggregate["cost_usd"] = sum(float(payload["metrics"].get("cost_usd", 0.0)) for payload in child_payloads)
    aggregate["total_seconds"] = sum(float(payload["metrics"].get("total_seconds", 0.0)) for payload in child_payloads)
    aggregate["num_queries"] = sum(float(payload["metrics"].get("num_queries", 0.0)) for payload in child_payloads)
    aggregate["evaluation_runs"] = len(child_payloads)
    aggregate["group_children"] = [
        {
            "benchmark_name": payload.get("benchmark_name"),
            "artifact_dir": payload.get("artifact_dir"),
            "ndcg@10": payload["metrics"].get("ndcg@10"),
            "recall@100": payload["metrics"].get("recall@100"),
            "latency_p95_ms": payload["metrics"].get("latency_p95_ms"),
            "mrr@10": payload["metrics"].get("mrr@10"),
            "peak_memory_mb": payload["metrics"].get("peak_memory_mb"),
            "total_seconds": payload["metrics"].get("total_seconds"),
        }
        for payload in child_payloads
    ]
    return aggregate


def build_repeat_acceptance_decision(
    attempts: List[Mapping[str, object]],
    reference_keep: Mapping[str, str] | None,
    min_ndcg_gain: float,
) -> Dict[str, object]:
    if reference_keep is None:
        return dict(attempts[0]["decision"])

    crashed = [payload for payload in attempts if payload["decision"]["status"] == "crash"]
    if crashed:
        return {
            "status": "discard",
            "reason": "repeat acceptance failed: repeated run crashed",
            "guardrails": {},
            "deltas": {},
        }

    recall_failures = []
    latency_failures = []
    peak_memory_failures = []
    cost_failures = []
    for payload in attempts[1:]:
        guardrails = payload["decision"].get("guardrails", {})
        if not guardrails.get("recall", {}).get("passed", True):
            recall_failures.append(payload["run_id"])
        if not guardrails.get("latency", {}).get("passed", True):
            latency_failures.append(payload["run_id"])
        if not guardrails.get("peak_memory", {}).get("passed", True):
            peak_memory_failures.append(payload["run_id"])
        if not guardrails.get("cost_usd", {}).get("passed", True):
            cost_failures.append(payload["run_id"])

    if recall_failures:
        return {
            "status": "discard",
            "reason": f"repeat acceptance failed: recall regressed in {', '.join(recall_failures)}",
            "guardrails": {},
            "deltas": {},
        }

    if latency_failures:
        return {
            "status": "discard",
            "reason": f"repeat acceptance failed: latency regressed in {', '.join(latency_failures)}",
            "guardrails": {},
            "deltas": {},
        }

    if peak_memory_failures:
        return {
            "status": "discard",
            "reason": f"repeat acceptance failed: peak memory regressed in {', '.join(peak_memory_failures)}",
            "guardrails": {},
            "deltas": {},
        }

    if cost_failures:
        return {
            "status": "discard",
            "reason": f"repeat acceptance failed: cost regressed in {', '.join(cost_failures)}",
            "guardrails": {},
            "deltas": {},
        }

    mean_ndcg = mean(float(payload["metrics"]["ndcg@10"]) for payload in attempts)
    previous_ndcg = row_metric(reference_keep, "ndcg@10") or 0.0
    ndcg_gain = mean_ndcg - previous_ndcg
    if ndcg_gain < min_ndcg_gain:
        return {
            "status": "discard",
            "reason": "repeat acceptance failed: mean ndcg@10 gain fell below threshold",
            "guardrails": {},
            "deltas": {"ndcg@10": ndcg_gain},
        }

    return {
        "status": "keep",
        "reason": "mean ndcg@10 gain survived repeated evaluation",
        "guardrails": {},
        "deltas": {"ndcg@10": ndcg_gain},
    }


def append_trial_records(results_path: Path, trial: Mapping[str, object]) -> None:
    for payload in trial["attempts"]:
        append_result(results_path, result_row_from_payload(payload, record_type="attempt"))
    append_result(results_path, result_row_from_payload(trial["summary"], record_type="summary"))


def benchmark_keep_reference(artifact_dir: Path, split: str) -> Dict[str, str] | None:
    manifest = load_manifest_if_present(artifact_dir)
    if not manifest:
        return None
    return last_keep_result(
        RESULTS_PATH,
        benchmark_name=str(manifest.get("dataset", "")),
        benchmark_manifest_hash=manifest_hash(artifact_dir),
        harness_version=CURRENT_HARNESS_VERSION,
        split=split,
    )


def execute_fast_group_attempt(
    *,
    artifact_root: Path,
    results_path: Path,
    label: str,
    split: str,
    run_id: str,
    run_dir: Path,
    parent_commit: str | None,
    commit: str | None,
    budget_seconds: float,
    min_ndcg_gain: float,
    max_recall_rel_drop: float,
    max_latency_rel_increase: float,
    max_peak_memory_rel_increase: float,
    max_cost_usd_increase: float,
    reference_keep: Mapping[str, str] | None = None,
) -> Dict[str, object]:
    specs = configured_fast_benchmark_specs(artifact_root)
    if not specs:
        raise RuntimeError(f"no configured fast benchmarks found under {artifact_root}")

    resolved_reference = reference_keep or fast_group_keep_reference(artifact_root, split)
    child_trials = []
    child_crashes: List[str] = []
    for spec in specs:
        child_payload = execute_run(
            artifact_dir=spec["artifact_dir"],
            results_path=results_path,
            runs_dir=RUNS_DIR,
            run_dir=run_dir / spec["slug"],
            label=f"{label}/{spec['slug']}",
            run_id=f"{run_id}-{spec['slug']}",
            split=split,
            budget_seconds=budget_seconds,
            min_ndcg_gain=min_ndcg_gain,
            max_recall_rel_drop=max_recall_rel_drop,
            max_latency_rel_increase=max_latency_rel_increase,
            max_peak_memory_rel_increase=max_peak_memory_rel_increase,
            max_cost_usd_increase=max_cost_usd_increase,
            parent_commit=parent_commit,
            commit=commit,
            append_results_row=False,
            reference_keep=benchmark_keep_reference(spec["artifact_dir"], split),
        )
        child_trials.append(child_payload)
        if str(child_payload["decision"]["status"]) == "crash":
            child_crashes.append(str(child_payload.get("benchmark_name", spec["slug"])))

    identity = fast_group_identity(artifact_root=artifact_root, split=split)
    aggregate = aggregate_fast_group_metrics(child_trials)
    aggregate.update(identity)
    if child_crashes:
        decision = {
            "status": "crash",
            "reason": "fast benchmark child crashed: " + ", ".join(child_crashes),
            "guardrails": {},
            "deltas": {},
        }
    else:
        decision = build_decision(
            metrics=aggregate,
            previous_keep=resolved_reference,
            min_ndcg_gain=min_ndcg_gain,
            max_recall_rel_drop=max_recall_rel_drop,
            max_latency_rel_increase=max_latency_rel_increase,
            max_peak_memory_rel_increase=max_peak_memory_rel_increase,
            max_cost_usd_increase=max_cost_usd_increase,
        )

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "run_id": run_id,
        "parent_commit": parent_commit,
        "commit": commit or current_commit(),
        "label": label,
        "split": split,
        "run_dir": str(run_dir),
        "artifact_dir": str(artifact_root),
        "reference_keep": resolved_reference,
        "metrics": aggregate,
        "decision": decision,
        "child_trials": child_trials,
        **identity,
    }


def run_fast_group_trial(
    *,
    artifact_root: Path,
    results_path: Path,
    label: str,
    split: str,
    run_id: str,
    run_dir: Path,
    parent_commit: str | None,
    commit: str | None,
    budget_seconds: float,
    min_ndcg_gain: float,
    max_recall_rel_drop: float,
    max_latency_rel_increase: float,
    max_peak_memory_rel_increase: float,
    max_cost_usd_increase: float,
    repeat_keep_runs: int,
) -> Dict[str, object]:
    primary = execute_fast_group_attempt(
        artifact_root=artifact_root,
        results_path=results_path,
        label=label,
        split=split,
        run_id=run_id,
        run_dir=run_dir,
        parent_commit=parent_commit,
        commit=commit,
        budget_seconds=budget_seconds,
        min_ndcg_gain=min_ndcg_gain,
        max_recall_rel_drop=max_recall_rel_drop,
        max_latency_rel_increase=max_latency_rel_increase,
        max_peak_memory_rel_increase=max_peak_memory_rel_increase,
        max_cost_usd_increase=max_cost_usd_increase,
    )
    attempts = [primary]
    summary_decision = dict(primary["decision"])

    if primary["decision"]["status"] == "keep" and primary["reference_keep"] is not None and repeat_keep_runs > 0:
        for index in range(repeat_keep_runs):
            attempts.append(
                execute_fast_group_attempt(
                    artifact_root=artifact_root,
                    results_path=results_path,
                    label=f"{label}/recheck-{index + 1}",
                    split=split,
                    run_id=f"{run_id}-recheck-{index + 1}",
                    run_dir=run_dir / f"recheck-{index + 1}",
                    parent_commit=parent_commit,
                    commit=commit,
                    budget_seconds=budget_seconds,
                    min_ndcg_gain=min_ndcg_gain,
                    max_recall_rel_drop=max_recall_rel_drop,
                    max_latency_rel_increase=max_latency_rel_increase,
                    max_peak_memory_rel_increase=max_peak_memory_rel_increase,
                    max_cost_usd_increase=max_cost_usd_increase,
                    reference_keep=primary["reference_keep"],
                )
            )
        summary_decision = build_repeat_acceptance_decision(
            attempts=attempts,
            reference_keep=primary["reference_keep"],
            min_ndcg_gain=min_ndcg_gain,
        )

    summary_payload = dict(primary)
    summary_payload["metrics"] = aggregate_metrics(attempts)
    summary_payload["decision"] = summary_decision
    summary_payload["run_id"] = run_id
    summary_payload["label"] = label
    return {
        "attempts": attempts,
        "summary": summary_payload,
    }


def run_strategy_review(
    *,
    loop_id: str,
    iteration: int,
    artifact_dir: Path,
    promotion_artifact_dir: Path | None,
    loop_results: List[Mapping[str, Any]],
    latest_review: Mapping[str, Any] | None,
    review_brain,
    loop_dir: Path,
    trigger: str,
) -> Dict[str, Any]:
    review_id = f"review-{iteration:02d}"
    response_path = loop_dir / f"{review_id}.md"
    validation_errors: List[str] = []
    raw_attempts: List[Dict[str, Any]] = []
    search_plan = build_search_plan(
        loop_results,
        next_iteration=iteration + 1,
        total_iterations=max(iteration + 1, len(loop_results) + 1),
    )

    for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
        prompt = build_strategy_review_prompt(
            loop_id=loop_id,
            iteration=iteration,
            artifact_dir=artifact_dir,
            promotion_artifact_dir=promotion_artifact_dir,
            loop_results=loop_results,
            search_plan=search_plan,
            latest_review=latest_review,
            validation_errors=validation_errors,
        )
        prompt_path = loop_dir / f"{review_id}-prompt-{attempt}.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        raw_content = review_brain.complete(prompt)
        content = strip_reasoning(raw_content).strip()
        raw_attempts.append(
            {
                "attempt": attempt,
                "prompt_path": str(prompt_path),
                "raw_content": raw_content,
                "content": content,
            }
        )
        (loop_dir / f"{review_id}-response-{attempt}.md").write_text(content + "\n", encoding="utf-8")
        try:
            validate_review_content(content)
            response_path.write_text(content + "\n", encoding="utf-8")
            return {
                "review_id": review_id,
                "iteration": iteration,
                "trigger": trigger,
                "content": content,
                "raw_content": raw_content,
                "brain": dict(review_brain.config()),
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "attempts": raw_attempts,
            }
        except ValueError as exc:
            validation_errors.append(str(exc))

    fallback_content = fallback_review_content(
        loop_results=loop_results,
        artifact_dir=artifact_dir,
        promotion_artifact_dir=promotion_artifact_dir,
    )
    response_path.write_text(fallback_content + "\n", encoding="utf-8")
    return {
        "review_id": review_id,
        "iteration": iteration,
        "trigger": trigger,
        "content": fallback_content,
        "raw_content": raw_attempts[-1]["raw_content"] if raw_attempts else "",
        "brain": dict(review_brain.config()),
        "prompt_path": str(loop_dir / f"{review_id}-prompt-{len(raw_attempts)}.md") if raw_attempts else "",
        "response_path": str(response_path),
        "attempts": raw_attempts,
        "fallback_used": True,
        "validation_errors": validation_errors,
    }


def run_trial(
    *,
    artifact_dir: Path,
    results_path: Path,
    label: str,
    split: str,
    run_id: str,
    run_dir: Path,
    parent_commit: str | None,
    commit: str | None,
    budget_seconds: float,
    min_ndcg_gain: float,
    max_recall_rel_drop: float,
    max_latency_rel_increase: float,
    max_peak_memory_rel_increase: float,
    max_cost_usd_increase: float,
    repeat_keep_runs: int,
) -> Dict[str, object]:
    primary = execute_run(
        artifact_dir=artifact_dir,
        results_path=results_path,
        runs_dir=RUNS_DIR,
        run_dir=run_dir,
        label=label,
        run_id=run_id,
        split=split,
        budget_seconds=budget_seconds,
        min_ndcg_gain=min_ndcg_gain,
        max_recall_rel_drop=max_recall_rel_drop,
        max_latency_rel_increase=max_latency_rel_increase,
        max_peak_memory_rel_increase=max_peak_memory_rel_increase,
        max_cost_usd_increase=max_cost_usd_increase,
        parent_commit=parent_commit,
        commit=commit,
        append_results_row=False,
    )
    attempts = [primary]
    summary_decision = dict(primary["decision"])

    if primary["decision"]["status"] == "keep" and primary["reference_keep"] is not None and repeat_keep_runs > 0:
        for index in range(repeat_keep_runs):
            attempt_run_id = f"{run_id}-recheck-{index + 1}"
            attempt_run_dir = run_dir / f"recheck-{index + 1}"
            attempts.append(
                execute_run(
                    artifact_dir=artifact_dir,
                    results_path=results_path,
                    runs_dir=RUNS_DIR,
                    run_dir=attempt_run_dir,
                    label=f"{label}/recheck-{index + 1}",
                    run_id=attempt_run_id,
                    split=split,
                    budget_seconds=budget_seconds,
                    min_ndcg_gain=min_ndcg_gain,
                    max_recall_rel_drop=max_recall_rel_drop,
                    max_latency_rel_increase=max_latency_rel_increase,
                    max_peak_memory_rel_increase=max_peak_memory_rel_increase,
                    max_cost_usd_increase=max_cost_usd_increase,
                    parent_commit=parent_commit,
                    commit=commit,
                    append_results_row=False,
                    reference_keep=primary["reference_keep"],
                )
            )
        summary_decision = build_repeat_acceptance_decision(
            attempts=attempts,
            reference_keep=primary["reference_keep"],
            min_ndcg_gain=min_ndcg_gain,
        )

    summary_payload = dict(primary)
    summary_payload["metrics"] = aggregate_metrics(attempts)
    summary_payload["decision"] = summary_decision
    summary_payload["run_id"] = run_id
    summary_payload["label"] = label
    return {
        "attempts": attempts,
        "summary": summary_payload,
    }


def run_once(
    *,
    label: str,
    split: str,
    run_id: str,
    skip_git: bool,
    artifact_dir: Path,
    promotion_artifact_dir: Path | None,
    budget_seconds: float,
    promotion_budget_seconds: float,
    min_ndcg_gain: float,
    max_recall_rel_drop: float,
    max_latency_rel_increase: float,
    max_peak_memory_rel_increase: float,
    max_cost_usd_increase: float,
    repeat_keep_runs: int,
) -> Dict[str, object]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = run_dir / "prompt.md"
    if not prompt_path.exists():
        write_prompt_package(run_id=run_id, label=label)

    commit_info = {"starting_commit": None, "experiment_commit": None}
    if not skip_git:
        commit_info = maybe_commit_strategy(label=label)

    strategy_commit = commit_info.get("experiment_commit") or current_commit()
    fast_trial = run_fast_group_trial(
        artifact_root=artifact_dir,
        results_path=RESULTS_PATH,
        label=label,
        split=split,
        run_id=run_id,
        run_dir=run_dir,
        parent_commit=commit_info.get("starting_commit"),
        commit=strategy_commit,
        budget_seconds=budget_seconds,
        min_ndcg_gain=min_ndcg_gain,
        max_recall_rel_drop=max_recall_rel_drop,
        max_latency_rel_increase=max_latency_rel_increase,
        max_peak_memory_rel_increase=max_peak_memory_rel_increase,
        max_cost_usd_increase=max_cost_usd_increase,
        repeat_keep_runs=repeat_keep_runs,
    )

    promotion_trial = None
    overall_status = str(fast_trial["summary"]["decision"]["status"])
    overall_reason = str(fast_trial["summary"]["decision"]["reason"])
    promotion_keep = None
    if promotion_artifact_dir is not None and promotion_artifact_dir.exists():
        promotion_keep = benchmark_keep_reference(promotion_artifact_dir, split)

    if overall_status == "keep" and promotion_artifact_dir is not None and promotion_keep is not None:
        promotion_trial = run_trial(
            artifact_dir=promotion_artifact_dir,
            results_path=RESULTS_PATH,
            label=f"{label}/promotion",
            split=split,
            run_id=f"{run_id}-promotion",
            run_dir=run_dir / "promotion",
            parent_commit=commit_info.get("starting_commit"),
            commit=strategy_commit,
            budget_seconds=promotion_budget_seconds,
            min_ndcg_gain=min_ndcg_gain,
            max_recall_rel_drop=max_recall_rel_drop,
            max_latency_rel_increase=max_latency_rel_increase,
            max_peak_memory_rel_increase=max_peak_memory_rel_increase,
            max_cost_usd_increase=max_cost_usd_increase,
            repeat_keep_runs=0,
        )
        promotion_status = str(promotion_trial["summary"]["decision"]["status"])
        if promotion_status != "keep":
            overall_status = "discard"
            overall_reason = f"promotion benchmark rejected candidate: {promotion_trial['summary']['decision']['reason']}"
    elif overall_status == "keep" and promotion_artifact_dir is not None and promotion_artifact_dir.exists():
        overall_reason = "promotion benchmark configured but unbaselined; validation skipped"

    fast_trial["summary"]["decision"] = {
        **fast_trial["summary"]["decision"],
        "status": overall_status,
        "reason": overall_reason,
    }

    append_trial_records(RESULTS_PATH, fast_trial)
    if promotion_trial is not None:
        append_trial_records(RESULTS_PATH, promotion_trial)

    if not skip_git:
        reset_target = commit_info.get("starting_commit")
        experiment_commit = commit_info.get("experiment_commit")
        if overall_status in {"discard", "crash"} and reset_target and experiment_commit:
            reset_to_commit(reset_target)

    summary_path = run_dir / "driver-summary.json"
    summary_payload = {
        "run_id": run_id,
        "label": label,
        "split": split,
        "skip_git": skip_git,
        "git": commit_info,
        "fast_trial": fast_trial,
        "promotion_trial": promotion_trial,
        "overall": {
            "status": overall_status,
            "reason": overall_reason,
            "strategy_commit": strategy_commit,
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    summary_payload["history_exports"] = refresh_history()
    return summary_payload


def run_autonomous_loop(
    *,
    iterations: int,
    label_prefix: str,
    split: str,
    skip_git: bool,
    artifact_dir: Path,
    promotion_artifact_dir: Path | None,
    budget_seconds: float,
    promotion_budget_seconds: float,
    min_ndcg_gain: float,
    max_recall_rel_drop: float,
    max_latency_rel_increase: float,
    max_peak_memory_rel_increase: float,
    max_cost_usd_increase: float,
    repeat_keep_runs: int,
    brain_backend: str,
    brain_model: str,
    controller_model: str | None,
    planner_model: str | None,
    proposal_model: str | None,
    review_model: str | None,
    brain_max_tokens: int,
    brain_temperature: float,
    brain_top_p: float,
    brain_llama_n_ctx: int,
    brain_llama_n_gpu_layers: int,
    controller_llama_n_gpu_layers: int | None,
    brain_llama_type_k: int | None,
    brain_llama_type_v: int | None,
    controller_gpu: int | None,
    proposal_gpu: int | None,
    keep_controller_loaded: bool,
    controller_warm_start: bool,
    review_max_tokens: int,
    review_first_interval: int,
    review_repeat_interval: int,
    explore_trials_per_family: int,
    exploit_top_families: int,
    max_draft_attempts: int,
    stop_after_keep: bool,
) -> Dict[str, object]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if max_draft_attempts < 1:
        raise ValueError("max_draft_attempts must be at least 1")
    if explore_trials_per_family < 1:
        raise ValueError("explore_trials_per_family must be at least 1")
    if exploit_top_families < 1:
        raise ValueError("exploit_top_families must be at least 1")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not skip_git:
        require_safe_worktree()

    loop_id = default_run_id(label_prefix)
    loop_dir = RUNS_DIR / loop_id
    loop_dir.mkdir(parents=True, exist_ok=True)

    controller_model_name = controller_model or planner_model or review_model or brain_model
    proposal_model_name = proposal_model or brain_model
    controller_gpu_layers = (
        brain_llama_n_gpu_layers
        if controller_llama_n_gpu_layers is None
        else controller_llama_n_gpu_layers
    )

    controller_brain = build_brain(
        backend=brain_backend,
        model_name=controller_model_name,
        max_tokens=brain_max_tokens,
        temperature=brain_temperature,
        top_p=brain_top_p,
        llama_n_ctx=brain_llama_n_ctx,
        llama_n_gpu_layers=controller_gpu_layers,
        llama_type_k=brain_llama_type_k,
        llama_type_v=brain_llama_type_v,
        preferred_gpu=controller_gpu,
        reasoning_mode=DEFAULT_BRAIN_REASONING_MODE,
        keep_loaded=keep_controller_loaded,
        warm_start=controller_warm_start,
    )
    can_share_controller = (
        proposal_model_name == controller_model_name
        and controller_gpu_layers == brain_llama_n_gpu_layers
        and controller_gpu == proposal_gpu
        and not keep_controller_loaded
    )
    if can_share_controller:
        proposal_brain = controller_brain
    else:
        proposal_brain = build_brain(
            backend=brain_backend,
            model_name=proposal_model_name,
            max_tokens=brain_max_tokens,
            temperature=brain_temperature,
            top_p=brain_top_p,
            llama_n_ctx=brain_llama_n_ctx,
            llama_n_gpu_layers=brain_llama_n_gpu_layers,
            llama_type_k=brain_llama_type_k,
            llama_type_v=brain_llama_type_v,
            preferred_gpu=proposal_gpu,
            reasoning_mode=DEFAULT_BRAIN_REASONING_MODE,
            keep_loaded=False,
            warm_start=False,
        )

    def unload_brains(*brains: object, force: bool = False) -> None:
        seen: set[int] = set()
        for brain in brains:
            if brain is None:
                continue
            identifier = id(brain)
            if identifier in seen:
                continue
            seen.add(identifier)
            try:
                brain.unload(force=force)
            except TypeError:
                brain.unload()

    loop_payload: Dict[str, Any] = {
        "loop_id": loop_id,
        "iterations_requested": iterations,
        "label_prefix": label_prefix,
        "split": split,
        "skip_git": skip_git,
        "controller_brain": dict(controller_brain.config()),
        "proposal_brain": dict(proposal_brain.config()),
        "policy": {
            "min_ndcg_gain": min_ndcg_gain,
            "max_recall_rel_drop": max_recall_rel_drop,
            "max_latency_rel_increase": max_latency_rel_increase,
            "max_peak_memory_rel_increase": max_peak_memory_rel_increase,
            "max_cost_usd_increase": max_cost_usd_increase,
            "repeat_keep_runs": repeat_keep_runs,
            "review_max_tokens": review_max_tokens,
            "review_first_interval": review_first_interval,
            "review_repeat_interval": review_repeat_interval,
            "explore_trials_per_family": explore_trials_per_family,
            "exploit_top_families": exploit_top_families,
            "family_cooldown_window": 5,
            "family_cooldown_threshold": 3,
            "family_cooldown_duration": 5,
        },
        "search_space": {
            "families": FAMILY_SPECS,
            "active_families": list(AUTONOMOUS_FAMILY_ORDER),
        },
        "results": [],
        "reviews": [],
    }
    refresh_loop_payload(loop_payload)
    write_json(loop_dir / "loop-config.json", loop_payload)

    last_outcome: Dict[str, Any] | None = None
    latest_review: Dict[str, Any] | None = None
    keeps = 0

    try:
        for iteration in range(1, iterations + 1):
            emit_loop_progress(
                iteration=iteration,
                total_iterations=iterations,
                phase="starting",
                keeps=keeps,
            )
            run_id = f"{loop_id}-iter-{iteration:02d}"
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            original_strategy = STRATEGY_PATH.read_text(encoding="utf-8")
            validation_errors: List[str] = []
            proposal = None
            proposal_attempts = []
            idea_attempts = []
            selected_idea: ProposalIdea | None = None
            reviewer_selection: Dict[str, str] | None = None
            search_plan = build_search_plan(
                loop_payload["results"],
                next_iteration=iteration,
                total_iterations=iterations,
                explore_trials_per_family=explore_trials_per_family,
                exploit_top_families=exploit_top_families,
            )

            for draft_attempt in range(1, max_draft_attempts + 1):
                emit_loop_progress(
                    iteration=iteration,
                    total_iterations=iterations,
                    phase="draft",
                    keeps=keeps,
                    latest_status=f"draft-{draft_attempt}/{max_draft_attempts}",
                )
                raw_response = ""
                synthesized_from_idea = False
                idea_prompt = build_idea_prompt(
                    loop_id=loop_id,
                    iteration=iteration,
                    total_iterations=iterations,
                    artifact_dir=artifact_dir,
                    promotion_artifact_dir=promotion_artifact_dir,
                    loop_results=loop_payload["results"],
                    search_plan=search_plan,
                    last_outcome=last_outcome,
                    latest_review=latest_review,
                    validation_errors=validation_errors,
                )
                (run_dir / f"idea-prompt-{draft_attempt}.md").write_text(idea_prompt, encoding="utf-8")
                try:
                    idea_raw = controller_brain.complete(idea_prompt)
                    (run_dir / f"idea-response-{draft_attempt}.md").write_text(idea_raw, encoding="utf-8")
                    recovered_idea_error = ""
                    try:
                        parsed_candidate_ideas = parse_candidate_ideas(idea_raw)
                    except ValueError as exc:
                        recovered_idea_error = str(exc)
                        parsed_candidate_ideas = recover_candidate_ideas(
                            idea_raw,
                            loop_results=loop_payload["results"],
                        )
                        if not parsed_candidate_ideas:
                            raise
                    candidate_ideas = supplement_candidate_ideas(
                        parsed_candidate_ideas,
                        loop_results=loop_payload["results"],
                        prioritized_families=list(search_plan.get("target_families", [])) + list(search_plan.get("backup_families", [])),
                        target_count=DEFAULT_IDEA_CANDIDATES,
                    )
                    valid_ideas: List[ProposalIdea] = []
                    idea_rejections: List[Dict[str, Any]] = []
                    for idea in candidate_ideas:
                        reasons = novelty_rejections(idea, loop_payload["results"])
                        if reasons:
                            idea_rejections.append(
                                {
                                    **idea_to_mapping(idea),
                                    "status": "rejected",
                                    "reasons": reasons,
                                }
                            )
                            continue
                        valid_ideas.append(idea)
                    if not valid_ideas:
                        raise ValueError(
                            "proposer did not yield a novel idea. "
                            + "; ".join(reason for item in idea_rejections for reason in item.get("reasons", []))
                        )

                    review_payload = {
                        "draft_attempt": draft_attempt,
                        "candidate_ideas": [idea_to_mapping(idea) for idea in candidate_ideas],
                        "accepted_ideas": [idea_to_mapping(idea) for idea in valid_ideas],
                        "rejected_ideas": idea_rejections,
                    }
                    if recovered_idea_error:
                        review_payload["candidate_recovery"] = recovered_idea_error
                    idea_attempts.append(review_payload)

                    emit_loop_progress(
                        iteration=iteration,
                        total_iterations=iterations,
                        phase="reviewer",
                        keeps=keeps,
                    )
                    reviewer_prompt = build_reviewer_prompt(
                        loop_id=loop_id,
                        iteration=iteration,
                        ideas=valid_ideas,
                        loop_results=loop_payload["results"],
                        search_plan=search_plan,
                        latest_review=latest_review,
                    )
                    (run_dir / f"reviewer-prompt-{draft_attempt}.md").write_text(reviewer_prompt, encoding="utf-8")
                    reviewer_raw = controller_brain.complete(reviewer_prompt)
                    (run_dir / f"reviewer-response-{draft_attempt}.md").write_text(reviewer_raw, encoding="utf-8")
                    try:
                        reviewer_selection = parse_reviewer_selection(reviewer_raw, valid_ideas)
                    except ValueError as exc:
                        reviewer_selection = fallback_reviewer_selection(
                            valid_ideas,
                            raw_text=reviewer_raw,
                            error=str(exc),
                        )
                    selected_idea = next(
                        idea for idea in valid_ideas
                        if idea.family == reviewer_selection["family"] and idea.label == reviewer_selection["label"]
                    )

                    proposal_prompt = build_proposal_prompt(
                        loop_id=loop_id,
                        iteration=iteration,
                        total_iterations=iterations,
                        artifact_dir=artifact_dir,
                        promotion_artifact_dir=promotion_artifact_dir,
                        loop_results=loop_payload["results"],
                        search_plan=search_plan,
                        selected_idea=selected_idea,
                        reviewer_selection=reviewer_selection,
                        last_outcome=last_outcome,
                        latest_review=latest_review,
                        validation_errors=validation_errors,
                    )
                    emit_loop_progress(
                        iteration=iteration,
                        total_iterations=iterations,
                        phase="proposal",
                        keeps=keeps,
                    )
                    (run_dir / f"proposal-prompt-{draft_attempt}.md").write_text(proposal_prompt, encoding="utf-8")
                    raw_response = proposal_brain.complete(proposal_prompt)
                    (run_dir / f"brain-response-{draft_attempt}.md").write_text(raw_response, encoding="utf-8")
                    try:
                        proposal = parse_proposal(
                            raw_response,
                            model_name=proposal_brain.config().get("model_name", ""),
                            fallback_label=f"{label_prefix}-iter-{iteration:02d}",
                        )
                    except ProposalFormatError as exc:
                        raise ValueError(
                            "proposal must include a valid metadata JSON block and a full strategy implementation or "
                            f"strategy dict: {exc}"
                        ) from exc
                    if proposal.family and normalize_family(proposal.family) != selected_idea.family:
                        raise ValueError(
                            f"proposal family `{proposal.family}` does not match selected idea family `{selected_idea.family}`"
                        )
                    proposal.family = selected_idea.family
                    if not proposal.changed_keys:
                        proposal.changed_keys = list(selected_idea.changed_keys)
                    if not proposal.primary_mechanism:
                        proposal.primary_mechanism = selected_idea.primary_mechanism
                    if not proposal.why_not_duplicate:
                        proposal.why_not_duplicate = selected_idea.why_not_duplicate
                    if not proposal.expected_ndcg_direction:
                        proposal.expected_ndcg_direction = selected_idea.expected_ndcg_direction
                    if not proposal.expected_recall_direction:
                        proposal.expected_recall_direction = selected_idea.expected_recall_direction
                    if not proposal.expected_latency_direction:
                        proposal.expected_latency_direction = selected_idea.expected_latency_direction
                    if not proposal.promotion_risk:
                        proposal.promotion_risk = selected_idea.promotion_risk
                    proposal.strategy_code, synthesized_from_idea = materialize_strategy_code(
                        proposal.strategy_code,
                        base_code=original_strategy,
                        selected_idea=selected_idea,
                    )
                    if proposal.strategy_code.strip() == original_strategy.strip():
                        raise ValueError("proposal did not change rerank_strategy.py")
                    validate_strategy_change_set(
                        original_code=original_strategy,
                        updated_code=proposal.strategy_code,
                        declared_changed_keys=proposal.changed_keys,
                    )
                    validate_proposal_novelty(proposal, loop_payload["results"])
                    STRATEGY_PATH.write_text(proposal.strategy_code, encoding="utf-8")
                    proposal_payload = {
                        "draft_attempt": draft_attempt,
                        "family": proposal.family,
                        "label": proposal.label,
                        "summary": proposal.summary,
                        "hypothesis": proposal.hypothesis,
                        "changed_keys": proposal.changed_keys,
                        "primary_mechanism": proposal.primary_mechanism,
                        "why_recent_attempts_failed": proposal.why_recent_attempts_failed,
                        "why_not_duplicate": proposal.why_not_duplicate,
                        "expected_ndcg_direction": proposal.expected_ndcg_direction,
                        "expected_recall_direction": proposal.expected_recall_direction,
                        "expected_latency_direction": proposal.expected_latency_direction,
                        "promotion_risk": proposal.promotion_risk,
                        "rationale_fingerprint": selected_idea.rationale_fingerprint(),
                        "selected_idea": idea_to_mapping(selected_idea),
                        "reviewer_selection": reviewer_selection,
                        "model_name": proposal.model_name,
                        "strategy_path": str(STRATEGY_PATH),
                        "synthesized_from_idea": synthesized_from_idea,
                    }
                    write_json(run_dir / "proposal.json", proposal_payload)
                    (run_dir / "brain-response.md").write_text(raw_response, encoding="utf-8")
                    proposal_attempts.append({**proposal_payload, "status": "accepted"})
                    break
                except (ProposalFormatError, ValueError, SyntaxError, StopIteration) as exc:
                    validation_errors.append(str(exc))
                    STRATEGY_PATH.write_text(original_strategy, encoding="utf-8")
                    rejected_payload = {
                        "draft_attempt": draft_attempt,
                        "status": "rejected",
                        "error": str(exc),
                    }
                    if proposal is not None:
                        rejected_payload.update(
                            {
                                "family": proposal.family,
                                "label": proposal.label,
                                "summary": proposal.summary,
                                "changed_keys": proposal.changed_keys,
                            }
                        )
                    proposal_attempts.append(
                        rejected_payload
                    )
                    proposal = None
                    selected_idea = None
                    reviewer_selection = None

            if proposal is None:
                summary_payload = {
                    "run_id": run_id,
                    "label": f"{label_prefix}-iter-{iteration:02d}-brain-error",
                    "split": split,
                    "skip_git": skip_git,
                    "git": {"starting_commit": current_commit() if has_git() else None, "experiment_commit": None},
                    "controller_brain": dict(controller_brain.config()),
                    "proposal_brain": dict(proposal_brain.config()),
                    "validation_errors": validation_errors,
                    "idea_attempts": idea_attempts,
                    "proposal_attempts": proposal_attempts,
                    "search_plan": search_plan,
                    "overall": {
                        "status": "brain_error",
                        "reason": "model failed to return a valid rerank_strategy.py proposal after max draft attempts",
                        "strategy_commit": current_commit() if has_git() else None,
                    },
                }
                write_json(run_dir / "driver-summary.json", summary_payload)
                iteration_payload = {
                    "iteration": iteration,
                    "run_id": run_id,
                    "family": "brain-error",
                    "label": f"{label_prefix}-iter-{iteration:02d}-brain-error",
                    "summary": "Brain failed to return a valid rerank_strategy.py proposal after max draft attempts.",
                    "validation_errors": validation_errors,
                    "idea_attempts": idea_attempts,
                    "proposal_attempts": proposal_attempts,
                    "search_plan": search_plan,
                    "overall": summary_payload["overall"],
                }
                loop_payload["results"].append(iteration_payload)
                refresh_loop_payload(loop_payload)
                write_json(loop_dir / "loop-summary.json", loop_payload)
                emit_loop_progress(
                    iteration=iteration,
                    total_iterations=iterations,
                    phase="done",
                    keeps=keeps,
                    latest_status="brain_error",
                    done=True,
                )
                last_outcome = compact_iteration_summary(iteration_payload)
                continue

            unload_brains(controller_brain, proposal_brain)
            emit_loop_progress(
                iteration=iteration,
                total_iterations=iterations,
                phase="evaluation",
                keeps=keeps,
            )
            summary = run_once(
                label=proposal.label,
                split=split,
                run_id=run_id,
                skip_git=skip_git,
                artifact_dir=artifact_dir,
                promotion_artifact_dir=promotion_artifact_dir,
                budget_seconds=budget_seconds,
                promotion_budget_seconds=promotion_budget_seconds,
                min_ndcg_gain=min_ndcg_gain,
                max_recall_rel_drop=max_recall_rel_drop,
                max_latency_rel_increase=max_latency_rel_increase,
                max_peak_memory_rel_increase=max_peak_memory_rel_increase,
                max_cost_usd_increase=max_cost_usd_increase,
                repeat_keep_runs=repeat_keep_runs,
            )
            if skip_git and summary["overall"]["status"] in {"discard", "crash"}:
                STRATEGY_PATH.write_text(original_strategy, encoding="utf-8")
            iteration_payload = {
                "iteration": iteration,
                "run_id": run_id,
                "family": proposal.family,
                "label": proposal.label,
                "summary": proposal.summary,
                "hypothesis": proposal.hypothesis,
                "changed_keys": proposal.changed_keys,
                "primary_mechanism": proposal.primary_mechanism,
                "why_not_duplicate": proposal.why_not_duplicate,
                "expected_ndcg_direction": proposal.expected_ndcg_direction,
                "expected_recall_direction": proposal.expected_recall_direction,
                "expected_latency_direction": proposal.expected_latency_direction,
                "promotion_risk": proposal.promotion_risk,
                "rationale_fingerprint": selected_idea.rationale_fingerprint() if selected_idea is not None else "",
                "search_plan": search_plan,
                "idea_attempts": idea_attempts,
                "proposal_attempts": proposal_attempts,
                "reviewer_selection": reviewer_selection,
                "overall": summary["overall"],
                "fast_metrics": summary["fast_trial"]["summary"]["metrics"],
                "fast_decision": compact_decision_summary(summary["fast_trial"]["summary"]["decision"]),
            }
            if summary["promotion_trial"] is not None:
                iteration_payload["promotion_metrics"] = summary["promotion_trial"]["summary"]["metrics"]
                iteration_payload["promotion_decision"] = compact_decision_summary(
                    summary["promotion_trial"]["summary"]["decision"]
                )
            loop_payload["results"].append(iteration_payload)
            refresh_loop_payload(loop_payload)
            write_json(loop_dir / "loop-summary.json", loop_payload)
            last_outcome = compact_iteration_summary(iteration_payload)

            if strategic_review_due(
                iteration,
                first_interval=review_first_interval,
                repeat_interval=review_repeat_interval,
            ):
                emit_loop_progress(
                    iteration=iteration,
                    total_iterations=iterations,
                    phase="strategic-review",
                    keeps=keeps,
                )
                latest_review = run_strategy_review(
                    loop_id=loop_id,
                    iteration=iteration,
                    artifact_dir=artifact_dir,
                    promotion_artifact_dir=promotion_artifact_dir,
                    loop_results=loop_payload["results"],
                    latest_review=latest_review,
                    review_brain=controller_brain,
                    loop_dir=loop_dir,
                    trigger="cadence",
                )
                loop_payload["reviews"].append(latest_review)
                refresh_loop_payload(loop_payload)
                write_json(loop_dir / "loop-summary.json", loop_payload)
                unload_brains(controller_brain)

            if summary["overall"]["status"] == "keep":
                keeps += 1
            emit_loop_progress(
                iteration=iteration,
                total_iterations=iterations,
                phase="done",
                keeps=keeps,
                latest_status=str(summary["overall"]["status"]),
                done=True,
            )
            if summary["overall"]["status"] == "keep":
                if stop_after_keep:
                    break
    finally:
        unload_brains(controller_brain, proposal_brain, force=True)

    loop_payload["keeps"] = keeps
    loop_payload["completed_iterations"] = len(loop_payload["results"])
    refresh_loop_payload(loop_payload)
    write_json(loop_dir / "loop-summary.json", loop_payload)
    loop_payload["history_exports"] = refresh_history()
    return loop_payload


def status_payload() -> Dict[str, object]:
    last_fast_keep = fast_group_keep_reference(FAST_ARTIFACT_DIR, "dev")
    last_promotion_keep = None
    if (DEFAULT_PROMOTION_ARTIFACT_DIR / "manifest.json").exists():
        last_promotion_keep = benchmark_keep_reference(DEFAULT_PROMOTION_ARTIFACT_DIR, "dev")
    last_milestone_keep = None
    if (DEFAULT_MILESTONE_ARTIFACT_DIR / "manifest.json").exists():
        last_milestone_keep = benchmark_keep_reference(DEFAULT_MILESTONE_ARTIFACT_DIR, "dev")
    report_manifest = load_manifest_if_present(DEFAULT_REPORT_ARTIFACT_DIR)
    milestone_manifest = load_manifest_if_present(DEFAULT_MILESTONE_ARTIFACT_DIR)
    return {
        "branch": current_branch(),
        "commit": current_commit(),
        "fast_manifest": fast_group_manifest(FAST_ARTIFACT_DIR),
        "fast_components": [spec["manifest"] for spec in configured_fast_benchmark_specs(FAST_ARTIFACT_DIR)],
        "promotion_manifest": load_manifest_if_present(DEFAULT_PROMOTION_ARTIFACT_DIR),
        "milestone_manifest": milestone_manifest,
        "report_manifest": report_manifest,
        "last_fast_keep": last_fast_keep,
        "last_promotion_keep": last_promotion_keep,
        "last_milestone_keep": last_milestone_keep,
        "dirty": classify_dirty_paths(),
        "results_count": len(read_results(RESULTS_PATH)),
        "history_exports": {
            "history_dir": str(HISTORY_DIR),
            "dashboard_html": str(HISTORY_DIR / "index.html"),
            "history_json": str(HISTORY_DIR / "history.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Git-native driver for autoresearch-style reranking experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Create or switch to an autoresearch branch and initialize the registry.")
    setup_parser.add_argument("--tag", required=True, help="Branch tag, used as autoresearch/<tag>.")
    setup_parser.add_argument("--base-ref", default="main")

    status_parser = subparsers.add_parser("status", help="Show current autoresearch state.")
    status_parser.add_argument("--json", action="store_true")

    prompt_parser = subparsers.add_parser("package-prompt", help="Write a prompt package for the next trial.")
    prompt_parser.add_argument("--label", required=True)
    prompt_parser.add_argument("--run-id", default=None)

    run_parser = subparsers.add_parser("run-once", help="Commit the current strategy change, run one experiment, and reset on discard.")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--split", default="dev")
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument("--skip-git", action="store_true", help="Skip git commit/reset actions. Useful while the repo is still dirty.")
    run_parser.add_argument("--artifact-dir", type=Path, default=FAST_ARTIFACT_DIR)
    run_parser.add_argument("--promotion-artifact-dir", type=Path, default=None)
    run_parser.add_argument("--budget-seconds", type=float, default=TIME_BUDGET)
    run_parser.add_argument("--promotion-budget-seconds", type=float, default=DEFAULT_PROMOTION_BUDGET_SECONDS)
    run_parser.add_argument("--repeat-keep-runs", type=int, default=2)
    run_parser.add_argument("--min-ndcg-gain", type=float, default=0.002)
    run_parser.add_argument("--max-recall-rel-drop", type=float, default=0.01)
    run_parser.add_argument("--max-latency-rel-increase", type=float, default=0.20)
    run_parser.add_argument("--max-peak-memory-rel-increase", type=float, default=0.20)
    run_parser.add_argument("--max-cost-usd-increase", type=float, default=0.0)

    auto_parser = subparsers.add_parser(
        "auto-loop",
        help="Use a local controller model as the research brain and run iterative strategy experiments.",
    )
    auto_parser.add_argument("--iterations", type=int, default=DEFAULT_AUTO_ITERATIONS)
    auto_parser.add_argument("--label-prefix", default=DEFAULT_LABEL_PREFIX)
    auto_parser.add_argument("--split", default="dev")
    auto_parser.add_argument("--skip-git", action="store_true", help="Skip git commit/reset actions.")
    auto_parser.add_argument("--artifact-dir", type=Path, default=FAST_ARTIFACT_DIR)
    auto_parser.add_argument("--promotion-artifact-dir", type=Path, default=None)
    auto_parser.add_argument("--budget-seconds", type=float, default=TIME_BUDGET)
    auto_parser.add_argument("--promotion-budget-seconds", type=float, default=DEFAULT_PROMOTION_BUDGET_SECONDS)
    auto_parser.add_argument("--repeat-keep-runs", type=int, default=2)
    auto_parser.add_argument("--min-ndcg-gain", type=float, default=0.002)
    auto_parser.add_argument("--max-recall-rel-drop", type=float, default=0.01)
    auto_parser.add_argument("--max-latency-rel-increase", type=float, default=0.20)
    auto_parser.add_argument("--max-peak-memory-rel-increase", type=float, default=0.20)
    auto_parser.add_argument("--max-cost-usd-increase", type=float, default=0.0)
    auto_parser.add_argument("--brain-backend", default=DEFAULT_BRAIN_BACKEND, choices=["auto", "llama-cpp", "transformers"])
    auto_parser.add_argument("--model", default=DEFAULT_BRAIN_MODEL)
    auto_parser.add_argument("--controller-model", default=None)
    auto_parser.add_argument("--planner-model", default=None)
    auto_parser.add_argument("--proposal-model", default=None)
    auto_parser.add_argument("--review-model", default=None)
    auto_parser.add_argument("--max-tokens", type=int, default=DEFAULT_BRAIN_MAX_TOKENS)
    auto_parser.add_argument("--temperature", type=float, default=DEFAULT_BRAIN_TEMPERATURE)
    auto_parser.add_argument("--top-p", type=float, default=DEFAULT_BRAIN_TOP_P)
    auto_parser.add_argument("--llama-n-ctx", type=int, default=DEFAULT_LLAMA_N_CTX)
    auto_parser.add_argument("--llama-n-gpu-layers", type=int, default=DEFAULT_LLAMA_N_GPU_LAYERS)
    auto_parser.add_argument("--controller-llama-n-gpu-layers", type=int, default=None)
    auto_parser.add_argument("--controller-gpu", type=int, default=None)
    auto_parser.add_argument("--proposal-gpu", type=int, default=None)
    auto_parser.add_argument("--llama-type-k", type=int, default=DEFAULT_LLAMA_TYPE_K)
    auto_parser.add_argument("--llama-type-v", type=int, default=DEFAULT_LLAMA_TYPE_V)
    auto_parser.add_argument(
        "--keep-controller-loaded",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BRAIN_KEEP_LOADED,
    )
    auto_parser.add_argument(
        "--controller-warm-start",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BRAIN_WARM_START,
    )
    auto_parser.add_argument("--review-max-tokens", type=int, default=DEFAULT_REVIEW_MAX_TOKENS)
    auto_parser.add_argument("--review-first-interval", type=int, default=DEFAULT_REVIEW_FIRST_INTERVAL)
    auto_parser.add_argument("--review-repeat-interval", type=int, default=DEFAULT_REVIEW_REPEAT_INTERVAL)
    auto_parser.add_argument("--explore-trials-per-family", type=int, default=DEFAULT_EXPLORE_TRIALS_PER_FAMILY)
    auto_parser.add_argument("--exploit-top-families", type=int, default=DEFAULT_EXPLOIT_TOP_FAMILIES)
    auto_parser.add_argument("--max-draft-attempts", type=int, default=DEFAULT_MAX_DRAFT_ATTEMPTS)
    auto_parser.add_argument("--stop-after-keep", action="store_true")

    history_parser = subparsers.add_parser(
        "export-history",
        help="Build plug-and-play JSON/CSV/HTML history exports from runs/ and results.tsv.",
    )
    history_parser.add_argument("--output-dir", type=Path, default=HISTORY_DIR)

    args = parser.parse_args()

    try:
        if args.command == "setup":
            branch = setup_branch(tag=args.tag, base_ref=args.base_ref)
            print(f"branch: {branch}")
            print(f"results_tsv: {RESULTS_PATH}")
            return 0

        if args.command == "status":
            payload = status_payload()
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"branch: {payload['branch']}")
                print(f"commit: {payload['commit']}")
                print(f"results_count: {payload['results_count']}")
                print(f"dirty: {json.dumps(payload['dirty'])}")
                if payload["last_fast_keep"] is not None:
                    print(f"last_fast_keep: {json.dumps(payload['last_fast_keep'])}")
                if payload["last_promotion_keep"] is not None:
                    print(f"last_promotion_keep: {json.dumps(payload['last_promotion_keep'])}")
                if payload["last_milestone_keep"] is not None:
                    print(f"last_milestone_keep: {json.dumps(payload['last_milestone_keep'])}")
                if payload["fast_manifest"]:
                    print(f"fast_benchmark: {payload['fast_manifest'].get('dataset')}")
                    print(f"fast_queries: {payload['fast_manifest'].get('query_count')}")
                if payload["promotion_manifest"]:
                    print(f"promotion_benchmark: {payload['promotion_manifest'].get('dataset')}")
                if payload["milestone_manifest"]:
                    print(f"milestone_benchmark: {payload['milestone_manifest'].get('dataset')}")
                if payload["report_manifest"]:
                    print(f"report_benchmark: {payload['report_manifest'].get('dataset')}")
                print(f"history_dashboard: {payload['history_exports']['dashboard_html']}")
            return 0

        if args.command == "package-prompt":
            run_id = args.run_id or default_run_id(args.label)
            prompt_path = write_prompt_package(run_id=run_id, label=args.label)
            print(f"run_id: {run_id}")
            print(f"prompt: {prompt_path}")
            return 0

        if args.command == "run-once":
            run_id = args.run_id or default_run_id(args.label)
            promotion_artifact_dir = args.promotion_artifact_dir
            if promotion_artifact_dir is None and (DEFAULT_PROMOTION_ARTIFACT_DIR / "manifest.json").exists():
                promotion_artifact_dir = DEFAULT_PROMOTION_ARTIFACT_DIR
            payload = run_once(
                label=args.label,
                split=args.split,
                run_id=run_id,
                skip_git=args.skip_git,
                artifact_dir=args.artifact_dir,
                promotion_artifact_dir=promotion_artifact_dir,
                budget_seconds=args.budget_seconds,
                promotion_budget_seconds=args.promotion_budget_seconds,
                min_ndcg_gain=args.min_ndcg_gain,
                max_recall_rel_drop=args.max_recall_rel_drop,
                max_latency_rel_increase=args.max_latency_rel_increase,
                max_peak_memory_rel_increase=args.max_peak_memory_rel_increase,
                max_cost_usd_increase=args.max_cost_usd_increase,
                repeat_keep_runs=args.repeat_keep_runs,
            )
            print(f"run_id: {payload['run_id']}")
            print(f"status: {payload['overall']['status']}")
            print(f"reason: {payload['overall']['reason']}")
            print(f"skip_git: {payload['skip_git']}")
            print(f"run_dir: {RUNS_DIR / run_id}")
            print(f"history_dashboard: {payload['history_exports']['dashboard_html']}")
            return 0

        if args.command == "auto-loop":
            promotion_artifact_dir = args.promotion_artifact_dir
            if promotion_artifact_dir is None and (DEFAULT_PROMOTION_ARTIFACT_DIR / "manifest.json").exists():
                promotion_artifact_dir = DEFAULT_PROMOTION_ARTIFACT_DIR
            payload = run_autonomous_loop(
                iterations=args.iterations,
                label_prefix=args.label_prefix,
                split=args.split,
                skip_git=args.skip_git,
                artifact_dir=args.artifact_dir,
                promotion_artifact_dir=promotion_artifact_dir,
                budget_seconds=args.budget_seconds,
                promotion_budget_seconds=args.promotion_budget_seconds,
                min_ndcg_gain=args.min_ndcg_gain,
                max_recall_rel_drop=args.max_recall_rel_drop,
                max_latency_rel_increase=args.max_latency_rel_increase,
                max_peak_memory_rel_increase=args.max_peak_memory_rel_increase,
                max_cost_usd_increase=args.max_cost_usd_increase,
                repeat_keep_runs=args.repeat_keep_runs,
                brain_backend=args.brain_backend,
                brain_model=args.model,
                controller_model=args.controller_model,
                planner_model=args.planner_model,
                proposal_model=args.proposal_model,
                review_model=args.review_model,
                brain_max_tokens=args.max_tokens,
                brain_temperature=args.temperature,
                brain_top_p=args.top_p,
                brain_llama_n_ctx=args.llama_n_ctx,
                brain_llama_n_gpu_layers=args.llama_n_gpu_layers,
                controller_llama_n_gpu_layers=args.controller_llama_n_gpu_layers,
                brain_llama_type_k=args.llama_type_k,
                brain_llama_type_v=args.llama_type_v,
                controller_gpu=args.controller_gpu,
                proposal_gpu=args.proposal_gpu,
                keep_controller_loaded=args.keep_controller_loaded,
                controller_warm_start=args.controller_warm_start,
                review_max_tokens=args.review_max_tokens,
                review_first_interval=args.review_first_interval,
                review_repeat_interval=args.review_repeat_interval,
                explore_trials_per_family=args.explore_trials_per_family,
                exploit_top_families=args.exploit_top_families,
                max_draft_attempts=args.max_draft_attempts,
                stop_after_keep=args.stop_after_keep,
            )
            print(f"loop_id: {payload['loop_id']}")
            print(f"completed_iterations: {payload['completed_iterations']}")
            print(f"keeps: {payload['keeps']}")
            print(f"loop_dir: {RUNS_DIR / payload['loop_id']}")
            print(f"history_dashboard: {payload['history_exports']['dashboard_html']}")
            return 0

        if args.command == "export-history":
            exports = refresh_history_exports(output_dir=args.output_dir, runs_dir=RUNS_DIR, results_path=RESULTS_PATH)
            print(f"history_json: {exports['history_json']}")
            print(f"loops_csv: {exports['loops_csv']}")
            print(f"iterations_csv: {exports['iterations_csv']}")
            print(f"reviews_csv: {exports['reviews_csv']}")
            print(f"results_csv: {exports['results_csv']}")
            print(f"dashboard_html: {exports['dashboard_html']}")
            return 0

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
