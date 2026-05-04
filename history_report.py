from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import quote


RUN_STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truncate(text: object, limit: int = 220) -> str:
    rendered = str(text or "").strip()
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 1].rstrip() + "…"


def _timestamp_from_run_id(run_id: str) -> str | None:
    match = RUN_STAMP_RE.search(run_id or "")
    if match is None:
        return None
    try:
        stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_timestamp_value(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = _timestamp_from_run_id(text)
    if parsed is not None:
        return parsed
    return text


def _timestamp_from_path(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_file_url(path: Path) -> str:
    return "file://" + quote(str(path))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_results_rows(results_path: Path) -> List[Dict[str, Any]]:
    if not results_path.exists():
        return []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                **row,
                "timestamp": _normalize_timestamp_value(row.get("timestamp")),
                "ndcg@10": _safe_float(row.get("ndcg@10")),
                "recall@100": _safe_float(row.get("recall@100")),
                "latency_p95_ms": _safe_float(row.get("latency_p95_ms")),
                "mrr@10": _safe_float(row.get("mrr@10")),
                "peak_memory_mb": _safe_float(row.get("peak_memory_mb")),
                "cost_usd": _safe_float(row.get("cost_usd")),
                "wall_clock_s": _safe_float(row.get("wall_clock_s")),
            }
        )
    return normalized


def _metric_value(summary: Mapping[str, Any], key: str) -> float | None:
    metrics = summary.get("metrics", {})
    if isinstance(metrics, Mapping):
        value = metrics.get(key)
        if value is not None:
            return _safe_float(value)
    return _safe_float(summary.get(key))


def _normalize_driver_summary(
    payload: Mapping[str, Any],
    path: Path,
    results_by_run_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    run_id = str(payload.get("run_id", path.parent.name))
    results_row = results_by_run_id.get(run_id, {})
    fast_summary = payload.get("fast_trial", {}).get("summary", {})
    promotion_summary = payload.get("promotion_trial", {}).get("summary", {}) if payload.get("promotion_trial") else {}
    timestamp = (
        _normalize_timestamp_value(results_row.get("timestamp", ""))
        or _timestamp_from_run_id(run_id)
        or _timestamp_from_path(path)
    )
    return {
        "run_id": run_id,
        "label": str(payload.get("label", "")),
        "split": str(payload.get("split", "")),
        "timestamp": timestamp,
        "status": str(payload.get("overall", {}).get("status", "")),
        "reason": str(payload.get("overall", {}).get("reason", "")),
        "strategy_commit": str(payload.get("overall", {}).get("strategy_commit", "")),
        "starting_commit": str(payload.get("git", {}).get("starting_commit", "")),
        "experiment_commit": str(payload.get("git", {}).get("experiment_commit", "")),
        "skip_git": bool(payload.get("skip_git", False)),
        "fast_ndcg@10": _metric_value(fast_summary, "ndcg@10"),
        "fast_recall@100": _metric_value(fast_summary, "recall@100"),
        "fast_latency_p95_ms": _metric_value(fast_summary, "latency_p95_ms"),
        "fast_mrr@10": _metric_value(fast_summary, "mrr@10"),
        "fast_peak_memory_mb": _metric_value(fast_summary, "peak_memory_mb"),
        "fast_wall_clock_s": _metric_value(fast_summary, "total_seconds"),
        "fast_benchmark_name": str(fast_summary.get("metrics", {}).get("benchmark_name", fast_summary.get("benchmark_name", ""))),
        "promotion_status": str(payload.get("promotion_trial", {}).get("summary", {}).get("decision", {}).get("status", "")) if payload.get("promotion_trial") else "",
        "promotion_reason": str(payload.get("promotion_trial", {}).get("summary", {}).get("decision", {}).get("reason", "")) if payload.get("promotion_trial") else "",
        "promotion_ndcg@10": _metric_value(promotion_summary, "ndcg@10"),
        "promotion_recall@100": _metric_value(promotion_summary, "recall@100"),
        "promotion_latency_p95_ms": _metric_value(promotion_summary, "latency_p95_ms"),
        "promotion_mrr@10": _metric_value(promotion_summary, "mrr@10"),
        "promotion_peak_memory_mb": _metric_value(promotion_summary, "peak_memory_mb"),
        "promotion_wall_clock_s": _metric_value(promotion_summary, "total_seconds"),
        "promotion_benchmark_name": str(promotion_summary.get("metrics", {}).get("benchmark_name", promotion_summary.get("benchmark_name", ""))),
        "source_path": str(path),
    }


def _normalize_iteration(
    loop_id: str,
    label_prefix: str,
    payload: Mapping[str, Any],
    driver_lookup: Mapping[str, Mapping[str, Any]],
    results_by_run_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    run_id = str(payload.get("run_id", ""))
    driver = driver_lookup.get(run_id, {})
    results_row = results_by_run_id.get(run_id, {})
    fast_metrics = payload.get("fast_metrics", {})
    promotion_metrics = payload.get("promotion_metrics", {})
    overall = payload.get("overall", {})
    timestamp = (
        _normalize_timestamp_value(driver.get("timestamp", ""))
        or _normalize_timestamp_value(results_row.get("timestamp", ""))
        or _timestamp_from_run_id(run_id)
    )
    changed_keys = payload.get("changed_keys", [])
    if not isinstance(changed_keys, list):
        changed_keys = []
    validation_errors = payload.get("validation_errors", [])
    if not isinstance(validation_errors, list):
        validation_errors = []
    return {
        "loop_id": loop_id,
        "label_prefix": label_prefix,
        "iteration": _safe_int(payload.get("iteration")) or 0,
        "run_id": run_id,
        "timestamp": timestamp,
        "family": str(payload.get("family", "")),
        "label": str(payload.get("label", "")),
        "summary": str(payload.get("summary", "")),
        "status": str(overall.get("status", payload.get("status", ""))),
        "reason": str(overall.get("reason", payload.get("reason", ""))),
        "strategy_commit": str(overall.get("strategy_commit", driver.get("strategy_commit", ""))),
        "changed_keys": changed_keys,
        "changed_keys_text": ",".join(str(item) for item in changed_keys),
        "proposal_attempt_count": len(payload.get("proposal_attempts", [])),
        "idea_attempt_count": len(payload.get("idea_attempts", [])),
        "validation_error_count": len(validation_errors),
        "validation_error_preview": _truncate("; ".join(str(item) for item in validation_errors), limit=180),
        "fast_ndcg@10": _safe_float(fast_metrics.get("ndcg@10")),
        "fast_recall@100": _safe_float(fast_metrics.get("recall@100")),
        "fast_latency_p95_ms": _safe_float(fast_metrics.get("latency_p95_ms")),
        "fast_mrr@10": _safe_float(fast_metrics.get("mrr@10")),
        "fast_peak_memory_mb": _safe_float(fast_metrics.get("peak_memory_mb")),
        "fast_wall_clock_s": _safe_float(fast_metrics.get("total_seconds")),
        "fast_benchmark_name": str(fast_metrics.get("benchmark_name", "")),
        "promotion_ndcg@10": _safe_float(promotion_metrics.get("ndcg@10")),
        "promotion_recall@100": _safe_float(promotion_metrics.get("recall@100")),
        "promotion_latency_p95_ms": _safe_float(promotion_metrics.get("latency_p95_ms")),
        "promotion_mrr@10": _safe_float(promotion_metrics.get("mrr@10")),
        "promotion_peak_memory_mb": _safe_float(promotion_metrics.get("peak_memory_mb")),
        "promotion_wall_clock_s": _safe_float(promotion_metrics.get("total_seconds")),
        "promotion_benchmark_name": str(promotion_metrics.get("benchmark_name", "")),
    }


def _normalize_review(loop_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    content = str(payload.get("content", ""))
    return {
        "loop_id": loop_id,
        "review_id": str(payload.get("review_id", "")),
        "iteration": _safe_int(payload.get("iteration")) or 0,
        "trigger": str(payload.get("trigger", "")),
        "fallback_used": bool(payload.get("fallback_used", False)),
        "content_preview": _truncate(content.replace("\n", " "), limit=260),
        "response_path": str(payload.get("response_path", "")),
    }


def _normalize_loop_summary(
    payload: Mapping[str, Any],
    path: Path,
    iterations: List[Mapping[str, Any]],
    reviews: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    status_counts = Counter(str(item.get("status", "")) for item in iterations if item.get("status"))
    family_counts = Counter(str(item.get("family", "")) for item in iterations if item.get("family"))
    timestamps = [_normalize_timestamp_value(item.get("timestamp", "")) for item in iterations if item.get("timestamp")]
    timestamps.sort()
    brain = payload.get("brain", {})
    diversity = payload.get("diversity", {})
    return {
        "loop_id": str(payload.get("loop_id", path.parent.name)),
        "label_prefix": str(payload.get("label_prefix", "")),
        "iterations_requested": _safe_int(payload.get("iterations_requested")) or len(iterations),
        "completed_iterations": _safe_int(payload.get("completed_iterations")) or len(iterations),
        "keeps": _safe_int(payload.get("keeps")) or sum(1 for item in iterations if item.get("status") == "keep"),
        "status_counts": dict(status_counts),
        "family_counts": dict(family_counts),
        "unique_family_count": len(family_counts),
        "review_count": len(reviews),
        "timestamp_start": timestamps[0] if timestamps else _timestamp_from_path(path),
        "timestamp_end": timestamps[-1] if timestamps else _timestamp_from_path(path),
        "brain_backend": str(brain.get("backend", "")),
        "brain_model": str(brain.get("model_name", "")),
        "repeat_rate": _safe_float(diversity.get("repeat_rate")),
        "duplicate_rejection_rate": _safe_float(diversity.get("duplicate_rejection_rate")),
        "source_path": str(path),
    }


def build_history_payload(*, runs_dir: Path, results_path: Path) -> Dict[str, Any]:
    results_rows = _load_results_rows(results_path)
    results_summary_rows = [row for row in results_rows if row.get("record_type", "summary") == "summary"]
    results_by_run_id = {
        str(row.get("run_id", "")): row
        for row in results_summary_rows
        if row.get("run_id")
    }

    driver_lookup: Dict[str, Dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("*/driver-summary.json")):
        payload = _load_json(path)
        normalized = _normalize_driver_summary(payload, path, results_by_run_id)
        driver_lookup[normalized["run_id"]] = normalized

    loops: List[Dict[str, Any]] = []
    iterations: List[Dict[str, Any]] = []
    reviews: List[Dict[str, Any]] = []
    loop_run_ids: set[str] = set()

    for path in sorted(runs_dir.glob("*/loop-summary.json")):
        payload = _load_json(path)
        loop_id = str(payload.get("loop_id", path.parent.name))
        label_prefix = str(payload.get("label_prefix", ""))
        loop_iterations = [
            _normalize_iteration(loop_id, label_prefix, item, driver_lookup, results_by_run_id)
            for item in payload.get("results", [])
            if isinstance(item, Mapping)
        ]
        loop_reviews = [
            _normalize_review(loop_id, item)
            for item in payload.get("reviews", [])
            if isinstance(item, Mapping)
        ]
        iterations.extend(loop_iterations)
        reviews.extend(loop_reviews)
        loop_run_ids.update(item["run_id"] for item in loop_iterations if item.get("run_id"))
        loops.append(_normalize_loop_summary(payload, path, loop_iterations, loop_reviews))

    standalone_runs = [
        driver
        for run_id, driver in sorted(driver_lookup.items(), key=lambda item: item[1].get("timestamp", ""))
        if run_id not in loop_run_ids
    ]

    iterations.sort(key=lambda item: (item.get("timestamp", ""), item.get("loop_id", ""), item.get("iteration", 0)))
    loops.sort(key=lambda item: (item.get("timestamp_start", ""), item.get("loop_id", "")))
    reviews.sort(key=lambda item: (item.get("loop_id", ""), item.get("iteration", 0)))
    results_summary_rows.sort(key=lambda item: str(item.get("timestamp", "")))

    total_keeps = sum(1 for item in iterations if item.get("status") == "keep")
    payload = {
        "generated_at": _utc_now(),
        "paths": {
            "runs_dir": str(runs_dir),
            "results_path": str(results_path),
        },
        "totals": {
            "results_summary_rows": len(results_summary_rows),
            "loops": len(loops),
            "iterations": len(iterations),
            "standalone_runs": len(standalone_runs),
            "reviews": len(reviews),
            "keeps": total_keeps,
            "keep_rate": round(total_keeps / len(iterations), 4) if iterations else 0.0,
        },
        "results_summary_rows": results_summary_rows,
        "loops": loops,
        "iterations": iterations,
        "reviews": reviews,
        "standalone_runs": standalone_runs,
    }
    return payload


def _write_csv(path: Path, rows: List[Mapping[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _value_range(values: Iterable[float]) -> tuple[float, float]:
    materialized = list(values)
    if not materialized:
        return (0.0, 1.0)
    low = min(materialized)
    high = max(materialized)
    if low == high:
        padding = low * 0.05 if low else 1.0
        return (low - padding, high + padding)
    return (low, high)


def _render_line_chart(title: str, rows: List[Mapping[str, Any]], key: str, color: str, fmt: str) -> str:
    points = [
        (str(row.get("run_id", "")), _safe_float(row.get(key)))
        for row in rows
        if _safe_float(row.get(key)) is not None
    ]
    if not points:
        return f"<section class='chart-card'><h3>{html.escape(title)}</h3><p class='empty'>No data yet.</p></section>"

    width = 820
    height = 240
    pad_x = 48
    pad_y = 28
    values = [value for _, value in points if value is not None]
    low, high = _value_range(value for value in values if value is not None)
    usable_width = width - pad_x * 2
    usable_height = height - pad_y * 2
    x_step = usable_width / max(len(points) - 1, 1)

    poly_points = []
    circles = []
    for index, (label, value) in enumerate(points):
        x = pad_x + x_step * index
        y = pad_y + usable_height * (1.0 - ((value - low) / (high - low)))
        poly_points.append(f"{x:.2f},{y:.2f}")
        circles.append(
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.5' fill='{color}'>"
            f"<title>{html.escape(label)}: {format(value, fmt)}</title></circle>"
        )

    grid_lines = []
    for fraction in (0.0, 0.5, 1.0):
        y = pad_y + usable_height * fraction
        grid_lines.append(f"<line x1='{pad_x}' y1='{y:.2f}' x2='{width - pad_x}' y2='{y:.2f}' class='grid' />")
    y_labels = [
        (pad_y, high),
        (pad_y + usable_height / 2.0, (low + high) / 2.0),
        (pad_y + usable_height, low),
    ]
    label_markup = "".join(
        f"<text x='8' y='{y + 4:.2f}' class='axis'>{format(value, fmt)}</text>"
        for y, value in y_labels
    )
    return (
        f"<section class='chart-card'><h3>{html.escape(title)}</h3>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        + "".join(grid_lines)
        + f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{' '.join(poly_points)}' />"
        + "".join(circles)
        + label_markup
        + "</svg></section>"
    )


def _render_bar_chart(title: str, values: Mapping[str, int], color: str) -> str:
    items = [(label, count) for label, count in values.items() if count > 0]
    if not items:
        return f"<section class='chart-card'><h3>{html.escape(title)}</h3><p class='empty'>No data yet.</p></section>"
    width = 820
    height = 240
    pad_x = 48
    pad_y = 28
    usable_width = width - pad_x * 2
    usable_height = height - pad_y * 2
    max_value = max(count for _, count in items) or 1
    bar_width = usable_width / max(len(items), 1) * 0.7
    gap = usable_width / max(len(items), 1)
    bars = []
    labels = []
    for index, (label, count) in enumerate(items):
        x = pad_x + gap * index + (gap - bar_width) / 2.0
        bar_height = usable_height * (count / max_value)
        y = pad_y + usable_height - bar_height
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width:.2f}' height='{bar_height:.2f}' fill='{color}'>"
            f"<title>{html.escape(label)}: {count}</title></rect>"
        )
        labels.append(
            f"<text x='{x + bar_width / 2.0:.2f}' y='{height - 6}' text-anchor='middle' class='axis'>{html.escape(label)}</text>"
        )
    return (
        f"<section class='chart-card'><h3>{html.escape(title)}</h3>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        + f"<line x1='{pad_x}' y1='{height - pad_y}' x2='{width - pad_x}' y2='{height - pad_y}' class='grid' />"
        + "".join(bars)
        + "".join(labels)
        + "</svg></section>"
    )


def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    thead = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return "<table><thead><tr>" + thead + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def render_history_dashboard(payload: Mapping[str, Any], output_dir: Path) -> str:
    iterations = list(payload.get("iterations", []))
    loops = list(payload.get("loops", []))
    results_rows = list(payload.get("results_summary_rows", []))
    reviews = list(payload.get("reviews", []))
    totals = payload.get("totals", {})

    latest_iterations = list(reversed(iterations[-12:]))
    latest_loops = list(reversed(loops[-8:]))
    latest_results = list(reversed(results_rows[-12:]))

    status_counts = Counter(str(item.get("status", "")) for item in iterations if item.get("status"))
    family_counts = Counter(str(item.get("family", "")) for item in iterations if item.get("family"))

    iteration_rows = []
    for item in latest_iterations:
        runs_dir = Path(str(payload.get("paths", {}).get("runs_dir", output_dir.parent / "runs")))
        driver_path = runs_dir / item["run_id"] / "driver-summary.json"
        details_link = (
            f"<a href='{html.escape(_json_file_url(driver_path))}'>driver-summary</a>"
            if driver_path.exists()
            else "-"
        )
        iteration_rows.append(
            [
                html.escape(str(item.get("timestamp", ""))),
                html.escape(str(item.get("loop_id", ""))),
                html.escape(str(item.get("iteration", ""))),
                html.escape(str(item.get("family", ""))),
                html.escape(str(item.get("label", ""))),
                html.escape(str(item.get("status", ""))),
                "" if item.get("fast_ndcg@10") is None else f"{item['fast_ndcg@10']:.6f}",
                "" if item.get("fast_latency_p95_ms") is None else f"{item['fast_latency_p95_ms']:.3f}",
                _truncate(item.get("reason", ""), limit=90),
                details_link,
            ]
        )

    loop_rows = []
    for item in latest_loops:
        source_path = Path(str(item.get("source_path", "")))
        details_link = (
            f"<a href='{html.escape(_json_file_url(source_path))}'>loop-summary</a>"
            if source_path.exists()
            else "-"
        )
        loop_rows.append(
            [
                html.escape(str(item.get("timestamp_start", ""))),
                html.escape(str(item.get("loop_id", ""))),
                html.escape(str(item.get("label_prefix", ""))),
                html.escape(str(item.get("completed_iterations", ""))),
                html.escape(str(item.get("keeps", ""))),
                html.escape(str(item.get("unique_family_count", ""))),
                "" if item.get("repeat_rate") is None else f"{item['repeat_rate']:.3f}",
                details_link,
            ]
        )

    result_rows = []
    for item in latest_results:
        result_rows.append(
            [
                html.escape(str(item.get("timestamp", ""))),
                html.escape(str(item.get("benchmark_name", ""))),
                html.escape(str(item.get("split", ""))),
                html.escape(str(item.get("status", ""))),
                "" if item.get("ndcg@10") is None else f"{item['ndcg@10']:.6f}",
                "" if item.get("latency_p95_ms") is None else f"{item['latency_p95_ms']:.3f}",
                _truncate(item.get("decision_reason", ""), limit=70),
                _truncate(item.get("description", ""), limit=70),
            ]
        )

    latest_review = reviews[-1] if reviews else {}
    latest_review_markup = (
        f"<p><strong>{html.escape(str(latest_review.get('review_id', '')))}</strong> at iteration "
        f"{html.escape(str(latest_review.get('iteration', '')))}: "
        f"{html.escape(str(latest_review.get('content_preview', '')))}</p>"
        if latest_review
        else "<p class='empty'>No review artifacts yet.</p>"
    )

    cards = [
        ("Loops", str(totals.get("loops", 0))),
        ("Iterations", str(totals.get("iterations", 0))),
        ("Keeps", str(totals.get("keeps", 0))),
        ("Keep Rate", f"{float(totals.get('keep_rate', 0.0)):.1%}"),
    ]

    card_markup = "".join(
        f"<div class='card'><div class='card-title'>{html.escape(title)}</div><div class='card-value'>{html.escape(value)}</div></div>"
        for title, value in cards
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Autoresearch History Dashboard</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 24px;
      background: #f5f7fb;
      color: #172033;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ margin: 0 0 10px; line-height: 1.45; }}
    .meta {{ color: #5e6b85; margin-bottom: 20px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .card, .panel, .chart-card {{
      background: white;
      border-radius: 14px;
      box-shadow: 0 10px 30px rgba(19, 31, 55, 0.08);
      padding: 16px;
    }}
    .card-title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #66758f;
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 30px;
      font-weight: 700;
    }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .grid {{
      stroke: #d7deeb;
      stroke-width: 1;
      stroke-dasharray: 4 4;
    }}
    .axis {{
      fill: #66758f;
      font-size: 11px;
    }}
    .panel-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e5ebf5;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      color: #66758f;
      letter-spacing: 0.06em;
    }}
    .empty {{ color: #66758f; }}
    a {{ color: #1f5eff; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>Autoresearch History Dashboard</h1>
  <p class="meta">Generated {html.escape(str(payload.get("generated_at", "")))} from {html.escape(str(payload.get("paths", {}).get("runs_dir", "")))} and {html.escape(str(payload.get("paths", {}).get("results_path", "")))}.</p>
  <section class="cards">{card_markup}</section>
  <section class="charts">
    {_render_line_chart("Fast ndcg@10 by Iteration", iterations, "fast_ndcg@10", "#2457ff", ".6f")}
    {_render_line_chart("Fast latency p95 (ms) by Iteration", iterations, "fast_latency_p95_ms", "#ff6b2c", ".3f")}
    {_render_bar_chart("Iteration Status Counts", status_counts, "#2b7fff")}
    {_render_bar_chart("Family Usage Counts", family_counts, "#18a57d")}
  </section>
  <section class="panel-grid">
    <div class="panel">
      <h2>Recent Loops</h2>
      {_render_table(["Started", "Loop", "Label Prefix", "Iterations", "Keeps", "Families", "Repeat Rate", "Raw"], loop_rows)}
    </div>
    <div class="panel">
      <h2>Latest Strategic Review</h2>
      {latest_review_markup}
    </div>
  </section>
  <section class="panel" style="margin-bottom: 20px;">
    <h2>Recent Iterations</h2>
    {_render_table(["Timestamp", "Loop", "#", "Family", "Label", "Status", "Fast ndcg@10", "Fast latency", "Reason", "Raw"], iteration_rows)}
  </section>
  <section class="panel">
    <h2>Recent Benchmark Summary Rows</h2>
    {_render_table(["Timestamp", "Benchmark", "Split", "Status", "ndcg@10", "Latency p95", "Reason", "Description"], result_rows)}
  </section>
</body>
</html>
"""
    return html_doc


def refresh_history_exports(*, output_dir: Path, runs_dir: Path, results_path: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_history_payload(runs_dir=runs_dir, results_path=results_path)

    history_json = output_dir / "history.json"
    loops_csv = output_dir / "loops.csv"
    iterations_csv = output_dir / "iterations.csv"
    reviews_csv = output_dir / "reviews.csv"
    results_csv = output_dir / "results-summary.csv"
    dashboard_html = output_dir / "index.html"

    history_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(
        loops_csv,
        payload["loops"],
        [
            "loop_id",
            "label_prefix",
            "timestamp_start",
            "timestamp_end",
            "iterations_requested",
            "completed_iterations",
            "keeps",
            "unique_family_count",
            "review_count",
            "repeat_rate",
            "duplicate_rejection_rate",
            "brain_backend",
            "brain_model",
            "source_path",
        ],
    )
    _write_csv(
        iterations_csv,
        payload["iterations"],
        [
            "timestamp",
            "loop_id",
            "label_prefix",
            "iteration",
            "run_id",
            "family",
            "label",
            "status",
            "reason",
            "strategy_commit",
            "changed_keys_text",
            "proposal_attempt_count",
            "idea_attempt_count",
            "validation_error_count",
            "validation_error_preview",
            "fast_benchmark_name",
            "fast_ndcg@10",
            "fast_recall@100",
            "fast_latency_p95_ms",
            "fast_mrr@10",
            "fast_peak_memory_mb",
            "fast_wall_clock_s",
            "promotion_benchmark_name",
            "promotion_ndcg@10",
            "promotion_recall@100",
            "promotion_latency_p95_ms",
            "promotion_mrr@10",
            "promotion_peak_memory_mb",
            "promotion_wall_clock_s",
        ],
    )
    _write_csv(
        reviews_csv,
        payload["reviews"],
        [
            "loop_id",
            "review_id",
            "iteration",
            "trigger",
            "fallback_used",
            "content_preview",
            "response_path",
        ],
    )
    _write_csv(
        results_csv,
        payload["results_summary_rows"],
        [
            "timestamp",
            "record_type",
            "run_id",
            "commit",
            "benchmark_name",
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
        ],
    )
    dashboard_html.write_text(render_history_dashboard(payload, output_dir), encoding="utf-8")
    return {
        "history_json": str(history_json),
        "loops_csv": str(loops_csv),
        "iterations_csv": str(iterations_csv),
        "reviews_csv": str(reviews_csv),
        "results_csv": str(results_csv),
        "dashboard_html": str(dashboard_html),
    }
