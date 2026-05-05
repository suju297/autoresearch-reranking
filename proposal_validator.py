from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


FAMILY_SPECS: Dict[str, str] = {
    "reranker_model": "change the reranker checkpoint or safe model runtime settings",
    "candidate_k": "change how many candidates are reranked or how candidate depth is chosen",
    "score_normalization": "change how reranker and retrieval scores are normalized before comparison or fusion",
    "fusion_weight": "change how reranker scores and retrieval scores are blended or tie-broken",
    "metadata_boost": "apply a lightweight metadata-derived boost such as title or source signals when present",
    "dedup_filter": "remove duplicate or near-duplicate candidates before or after reranking",
    "truncation_policy": "change how candidate text is truncated, excerpted, or formatted before reranking",
    "pre_rerank_filter": "apply a cheap filter before cross-encoder scoring",
    "query_type_heuristic": "change query-conditional logic such as entity-like or long-query handling",
}
FAMILY_ORDER = tuple(FAMILY_SPECS.keys())
FAMILY_ALIASES = {
    "model": "reranker_model",
    "model_choice": "reranker_model",
    "model_name": "reranker_model",
    "reranker": "reranker_model",
    "reranker-choice": "reranker_model",
    "reranker_model_choice": "reranker_model",
    "candidate-depth": "candidate_k",
    "candidate-k": "candidate_k",
    "candidate-head": "candidate_k",
    "candiate_k": "candidate_k",
    "candidaite_k": "candidate_k",
    "score-fusion": "fusion_weight",
    "score-weighting": "fusion_weight",
    "score-blending": "fusion_weight",
    "fusion": "fusion_weight",
    "normalization": "score_normalization",
    "truncation-format": "truncation_policy",
    "truncation": "truncation_policy",
    "dedup": "dedup_filter",
    "candidate-filtering": "dedup_filter",
    "metadata": "metadata_boost",
    "query-heuristic": "query_type_heuristic",
}
FAMILY_KEYWORDS = {
    "reranker_model": ("model_name", "reranker model", "checkpoint", "bge-reranker", "qwen3-reranker"),
    "candidate_k": ("candidate_k", "top-k", "topk", "head size", "candidate depth", "rerank more"),
    "score_normalization": ("normalize", "normalization", "minmax", "zscore", "scale score"),
    "fusion_weight": ("fusion", "blend", "weight", "retrieval score", "tie-break", "alpha"),
    "metadata_boost": ("metadata", "title", "source_dataset", "field boost", "source boost"),
    "dedup_filter": ("dedup", "duplicate", "near-duplicate", "unique", "collapse duplicates"),
    "truncation_policy": ("truncate", "excerpt", "head-tail", "snippet", "chars", "format"),
    "pre_rerank_filter": ("filter", "prefilter", "pre-rerank", "cheap filter", "term match"),
    "query_type_heuristic": ("query type", "entity", "long query", "short query", "heuristic"),
}
CHANGED_KEY_ALIASES = {
    "model": "model_name",
    "reranker": "model_name",
    "reranker_model": "model_name",
    "model_choice": "model_name",
}
EXPECTED_DIRECTIONS = {"up", "flat", "down", "slightly_up", "slightly_down"}
PROMOTION_RISKS = {"low", "medium", "high"}
KNOWN_CHANGED_KEYS = {
    "model_name",
    "model_batch_size",
    "model_max_length",
    "candidate_k",
    "score_normalization",
    "fusion_weight",
    "metadata_boost",
    "dedup_filter",
    "truncation_policy",
    "pre_rerank_filter",
    "query_type_heuristic",
    "doc_max_chars",
}
COOLDOWN_DISCARD_THRESHOLD = 3
COOLDOWN_LOOKBACK_ATTEMPTS = 5
COOLDOWN_DURATION_RUNS = 5
NOVELTY_HISTORY = 20
RECENT_REJECTED_LABELS = 10


@dataclass
class ProposalIdea:
    family: str
    label: str
    hypothesis: str
    changed_keys: List[str]
    why_not_duplicate: str
    expected_ndcg_direction: str
    expected_recall_direction: str
    expected_latency_direction: str
    promotion_risk: str
    primary_mechanism: str
    why_recent_attempts_failed: str = ""

    def rationale_fingerprint(self) -> str:
        base = {
            "family": self.family,
            "changed_keys": sorted(self.changed_keys),
            "primary_mechanism": self.primary_mechanism.strip().lower(),
            "expected_ndcg_direction": self.expected_ndcg_direction,
            "expected_recall_direction": self.expected_recall_direction,
            "expected_latency_direction": self.expected_latency_direction,
        }
        payload = json.dumps(base, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def sanitize_slug(value: str, fallback: str = "trial") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", normalized) or fallback


def normalize_family(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    if normalized in FAMILY_SPECS:
        return normalized
    return FAMILY_ALIASES.get(normalized)


def infer_family(*parts: object) -> str | None:
    text = " ".join(str(part) for part in parts if part).lower()
    best_family = None
    best_score = 0
    for family, keywords in FAMILY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > best_score:
            best_family = family
            best_score = score
    return best_family


def normalize_changed_keys(values: Sequence[str]) -> List[str]:
    keys = []
    for value in values:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        normalized = "_".join(part for part in normalized.split("_") if part)
        if not normalized:
            continue
        normalized = CHANGED_KEY_ALIASES.get(normalized, normalized)
        if normalized not in KNOWN_CHANGED_KEYS:
            normalized = FAMILY_ALIASES.get(normalized, normalized)
        keys.append(normalized)
    deduped = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def family_catalog_markdown() -> str:
    return "\n".join(f"- `{family}`: {description}" for family, description in FAMILY_SPECS.items())


def parse_expected_direction(value: str, *, field_name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in EXPECTED_DIRECTIONS:
        raise ValueError(f"{field_name} must be one of: {', '.join(sorted(EXPECTED_DIRECTIONS))}")
    return normalized


def parse_promotion_risk(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in PROMOTION_RISKS:
        raise ValueError(f"promotion_risk must be one of: {', '.join(sorted(PROMOTION_RISKS))}")
    return normalized


def idea_from_mapping(payload: Mapping[str, Any]) -> ProposalIdea:
    family = normalize_family(str(payload.get("family", "")))
    if family is None:
        raise ValueError("idea family is missing or invalid")
    hypothesis = str(payload.get("hypothesis", "")).strip()
    if not hypothesis:
        raise ValueError("idea hypothesis is required")
    changed_keys = normalize_changed_keys(payload.get("changed_keys", []))
    if not changed_keys:
        raise ValueError("idea changed_keys must not be empty")
    for key in changed_keys:
        if key not in KNOWN_CHANGED_KEYS:
            raise ValueError(f"idea changed_keys includes unsupported key: {key}")
    label = sanitize_slug(str(payload.get("label", "")).strip() or f"{family}-{changed_keys[0]}", fallback=family)
    primary_mechanism = str(payload.get("primary_mechanism", "")).strip() or hypothesis
    why_not_duplicate = str(payload.get("why_not_duplicate", "")).strip()
    if not why_not_duplicate:
        raise ValueError("idea why_not_duplicate is required")
    why_recent_attempts_failed = str(payload.get("why_recent_attempts_failed", "")).strip()
    return ProposalIdea(
        family=family,
        label=label,
        hypothesis=hypothesis,
        changed_keys=changed_keys,
        why_not_duplicate=why_not_duplicate,
        expected_ndcg_direction=parse_expected_direction(
            str(payload.get("expected_ndcg_direction", "")),
            field_name="expected_ndcg_direction",
        ),
        expected_recall_direction=parse_expected_direction(
            str(payload.get("expected_recall_direction", "")),
            field_name="expected_recall_direction",
        ),
        expected_latency_direction=parse_expected_direction(
            str(payload.get("expected_latency_direction", "")),
            field_name="expected_latency_direction",
        ),
        promotion_risk=parse_promotion_risk(str(payload.get("promotion_risk", ""))),
        primary_mechanism=primary_mechanism,
        why_recent_attempts_failed=why_recent_attempts_failed,
    )


def loop_result_family(result: Mapping[str, Any]) -> str | None:
    explicit = normalize_family(str(result.get("family", "")))
    if explicit is not None:
        return explicit
    return infer_family(result.get("label", ""), result.get("summary", ""), result.get("reason", ""))


def loop_result_changed_keys(result: Mapping[str, Any]) -> List[str]:
    values = result.get("changed_keys")
    if isinstance(values, list):
        return normalize_changed_keys([str(value) for value in values])
    proposal_attempts = result.get("proposal_attempts", [])
    if isinstance(proposal_attempts, list):
        for attempt in proposal_attempts:
            if not isinstance(attempt, Mapping):
                continue
            keys = attempt.get("changed_keys")
            if isinstance(keys, list):
                return normalize_changed_keys([str(value) for value in keys])
    return []


def loop_result_fingerprint(result: Mapping[str, Any]) -> str | None:
    fingerprint = str(result.get("rationale_fingerprint", "")).strip()
    return fingerprint or None


def loop_result_expected_profile(result: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (
        str(result.get("expected_ndcg_direction", "")).strip() or None,
        str(result.get("expected_recall_direction", "")).strip() or None,
        str(result.get("expected_latency_direction", "")).strip() or None,
    )


def recent_rejected_labels(loop_results: Sequence[Mapping[str, Any]], limit: int = RECENT_REJECTED_LABELS) -> List[str]:
    labels: List[str] = []
    for row in reversed(list(loop_results)):
        overall = row.get("overall")
        if not isinstance(overall, Mapping):
            continue
        status = str(overall.get("status", ""))
        if status == "keep":
            break
        if status not in {"discard", "crash", "brain_error"}:
            continue
        label = str(row.get("label", "")).strip()
        if not label or label in labels:
            continue
        labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def family_counts(loop_results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in loop_results:
        family = loop_result_family(row)
        if family is None:
            continue
        counts[family] = counts.get(family, 0) + 1
    return counts


def family_failure_counts(loop_results: Sequence[Mapping[str, Any]], lookback: int = NOVELTY_HISTORY) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in list(loop_results)[-lookback:]:
        overall = row.get("overall")
        if not isinstance(overall, Mapping):
            continue
        if str(overall.get("status", "")) != "discard":
            continue
        family = loop_result_family(row)
        if family is None:
            continue
        counts[family] = counts.get(family, 0) + 1
    return counts


def family_success_counts(loop_results: Sequence[Mapping[str, Any]], lookback: int = NOVELTY_HISTORY) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in list(loop_results)[-lookback:]:
        overall = row.get("overall")
        if not isinstance(overall, Mapping):
            continue
        if str(overall.get("status", "")) != "keep":
            continue
        family = loop_result_family(row)
        if family is None:
            continue
        counts[family] = counts.get(family, 0) + 1
    return counts


def family_cooldowns(loop_results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    recent = list(loop_results)[-COOLDOWN_LOOKBACK_ATTEMPTS:]
    discard_counts: Dict[str, int] = {}
    for row in recent:
        overall = row.get("overall")
        if not isinstance(overall, Mapping):
            continue
        if str(overall.get("status", "")) != "discard":
            continue
        family = loop_result_family(row)
        if family is None:
            continue
        discard_counts[family] = discard_counts.get(family, 0) + 1

    cooldowns: Dict[str, int] = {}
    since_trigger: Dict[str, int] = {}
    for row in reversed(list(loop_results)):
        family = loop_result_family(row)
        if family is None:
            continue
        since_trigger[family] = since_trigger.get(family, 0) + 1
    for family, count in discard_counts.items():
        if count < COOLDOWN_DISCARD_THRESHOLD:
            continue
        runs_since = since_trigger.get(family, 0)
        if runs_since <= COOLDOWN_DURATION_RUNS:
            cooldowns[family] = COOLDOWN_DURATION_RUNS - runs_since + 1
    return cooldowns


def preferred_families(loop_results: Sequence[Mapping[str, Any]]) -> List[str]:
    counts = family_counts(loop_results)
    cooldowns = family_cooldowns(loop_results)
    recent_family = loop_result_family(loop_results[-1]) if loop_results else None
    underexplored = [family for family in FAMILY_ORDER if counts.get(family, 0) == 0 and family not in cooldowns]
    available = [family for family in FAMILY_ORDER if family not in cooldowns]
    if recent_family in available:
        available.remove(recent_family)
        available.append(recent_family)
    seen = set()
    ordered: List[str] = []
    for family in [*underexplored, *available]:
        if family in seen:
            continue
        ordered.append(family)
        seen.add(family)
    return ordered


def summarize_family_feedback(loop_results: Sequence[Mapping[str, Any]], *, limit: int = 5) -> str:
    if not loop_results:
        return "_No family feedback yet._"
    failures = family_failure_counts(loop_results)
    successes = family_success_counts(loop_results)
    cooldowns = family_cooldowns(loop_results)
    lines = []
    ordered = preferred_families(loop_results)
    for family in FAMILY_ORDER:
        fail_count = failures.get(family, 0)
        success_count = successes.get(family, 0)
        cooldown = cooldowns.get(family)
        if fail_count == 0 and success_count == 0 and cooldown is None and family not in ordered[:limit]:
            continue
        summary = f"- `{family}`: {success_count} keeps, {fail_count} discards"
        if cooldown is not None:
            summary += f", cooldown {cooldown} run(s) left"
        if family in ordered[:limit]:
            summary += ", preferred"
        lines.append(summary)
        if len(lines) >= limit:
            break
    return "\n".join(lines) if lines else "_No family feedback yet._"


def novelty_rejections(
    idea: ProposalIdea,
    loop_results: Sequence[Mapping[str, Any]],
) -> List[str]:
    reasons: List[str] = []
    if loop_results and idea.family == loop_result_family(loop_results[-1]):
        reasons.append("same family as the immediately previous run")

    cooldowns = family_cooldowns(loop_results)
    if idea.family in cooldowns:
        reasons.append(f"family `{idea.family}` is on cooldown for {cooldowns[idea.family]} more run(s)")

    if idea.label in recent_rejected_labels(loop_results):
        reasons.append(f"label `{idea.label}` matches a recently rejected label")

    for row in list(loop_results)[-NOVELTY_HISTORY:]:
        overall = row.get("overall")
        if not isinstance(overall, Mapping):
            continue
        if str(overall.get("status", "")) not in {"discard", "crash"}:
            continue
        if idea.family != loop_result_family(row):
            continue
        previous_keys = set(loop_result_changed_keys(row))
        if previous_keys and previous_keys == set(idea.changed_keys):
            reasons.append(
                f"same family and changed_keys as recent failed run `{row.get('label', 'unknown')}`"
            )
            break

    for row in list(loop_results)[-NOVELTY_HISTORY:]:
        overall = row.get("overall")
        if not isinstance(overall, Mapping):
            continue
        if str(overall.get("status", "")) not in {"discard", "crash"}:
            continue
        if idea.rationale_fingerprint() == loop_result_fingerprint(row):
            reasons.append(
                f"same rationale fingerprint as recent failed run `{row.get('label', 'unknown')}`"
            )
            break

    for row in list(loop_results)[-NOVELTY_HISTORY:]:
        overall = row.get("overall")
        if not isinstance(overall, Mapping):
            continue
        if str(overall.get("status", "")) not in {"discard", "crash"}:
            continue
        if idea.family != loop_result_family(row):
            continue
        previous_profile = loop_result_expected_profile(row)
        current_profile = (
            idea.expected_ndcg_direction,
            idea.expected_recall_direction,
            idea.expected_latency_direction,
        )
        if previous_profile == current_profile:
            reasons.append(
                f"same expected tradeoff profile as recent failed run `{row.get('label', 'unknown')}`"
            )
            break

    return reasons


def diversity_metrics(loop_results: Sequence[Mapping[str, Any]], *, window: int = 10) -> Dict[str, Any]:
    recent = list(loop_results)[-window:]
    families = [loop_result_family(row) for row in recent if loop_result_family(row) is not None]
    unique_families = len(set(families))
    repeat_rate = 0.0
    if families:
        repeat_rate = 1.0 - (unique_families / float(len(families)))
    keep_rate_by_family: Dict[str, Dict[str, int]] = {}
    promotion_pass_rate_by_family: Dict[str, Dict[str, int]] = {}
    duplicate_rejections = 0
    for row in recent:
        family = loop_result_family(row)
        if family is None:
            continue
        keep_stats = keep_rate_by_family.setdefault(family, {"keeps": 0, "attempts": 0})
        keep_stats["attempts"] += 1
        overall = row.get("overall")
        if isinstance(overall, Mapping) and str(overall.get("status", "")) == "keep":
            keep_stats["keeps"] += 1
        if "duplicate" in str(row.get("reason", "")).lower():
            duplicate_rejections += 1
        promotion_stats = promotion_pass_rate_by_family.setdefault(family, {"passes": 0, "attempts": 0})
        if "promotion_metrics" in row:
            promotion_stats["attempts"] += 1
            if isinstance(overall, Mapping) and str(overall.get("status", "")) == "keep":
                promotion_stats["passes"] += 1
    return {
        "window": window,
        "unique_families": unique_families,
        "repeat_rate": repeat_rate,
        "duplicate_rejection_rate": duplicate_rejections / float(len(recent) or 1),
        "keep_rate_by_family": keep_rate_by_family,
        "promotion_pass_rate_by_family": promotion_pass_rate_by_family,
    }
