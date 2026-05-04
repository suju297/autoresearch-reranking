from __future__ import annotations

import os
from typing import Dict, List, Sequence

import torch
from sentence_transformers import CrossEncoder


# This is the only file the agent should edit.
MODEL_NAME = "BAAI/bge-reranker-base"
MODEL_BATCH_SIZE = int(os.environ.get("AUTORESEARCH_RERANK_MODEL_BATCH_SIZE", "4"))
MODEL_MAX_LENGTH = int(os.environ.get("AUTORESEARCH_RERANK_MODEL_MAX_LENGTH", "512"))
DEVICE = os.environ.get("AUTORESEARCH_RERANK_DEVICE", "")

STRATEGY: Dict[str, object] = {
    "family": "baseline",
    "candidate_k": 10,
    "score_normalization": "none",
    "fusion_weight": 1.0,
    "metadata_boost": {
        "enabled": False,
        "title_match_weight": 0.05,
        "source_match_weight": 0.02,
    },
    "dedup_filter": {
        "enabled": False,
        "mode": "doc_id",
    },
    "truncation_policy": {
        "mode": "head",
        "max_chars": 2000,
    },
    "pre_rerank_filter": {
        "enabled": False,
        "min_query_term_matches": 1,
    },
    "query_type_heuristic": {
        "enabled": False,
        "entity_like_max_terms": 3,
        "entity_like_candidate_k": 12,
        "long_query_min_terms": 8,
        "long_query_candidate_k": 8,
    },
}

_RERANKER: CrossEncoder | None = None
_RERANKER_NAME: str | None = None


def resolve_device() -> str:
    if DEVICE:
        return DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def strategy_config() -> Dict[str, object]:
    return {
        "model_name": MODEL_NAME,
        "model_batch_size": MODEL_BATCH_SIZE,
        "model_max_length": MODEL_MAX_LENGTH,
        "device": resolve_device(),
        **STRATEGY,
    }


def get_reranker() -> CrossEncoder:
    global _RERANKER, _RERANKER_NAME
    model_name = MODEL_NAME
    if _RERANKER is None or _RERANKER_NAME != model_name:
        _RERANKER = CrossEncoder(
            model_name,
            max_length=MODEL_MAX_LENGTH,
            device=resolve_device(),
        )
        _RERANKER_NAME = model_name
    return _RERANKER


def tokenize(text: str) -> List[str]:
    return [token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if token]


def query_term_matches(query_terms: Sequence[str], text: str) -> int:
    lowered = text.lower()
    return sum(1 for term in query_terms if term in lowered)


def select_candidate_k(query_terms: Sequence[str], candidate_count: int) -> int:
    base_k = int(STRATEGY.get("candidate_k", 10))
    heuristic = STRATEGY.get("query_type_heuristic", {})
    if isinstance(heuristic, dict) and heuristic.get("enabled"):
        if len(query_terms) <= int(heuristic.get("entity_like_max_terms", 3)):
            base_k = max(base_k, int(heuristic.get("entity_like_candidate_k", base_k)))
        if len(query_terms) >= int(heuristic.get("long_query_min_terms", 8)):
            base_k = min(base_k, int(heuristic.get("long_query_candidate_k", base_k)))
    return max(1, min(base_k, candidate_count))


def dedup_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    cfg = STRATEGY.get("dedup_filter", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return candidates
    seen_doc_ids = set()
    seen_text_keys = set()
    deduped = []
    for candidate in candidates:
        doc_id = str(candidate.get("doc_id", ""))
        text_key = str(candidate.get("text", ""))[:256].strip().lower()
        if doc_id and doc_id in seen_doc_ids:
            continue
        if cfg.get("mode") == "doc_id_text" and text_key and text_key in seen_text_keys:
            continue
        if doc_id:
            seen_doc_ids.add(doc_id)
        if text_key:
            seen_text_keys.add(text_key)
        deduped.append(candidate)
    return deduped


def apply_pre_rerank_filter(query_terms: Sequence[str], candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    cfg = STRATEGY.get("pre_rerank_filter", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return candidates
    minimum = max(1, int(cfg.get("min_query_term_matches", 1)))
    filtered = [
        candidate
        for candidate in candidates
        if query_term_matches(query_terms, str(candidate.get("text", ""))) >= minimum
    ]
    return filtered or candidates


def truncate_text(text: str) -> str:
    policy = STRATEGY.get("truncation_policy", {})
    if not isinstance(policy, dict):
        return text
    max_chars = int(policy.get("max_chars", 2000))
    if len(text) <= max_chars:
        return text
    mode = str(policy.get("mode", "head"))
    if mode == "head_tail" and max_chars >= 8:
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars - 1
        return text[:head_chars] + "\n" + text[-tail_chars:]
    return text[:max_chars]


def score_normalize(values: Sequence[float]) -> List[float]:
    mode = str(STRATEGY.get("score_normalization", "none"))
    values = [float(value) for value in values]
    if not values or mode == "none":
        return values
    if mode == "minmax":
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            return [0.0 for _ in values]
        return [(value - lo) / (hi - lo) for value in values]
    if mode == "softsign":
        return [value / (1.0 + abs(value)) for value in values]
    return values


def metadata_boost(query_terms: Sequence[str], candidate: Dict[str, object]) -> float:
    cfg = STRATEGY.get("metadata_boost", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return 0.0
    boost = 0.0
    title = str(candidate.get("title", ""))
    if title and any(term in title.lower() for term in query_terms):
        boost += float(cfg.get("title_match_weight", 0.05))
    source_dataset = str(candidate.get("source_dataset", ""))
    if source_dataset and any(term in source_dataset.lower() for term in query_terms):
        boost += float(cfg.get("source_match_weight", 0.02))
    return boost


def select_candidates(query_terms: Sequence[str], candidates: List[Dict[str, object]]) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    prepared = apply_pre_rerank_filter(query_terms, dedup_candidates(candidates))
    candidate_k = select_candidate_k(query_terms, len(prepared))
    return prepared[:candidate_k], prepared[candidate_k:]


def rerank_with_model(query: str, head: List[Dict[str, object]]) -> List[float]:
    reranker = get_reranker()
    pairs = [(query, truncate_text(str(candidate.get("text", "")))) for candidate in head]
    scores = reranker.predict(
        pairs,
        batch_size=MODEL_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [float(score) for score in scores]


def use_baseline_sort() -> bool:
    metadata_cfg = STRATEGY.get("metadata_boost", {})
    dedup_cfg = STRATEGY.get("dedup_filter", {})
    prefilter_cfg = STRATEGY.get("pre_rerank_filter", {})
    heuristic_cfg = STRATEGY.get("query_type_heuristic", {})
    return (
        float(STRATEGY.get("fusion_weight", 1.0)) >= 0.999
        and str(STRATEGY.get("score_normalization", "none")) == "none"
        and (not isinstance(metadata_cfg, dict) or not metadata_cfg.get("enabled"))
        and (not isinstance(dedup_cfg, dict) or not dedup_cfg.get("enabled"))
        and (not isinstance(prefilter_cfg, dict) or not prefilter_cfg.get("enabled"))
        and (not isinstance(heuristic_cfg, dict) or not heuristic_cfg.get("enabled"))
    )


def apply_final_scores(
    query_terms: Sequence[str],
    head: List[Dict[str, object]],
    rerank_scores: Sequence[float],
) -> List[Dict[str, object]]:
    retrieval_scores = [float(candidate.get("retrieval_score", 0.0)) for candidate in head]
    normalized_rerank = score_normalize(rerank_scores)
    normalized_retrieval = score_normalize(retrieval_scores)
    fusion_weight = float(STRATEGY.get("fusion_weight", 1.0))

    ranked_head = []
    for candidate, rerank_score, norm_rerank, retrieval_score, norm_retrieval in zip(
        head,
        rerank_scores,
        normalized_rerank,
        retrieval_scores,
        normalized_retrieval,
    ):
        updated = dict(candidate)
        updated["rerank_score"] = float(rerank_score)
        if use_baseline_sort():
            updated["final_score"] = float(rerank_score)
        else:
            blended = (fusion_weight * float(norm_rerank)) + ((1.0 - fusion_weight) * float(norm_retrieval))
            blended += metadata_boost(query_terms, candidate)
            updated["final_score"] = blended
        updated["retrieval_score"] = retrieval_score
        ranked_head.append(updated)

    if use_baseline_sort():
        ranked_head.sort(
            key=lambda item: (
                float(item["rerank_score"]),
                float(item.get("retrieval_score", 0.0)),
            ),
            reverse=True,
        )
    else:
        ranked_head.sort(
            key=lambda item: (
                float(item.get("final_score", 0.0)),
                float(item.get("rerank_score", 0.0)),
                float(item.get("retrieval_score", 0.0)),
            ),
            reverse=True,
        )
    return ranked_head


def rerank(query: str, candidates: List[Dict[str, object]], ctx: Dict[str, object]) -> List[Dict[str, object]]:
    query_terms = tokenize(query)
    head, tail = select_candidates(query_terms, candidates)
    rerank_scores = rerank_with_model(query, head)
    ranked_head = apply_final_scores(query_terms, head, rerank_scores)
    return ranked_head + tail
