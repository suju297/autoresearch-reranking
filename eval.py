from __future__ import annotations

import math
from typing import Dict, Iterable, List

import ir_measures
from ir_measures import R, RR, nDCG


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def build_qrels(qrels_by_query: Dict[str, Dict[str, int]]) -> List[ir_measures.Qrel]:
    qrels: List[ir_measures.Qrel] = []
    for query_id, query_qrels in qrels_by_query.items():
        for doc_id, relevance in query_qrels.items():
            qrels.append(ir_measures.Qrel(str(query_id), str(doc_id), int(relevance)))
    return qrels


def build_run(rankings: Dict[str, List[str]]) -> List[ir_measures.ScoredDoc]:
    run: List[ir_measures.ScoredDoc] = []
    for query_id, ranked_doc_ids in rankings.items():
        total = len(ranked_doc_ids)
        for rank, doc_id in enumerate(ranked_doc_ids):
            # ir_measures expects scored docs; a descending synthetic score preserves rank order.
            run.append(ir_measures.ScoredDoc(str(query_id), str(doc_id), float(total - rank)))
    return run


def evaluate_rankings(
    rankings: Dict[str, List[str]],
    qrels_by_query: Dict[str, Dict[str, int]],
    latencies_ms: List[float],
) -> Dict[str, float]:
    measures = [nDCG @ 10, RR @ 10, R @ 100]
    aggregates = ir_measures.calc_aggregate(measures, build_qrels(qrels_by_query), build_run(rankings))
    return {
        "ndcg@10": float(aggregates[nDCG @ 10]),
        "recall@100": float(aggregates[R @ 100]),
        "latency_p95_ms": percentile(latencies_ms, 0.95),
        "mrr@10": float(aggregates[RR @ 10]),
    }
