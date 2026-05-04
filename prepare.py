from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from heapq import nlargest
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts" / "current"
TIME_BUDGET = 600.0
DEFAULT_DATASET_ID = "fast-fiqa-dev"
DEFAULT_TEST_FRACTION = 0.2
CANDIDATE_GENERATION_VERSION = "bm25-v1"
HARNESS_VERSION = "sandboxed-worker-ir-measures-v4-bge-base-fiqa"
FIXED_RERANKER_MODEL = "BAAI/bge-reranker-base"
BM25_K1 = 1.2
BM25_B = 0.75
DATASET_PRESETS = {
    "fast-fiqa-dev": [
        "beir/fiqa/dev",
    ],
    "promotion-scifact": [
        "beir/scifact/test",
    ],
    "milestone-fiqa-dev": [
        "beir/fiqa/dev",
    ],
    "report-fiqa-test": [
        "beir/fiqa/test",
    ],
    "diverse-medium": [
        "beir/scifact/test",
        "beir/nfcorpus/test",
        "nano-beir/webis-touche2020",
        "nano-beir/dbpedia-entity",
    ],
    "diverse-nano": [
        "nano-beir/fiqa",
        "nano-beir/scifact",
        "nano-beir/hotpotqa",
        "nano-beir/webis-touche2020",
        "nano-beir/dbpedia-entity",
    ],
}
PRESET_TEST_FRACTIONS = {
    "fast-fiqa-dev": 0.0,
    "promotion-scifact": 0.0,
    "milestone-fiqa-dev": 0.0,
    "report-fiqa-test": 0.0,
}


def tokenize(text: str) -> List[str]:
    return [token.strip(".,!?;:()[]{}\"'").lower() for token in text.split() if token.strip()]


def build_bm25_index(docs: List[Dict[str, str]]) -> Dict[str, object]:
    postings: Dict[str, List[tuple[int, int]]] = defaultdict(list)
    doc_lengths: List[int] = []
    avg_doc_length = 0.0

    for doc_index, doc in enumerate(docs):
        term_counts = Counter(tokenize(doc["text"]))
        doc_length = sum(term_counts.values())
        doc_lengths.append(doc_length)
        for term, count in term_counts.items():
            postings[term].append((doc_index, count))

    if doc_lengths:
        avg_doc_length = float(sum(doc_lengths)) / float(len(doc_lengths))

    idf: Dict[str, float] = {}
    num_docs = len(docs)
    for term, posting_list in postings.items():
        doc_freq = len(posting_list)
        idf[term] = math.log(1.0 + ((num_docs - doc_freq + 0.5) / (doc_freq + 0.5)))

    return {
        "postings": postings,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "idf": idf,
        "doc_ids": [doc["doc_id"] for doc in docs],
    }


def build_query_candidates(
    query_text: str,
    retrieval_index: Dict[str, object],
    top_k: int,
) -> List[Dict[str, float]]:
    postings: Dict[str, List[tuple[int, int]]] = retrieval_index["postings"]  # type: ignore[assignment]
    doc_lengths: List[int] = retrieval_index["doc_lengths"]  # type: ignore[assignment]
    avg_doc_length = float(retrieval_index["avg_doc_length"])  # type: ignore[arg-type]
    idf: Dict[str, float] = retrieval_index["idf"]  # type: ignore[assignment]
    doc_ids: List[str] = retrieval_index["doc_ids"]  # type: ignore[assignment]
    query_terms = Counter(tokenize(query_text))
    scores: Dict[int, float] = {}

    for term in query_terms:
        term_idf = idf.get(term)
        if term_idf is None:
            continue
        for doc_index, term_frequency in postings.get(term, []):
            doc_length = doc_lengths[doc_index]
            length_norm = 1.0 - BM25_B + BM25_B * (
                (float(doc_length) / avg_doc_length) if avg_doc_length > 0.0 else 0.0
            )
            numerator = term_frequency * (BM25_K1 + 1.0)
            denominator = term_frequency + BM25_K1 * length_norm
            if denominator <= 0.0:
                continue
            scores[doc_index] = scores.get(doc_index, 0.0) + term_idf * (numerator / denominator)

    ranked_doc_indices = [
        doc_index
        for doc_index, _ in nlargest(
            top_k,
            scores.items(),
            key=lambda item: (item[1], -item[0]),
        )
    ]

    if len(ranked_doc_indices) < top_k:
        selected = set(ranked_doc_indices)
        for doc_index in range(len(doc_ids)):
            if doc_index in selected:
                continue
            ranked_doc_indices.append(doc_index)
            if len(ranked_doc_indices) == top_k:
                break

    return [
        {
            "doc_id": doc_ids[doc_index],
            "retrieval_score": float(scores.get(doc_index, 0.0)),
        }
        for doc_index in ranked_doc_indices
    ]


def build_candidates(queries: List[Dict[str, str]], docs: List[Dict[str, str]], top_k: int) -> Dict[str, List[Dict[str, float]]]:
    retrieval_index = build_bm25_index(docs)
    candidates: Dict[str, List[Dict[str, float]]] = {}
    for query in queries:
        candidates[query["query_id"]] = build_query_candidates(
            query_text=query["text"],
            retrieval_index=retrieval_index,
            top_k=top_k,
        )
    return candidates


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_path_for(artifact_dir: Path = ARTIFACT_DIR) -> Path:
    return artifact_dir / "manifest.json"


def load_manifest(artifact_dir: Path = ARTIFACT_DIR) -> Dict[str, object]:
    path = manifest_path_for(artifact_dir=artifact_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_hash(artifact_dir: Path = ARTIFACT_DIR) -> str:
    path = manifest_path_for(artifact_dir=artifact_dir)
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def benchmark_metadata(artifact_dir: Path = ARTIFACT_DIR, split: str = "dev") -> Dict[str, object]:
    manifest = load_manifest(artifact_dir=artifact_dir)
    split_manifest_path = artifact_dir / split / "manifest.json"
    split_manifest = load_json(split_manifest_path) if split_manifest_path.exists() else {}
    return {
        "benchmark_name": str(manifest.get("dataset", "unknown")),
        "benchmark_manifest_hash": manifest_hash(artifact_dir=artifact_dir),
        "candidate_generation_version": str(
            manifest.get("candidate_generation_version", CANDIDATE_GENERATION_VERSION)
        ),
        "harness_version": HARNESS_VERSION,
        "split": split,
        "candidate_top_k": int(split_manifest.get("candidate_top_k", manifest.get("candidate_top_k", 0))),
    }


def split_dir_for(artifact_dir: Path, split: str) -> Path:
    split_dir = artifact_dir / split
    if split_dir.exists():
        return split_dir
    return artifact_dir


def load_split_artifacts(artifact_dir: Path = ARTIFACT_DIR, split: str = "dev"):
    docs = load_json(artifact_dir / "docs.json")
    split_dir = split_dir_for(artifact_dir, split)
    queries = load_json(split_dir / "queries.json")
    candidates = load_json(split_dir / "candidates.json")
    return docs, queries, candidates


def load_qrels_by_query(artifact_dir: Path = ARTIFACT_DIR, split: str = "dev"):
    split_dir = split_dir_for(artifact_dir, split)
    qrels_rows = load_json(split_dir / "qrels.json")
    qrels_by_query: Dict[str, Dict[str, int]] = {}
    for row in qrels_rows:
        qrels_by_query.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    return qrels_by_query


def import_ir_datasets():
    try:
        import ir_datasets
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency `ir_datasets`. Install project dependencies first, for example with "
            "`python3 -m pip install -e .`."
        ) from exc
    return ir_datasets


def ensure_reranker_cached(model_name: str = FIXED_RERANKER_MODEL) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency support for Hugging Face model downloads. Install project dependencies first."
        ) from exc
    snapshot_download(repo_id=model_name)


def join_text_parts(parts: Iterable[object]) -> str:
    cleaned = [str(part).strip() for part in parts if part is not None and str(part).strip()]
    return "\n\n".join(cleaned)


def doc_to_record(doc: object) -> Dict[str, str]:
    values = doc._asdict()
    text_parts = [value for key, value in values.items() if key != "doc_id"]
    return {
        "doc_id": str(values["doc_id"]),
        "text": join_text_parts(text_parts),
    }


def query_to_record(query: object) -> Dict[str, str]:
    values = query._asdict()
    return {
        "query_id": str(values["query_id"]),
        "text": str(values["text"]),
    }


def qrel_to_record(qrel: object) -> Dict[str, object]:
    values = qrel._asdict()
    return {
        "query_id": str(values["query_id"]),
        "doc_id": str(values["doc_id"]),
        "relevance": int(values["relevance"]),
    }


def build_toy_dataset():
    docs = [
        {
            "doc_id": "d1",
            "text": "Python virtual environments Create isolated Python environments with venv and activate them before installing packages.",
        },
        {
            "doc_id": "d2",
            "text": "Git branching basics Use git branches to isolate work and merge reviewed changes into main.",
        },
        {
            "doc_id": "d3",
            "text": "FastAPI introduction FastAPI is a Python web framework for building APIs with type hints and automatic validation.",
        },
        {
            "doc_id": "d4",
            "text": "Hybrid retrieval overview Hybrid retrieval combines lexical search and dense retrieval before reranking a candidate pool.",
        },
        {
            "doc_id": "d5",
            "text": "Cross-encoder reranking Cross-encoders score a query and document together and are strong second-stage rerankers.",
        },
        {
            "doc_id": "d6",
            "text": "Document freshness policy Freshness-sensitive queries should prefer recent sources and clearly track publication dates.",
        },
    ]
    queries = [
        {"query_id": "q1", "text": "how do i create a python virtual environment"},
        {"query_id": "q2", "text": "what is hybrid retrieval before reranking"},
        {"query_id": "q3", "text": "when should i use a cross encoder reranker"},
    ]
    qrels = [
        {"query_id": "q1", "doc_id": "d1", "relevance": 3},
        {"query_id": "q2", "doc_id": "d4", "relevance": 3},
        {"query_id": "q3", "doc_id": "d5", "relevance": 3},
    ]
    return docs, queries, qrels


def load_single_benchmark_dataset(dataset_id: str, max_docs: int | None = None, max_queries: int | None = None):
    if dataset_id == "toy":
        return build_toy_dataset()

    ir_datasets = import_ir_datasets()
    dataset = ir_datasets.load(dataset_id)
    docs = [doc_to_record(doc) for doc in dataset.docs_iter()]
    queries = [query_to_record(query) for query in dataset.queries_iter()]
    qrels = [qrel_to_record(qrel) for qrel in dataset.qrels_iter()]

    if max_docs is not None:
        docs = docs[:max_docs]
        allowed_doc_ids = {doc["doc_id"] for doc in docs}
        qrels = [row for row in qrels if row["doc_id"] in allowed_doc_ids]

    if max_queries is not None:
        queries = queries[:max_queries]
        allowed_query_ids = {query["query_id"] for query in queries}
        qrels = [row for row in qrels if row["query_id"] in allowed_query_ids]

    qrels_query_ids = {row["query_id"] for row in qrels}
    queries = [query for query in queries if query["query_id"] in qrels_query_ids]
    return docs, queries, qrels


def resolve_dataset_ids(dataset_spec: str) -> List[str]:
    if dataset_spec in DATASET_PRESETS:
        return DATASET_PRESETS[dataset_spec]
    return [part.strip() for part in dataset_spec.split(",") if part.strip()]


def resolve_test_fraction(dataset_spec: str, requested: float | None) -> float:
    if requested is not None:
        return requested
    if dataset_spec in PRESET_TEST_FRACTIONS:
        return PRESET_TEST_FRACTIONS[dataset_spec]
    return DEFAULT_TEST_FRACTION


def prefix_records(dataset_id: str, docs, queries, qrels):
    doc_prefix = f"{dataset_id}::"
    docs_out = []
    for doc in docs:
        updated = dict(doc)
        updated["doc_id"] = f"{doc_prefix}{doc['doc_id']}"
        updated["source_dataset"] = dataset_id
        docs_out.append(updated)

    query_prefix = f"{dataset_id}::"
    queries_out = []
    for query in queries:
        updated = dict(query)
        updated["query_id"] = f"{query_prefix}{query['query_id']}"
        updated["source_dataset"] = dataset_id
        queries_out.append(updated)

    qrels_out = []
    for row in qrels:
        qrels_out.append(
            {
                "query_id": f"{query_prefix}{row['query_id']}",
                "doc_id": f"{doc_prefix}{row['doc_id']}",
                "relevance": row["relevance"],
                "source_dataset": dataset_id,
            }
        )
    return docs_out, queries_out, qrels_out


def load_benchmark_dataset(dataset_spec: str, max_docs: int | None = None, max_queries: int | None = None):
    dataset_ids = resolve_dataset_ids(dataset_spec)
    combined_docs = []
    combined_queries = []
    combined_qrels = []
    source_summaries = []

    for dataset_id in dataset_ids:
        docs, queries, qrels = load_single_benchmark_dataset(
            dataset_id=dataset_id,
            max_docs=max_docs,
            max_queries=max_queries,
        )
        docs, queries, qrels = prefix_records(dataset_id=dataset_id, docs=docs, queries=queries, qrels=qrels)
        combined_docs.extend(docs)
        combined_queries.extend(queries)
        combined_qrels.extend(qrels)
        source_summaries.append(
            {
                "dataset": dataset_id,
                "doc_count": len(docs),
                "query_count": len(queries),
                "qrel_count": len(qrels),
            }
        )

    return combined_docs, combined_queries, combined_qrels, source_summaries


def split_queries_and_qrels(
    queries: List[Dict[str, str]],
    qrels: List[Dict[str, object]],
    test_fraction: float,
):
    query_ids = [query["query_id"] for query in queries]
    if len(query_ids) < 2 or test_fraction <= 0.0:
        test_query_ids: set[str] = set()
    else:
        ranked_ids = sorted(
            query_ids,
            key=lambda query_id: hashlib.sha256(query_id.encode("utf-8")).hexdigest(),
        )
        test_count = min(len(ranked_ids) - 1, max(1, round(len(ranked_ids) * test_fraction)))
        test_query_ids = set(ranked_ids[:test_count])

    split_map: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        "dev": {"queries": [], "qrels": []},
        "test": {"queries": [], "qrels": []},
    }
    for query in queries:
        split_name = "test" if query["query_id"] in test_query_ids else "dev"
        split_map[split_name]["queries"].append(query)

    for row in qrels:
        split_name = "test" if row["query_id"] in test_query_ids else "dev"
        split_map[split_name]["qrels"].append(row)

    return split_map


def write_split_artifacts(
    artifact_dir: Path,
    split_name: str,
    docs: List[Dict[str, str]],
    queries: List[Dict[str, str]],
    qrels: List[Dict[str, object]],
    top_k: int,
) -> None:
    split_dir = artifact_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(queries=queries, docs=docs, top_k=top_k)
    write_json(split_dir / "queries.json", queries)
    write_json(split_dir / "qrels.json", qrels)
    write_json(split_dir / "candidates.json", candidates)
    write_json(
        split_dir / "manifest.json",
        {
            "split": split_name,
            "query_count": len(queries),
            "candidate_top_k": top_k,
            "candidate_generation_version": CANDIDATE_GENERATION_VERSION,
        },
    )


def remove_legacy_root_artifacts(artifact_dir: Path) -> None:
    for filename in ("queries.json", "qrels.json", "candidates.json"):
        legacy_path = artifact_dir / filename
        if legacy_path.exists():
            legacy_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen evaluation package.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Output directory for the frozen benchmark package.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=(
            "Dataset identifier, comma-separated dataset identifiers, or a preset such as "
            f"{', '.join(sorted(DATASET_PRESETS))}. Use `toy` for the built-in fallback."
        ),
    )
    parser.add_argument("--top-k", type=int, default=100, help="Number of cached retrieval candidates per query.")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for docs during local testing.")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap for queries during local testing.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="Fraction of labeled queries held out for the test split.",
    )
    parser.add_argument(
        "--skip-reranker-cache",
        action="store_true",
        help="Skip downloading the fixed reranker checkpoint into the local Hugging Face cache.",
    )
    args = parser.parse_args()
    test_fraction = resolve_test_fraction(args.dataset_id, args.test_fraction)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    docs, queries, qrels, source_summaries = load_benchmark_dataset(
        dataset_spec=args.dataset_id,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
    )
    split_map = split_queries_and_qrels(
        queries=queries,
        qrels=qrels,
        test_fraction=test_fraction,
    )

    write_json(args.artifact_dir / "docs.json", docs)
    for split_name, split_payload in split_map.items():
        write_split_artifacts(
            artifact_dir=args.artifact_dir,
            split_name=split_name,
            docs=docs,
            queries=split_payload["queries"],
            qrels=split_payload["qrels"],
            top_k=args.top_k,
        )
    remove_legacy_root_artifacts(args.artifact_dir)
    write_json(
        args.artifact_dir / "manifest.json",
        {
            "dataset": args.dataset_id,
            "source_datasets": source_summaries,
            "candidate_top_k": args.top_k,
            "candidate_generation_version": CANDIDATE_GENERATION_VERSION,
            "doc_count": len(docs),
            "query_count": len(queries),
            "dev_query_count": len(split_map["dev"]["queries"]),
            "test_query_count": len(split_map["test"]["queries"]),
            "test_fraction": test_fraction,
        },
    )

    if not args.skip_reranker_cache:
        ensure_reranker_cached()

    print(f"Prepared frozen evaluation package for {args.dataset_id} at {args.artifact_dir}")


if __name__ == "__main__":
    main()
