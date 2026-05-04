from __future__ import annotations

import importlib.util
import json
import resource
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List


def peak_memory_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def load_strategy_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("sandboxed_rerank_strategy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load strategy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "rerank"):
        raise AttributeError("Strategy module is missing `rerank`.")
    if not hasattr(module, "strategy_config"):
        raise AttributeError("Strategy module is missing `strategy_config`.")
    return module


def json_response(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def validate_ranked_items(items: List[Dict[str, Any]]) -> List[str]:
    ranked_doc_ids: List[str] = []
    for item in items:
        doc_id = item.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Strategy returned an item without a valid `doc_id`.")
        ranked_doc_ids.append(doc_id)
    return ranked_doc_ids


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: strategy_worker.py <strategy-path>", file=sys.stderr)
        return 2

    sys.stdout.reconfigure(line_buffering=True)
    strategy_path = Path(sys.argv[1]).resolve()
    module = load_strategy_module(strategy_path)
    rerank = getattr(module, "rerank")
    strategy_config = getattr(module, "strategy_config")
    warmup = getattr(module, "get_reranker", None)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            op = request.get("op")

            if op == "config":
                json_response(
                    {
                        "ok": True,
                        "strategy": strategy_config(),
                        "peak_memory_mb": peak_memory_mb(),
                    }
                )
                continue

            if op == "warmup":
                if callable(warmup):
                    warmup()
                json_response({"ok": True, "peak_memory_mb": peak_memory_mb()})
                continue

            if op == "rerank":
                reranked = rerank(
                    request["query"],
                    request["candidates"],
                    ctx=request.get("ctx", {}),
                )
                if not isinstance(reranked, list):
                    raise TypeError("Strategy rerank output must be a list.")
                ranked_doc_ids = validate_ranked_items(reranked)
                json_response(
                    {
                        "ok": True,
                        "ranked_doc_ids": ranked_doc_ids,
                        "peak_memory_mb": peak_memory_mb(),
                    }
                )
                continue

            if op == "shutdown":
                json_response({"ok": True, "peak_memory_mb": peak_memory_mb()})
                return 0

            raise ValueError(f"Unknown worker op: {op}")
        except Exception as exc:
            json_response(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "peak_memory_mb": peak_memory_mb(),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
