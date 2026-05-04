from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
DEFAULT_STRATEGY_PATH = ROOT / "rerank_strategy.py"
WORKER_SOURCE_PATH = ROOT / "strategy_worker.py"
SENSITIVE_REPO_PATHS = [
    ROOT / ".git",
    ROOT / "artifacts",
    ROOT / "docs",
    ROOT / "runs",
    ROOT / "README.md",
    ROOT / "program.md",
    ROOT / "prepare.py",
    ROOT / "eval.py",
    ROOT / "train.py",
    ROOT / "run.py",
    ROOT / "autoresearch_driver.py",
    ROOT / "pyproject.toml",
    ROOT / "results.tsv",
    ROOT / "upstream-autoresearch",
    ROOT / "rerank_strategy.py",
]


class StrategyRuntimeError(RuntimeError):
    pass


def sandbox_enabled() -> bool:
    return sys.platform == "darwin" and os.environ.get("AUTORESEARCH_DISABLE_SANDBOX", "0") != "1"


def seatbelt_quote(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def seatbelt_rule(path: Path, operation: str) -> str:
    quoted = seatbelt_quote(str(path))
    if path.is_dir():
        return f"({operation} (subpath \"{quoted}\"))"
    return f"({operation} (literal \"{quoted}\"))"


def expand_sandbox_paths(paths: List[Path]) -> List[Path]:
    expanded: List[Path] = []
    seen = set()
    for path in paths:
        for candidate in (path, path.resolve()):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(candidate)
    return expanded


def build_sandbox_profile(sandbox_dir: Path) -> str:
    temp_paths = expand_sandbox_paths(
        [
        sandbox_dir,
        Path(tempfile.gettempdir()),
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/private/tmp"),
        ]
    )
    cache_paths = expand_sandbox_paths(
        [
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch",
        Path.home() / "Library" / "Caches",
        ]
    )

    lines = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-write*)",
    ]
    for path in SENSITIVE_REPO_PATHS:
        lines.append(seatbelt_rule(path, "deny file-read*"))
    for path in [*temp_paths, *cache_paths]:
        lines.append(seatbelt_rule(path, "allow file-write*"))
    return "\n".join(lines) + "\n"


class StrategyRuntime:
    def __init__(
        self,
        *,
        strategy_path: Path = DEFAULT_STRATEGY_PATH,
        use_sandbox: bool | None = None,
        env_overrides: Dict[str, str] | None = None,
    ) -> None:
        self.strategy_path = strategy_path.resolve()
        self.use_sandbox = sandbox_enabled() if use_sandbox is None else use_sandbox
        self.env_overrides = dict(env_overrides or {})
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._sandbox_dir: Path | None = None
        self._stderr_path: Path | None = None
        self._stderr_handle = None
        self._process: subprocess.Popen[str] | None = None
        self.strategy_peak_memory_mb = 0.0
        self._strategy_config: Dict[str, Any] | None = None

    def __enter__(self) -> "StrategyRuntime":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _prepare_sandbox_dir(self) -> tuple[Path, Path]:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="autoresearch-strategy-")
        sandbox_dir = Path(self._tmpdir.name)
        worker_copy = sandbox_dir / "strategy_worker.py"
        strategy_copy = sandbox_dir / "rerank_strategy.py"
        shutil.copy2(WORKER_SOURCE_PATH, worker_copy)
        shutil.copy2(self.strategy_path, strategy_copy)
        self._sandbox_dir = sandbox_dir
        self._stderr_path = sandbox_dir / "worker.stderr.log"
        return worker_copy, strategy_copy

    def start(self) -> None:
        if self._process is not None:
            return

        worker_copy, strategy_copy = self._prepare_sandbox_dir()
        self._stderr_handle = self._stderr_path.open("w", encoding="utf-8")
        command: List[str]
        if self.use_sandbox:
            profile_path = self._sandbox_dir / "sandbox.sb"
            profile_path.write_text(build_sandbox_profile(self._sandbox_dir), encoding="utf-8")
            command = [
                "/usr/bin/sandbox-exec",
                "-f",
                str(profile_path),
                sys.executable,
                str(worker_copy),
                str(strategy_copy),
            ]
        else:
            command = [sys.executable, str(worker_copy), str(strategy_copy)]

        env = os.environ.copy()
        temp_dir = str(self._sandbox_dir)
        env.update(
            {
                "TMPDIR": temp_dir,
                "TEMP": temp_dir,
                "TMP": temp_dir,
                "TEMPDIR": temp_dir,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        env.update(self.env_overrides)
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self._sandbox_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_handle,
                text=True,
                bufsize=1,
                env=env,
            )
            self._strategy_config = self.strategy_config()
        except Exception:
            if self._process is not None:
                self._process.kill()
                self._process.wait()
                self._process = None
            if self._stderr_handle is not None:
                self._stderr_handle.close()
                self._stderr_handle = None
            if self._tmpdir is not None:
                self._tmpdir.cleanup()
                self._tmpdir = None
            raise

    def _read_stderr(self) -> str:
        if self._stderr_path is None or not self._stderr_path.exists():
            return ""
        return self._stderr_path.read_text(encoding="utf-8").strip()

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise StrategyRuntimeError("Strategy runtime has not been started.")

        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr_text = self._read_stderr()
            raise StrategyRuntimeError(
                "Strategy worker exited unexpectedly."
                + (f" stderr: {stderr_text}" if stderr_text else "")
            )

        response = json.loads(line)
        peak_memory = float(response.get("peak_memory_mb", 0.0))
        self.strategy_peak_memory_mb = max(self.strategy_peak_memory_mb, peak_memory)
        if not response.get("ok", False):
            stderr_text = self._read_stderr()
            message = str(response.get("error", "Unknown strategy worker error"))
            tb = str(response.get("traceback", "")).strip()
            detail_parts = [message]
            if tb:
                detail_parts.append(tb)
            if stderr_text:
                detail_parts.append(f"stderr:\n{stderr_text}")
            raise StrategyRuntimeError("\n\n".join(detail_parts))
        return response

    def strategy_config(self) -> Dict[str, Any]:
        response = self._request({"op": "config"})
        self._strategy_config = dict(response["strategy"])
        return dict(self._strategy_config)

    def runtime_info(self) -> Dict[str, Any]:
        return {
            "use_sandbox": self.use_sandbox,
            "env_overrides": dict(self.env_overrides),
            "strategy_peak_memory_mb": self.strategy_peak_memory_mb,
        }

    def warmup(self) -> None:
        self._request({"op": "warmup"})

    def rerank(self, query: str, candidates: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[str]:
        response = self._request(
            {
                "op": "rerank",
                "query": query,
                "candidates": candidates,
                "ctx": ctx,
            }
        )
        ranked_doc_ids = response.get("ranked_doc_ids", [])
        if not isinstance(ranked_doc_ids, list):
            raise StrategyRuntimeError("Strategy worker returned malformed ranked doc ids.")
        return [str(doc_id) for doc_id in ranked_doc_ids]

    def close(self) -> None:
        if self._process is None:
            if self._tmpdir is not None:
                self._tmpdir.cleanup()
                self._tmpdir = None
            return
        try:
            self._request({"op": "shutdown"})
        except Exception:
            pass
        finally:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
            if self._stderr_handle is not None:
                self._stderr_handle.close()
                self._stderr_handle = None
            if self._tmpdir is not None:
                self._tmpdir.cleanup()
                self._tmpdir = None
