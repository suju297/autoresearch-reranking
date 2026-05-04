from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from history_report import refresh_history_exports
from local_brain import DEFAULT_BRAIN_MAX_TOKENS


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
FAST_ARTIFACT_DIR = ARTIFACTS_DIR / "fast" / "fiqa-dev"
PROMOTION_ARTIFACT_DIR = ARTIFACTS_DIR / "promotion"
REPORT_ARTIFACT_DIR = ARTIFACTS_DIR / "report"
RUNS_DIR = ROOT / "runs"
HISTORY_DIR = ROOT / "history"
RESULTS_PATH = ROOT / "results.tsv"

DEFAULT_LABEL_PREFIX = "kaggle-fiqa-t4x2"
DEFAULT_CONTROLLER_GGUF_REPO = "unsloth/gemma-4-E2B-it-GGUF"
DEFAULT_CONTROLLER_GGUF_FILENAME = "gemma-4-E2B-it-Q3_K_M.gguf"
DEFAULT_CONTROLLER_GGUF_DIR = Path("/kaggle/working/models/gemma4")
DEFAULT_CONTROLLER_MODEL = str(DEFAULT_CONTROLLER_GGUF_DIR / DEFAULT_CONTROLLER_GGUF_FILENAME)
DEFAULT_PROPOSAL_GGUF_REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
DEFAULT_PROPOSAL_GGUF_FILENAME = "Qwen3-4B-Instruct-2507-F16.gguf"
DEFAULT_PROPOSAL_GGUF_DIR = Path("/kaggle/working/models/qwen3")
DEFAULT_PROPOSAL_MODEL = str(DEFAULT_PROPOSAL_GGUF_DIR / DEFAULT_PROPOSAL_GGUF_FILENAME)
DEFAULT_CACHE_DIR = Path("/kaggle/working/.cache/huggingface")
DEFAULT_EXPORT_DIR = Path("/kaggle/working/autoresearch-reranking-output")
DEFAULT_PARALLEL_DEVICES = "0,1"
DEFAULT_RERANK_BATCH_SIZE = 12
DEFAULT_SOURCE_BUNDLE = ROOT / "dist" / "kaggle" / "autoresearch-reranking-source.tar.gz"
DEFAULT_ARTIFACTS_BUNDLE = ROOT / "dist" / "kaggle" / "autoresearch-reranking-artifacts.tar.gz"
DEFAULT_RESULTS_BUNDLE = "autoresearch-reranking-output"
SOURCE_BUNDLE_EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    "artifacts",
    "history",
    "runs",
    "dist",
}
SOURCE_BUNDLE_SUFFIX_EXCLUDES = {".pyc", ".pyo"}
SOURCE_SNAPSHOT_FILES = [
    "README.md",
    "program.md",
    "pyproject.toml",
    "uv.lock",
    "prepare.py",
    "eval.py",
    "rerank_strategy.py",
    "train.py",
    "run.py",
    "local_brain.py",
    "strategy_worker.py",
    "strategy_runner.py",
    "autoresearch_driver.py",
    "proposal_validator.py",
    "history_report.py",
    "autoresearch_reranking_cli.py",
    "kaggle_t4_cli.py",
    "kaggle_kernel.py",
    "docs/reranking_playbook.md",
    "docs/metric-policy.md",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_command(
    args: list[str],
    *,
    env: dict[str, str],
    cwd: Path = ROOT,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        cwd=cwd,
        env=env,
        capture_output=capture_output,
        text=True,
    )


def current_commit() -> str:
    try:
        result = run_command(["git", "rev-parse", "HEAD"], env=os.environ.copy(), capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return result.stdout.strip()


def kaggle_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("HF_HOME", str(cache_dir))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    if importlib.util.find_spec("hf_transfer") is not None:
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    else:
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env.setdefault("AUTORESEARCH_DISABLE_SANDBOX", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("UV_LINK_MODE", "copy")
    env["AUTORESEARCH_RERANK_PARALLEL_DEVICES"] = args.parallel_devices
    env["AUTORESEARCH_RERANK_MODEL_BATCH_SIZE"] = str(args.rerank_batch_size)
    return env


def resolve_model_name(model_name: str, *, cache_dir: Path) -> str:
    model_path = Path(model_name)
    default_downloads = {
        Path(DEFAULT_CONTROLLER_MODEL): (
            DEFAULT_CONTROLLER_GGUF_REPO,
            DEFAULT_CONTROLLER_GGUF_FILENAME,
            DEFAULT_CONTROLLER_GGUF_DIR,
        ),
        Path(DEFAULT_PROPOSAL_MODEL): (
            DEFAULT_PROPOSAL_GGUF_REPO,
            DEFAULT_PROPOSAL_GGUF_FILENAME,
            DEFAULT_PROPOSAL_GGUF_DIR,
        ),
    }
    if model_path in default_downloads:
        if model_path.exists():
            return str(model_path)
        from huggingface_hub import hf_hub_download

        repo_id, filename, local_dir = default_downloads[model_path]
        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            cache_dir=str(cache_dir),
        )
        return str(Path(downloaded))
    return model_name


def artifact_inventory() -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    if not ARTIFACTS_DIR.exists():
        return inventory
    for path in sorted(ARTIFACTS_DIR.glob("*")):
        if not path.exists():
            continue
        file_count = 0
        total_bytes = 0
        for child in path.rglob("*"):
            if child.is_file():
                file_count += 1
                total_bytes += child.stat().st_size
        inventory.append(
            {
                "path": str(path.relative_to(ROOT)),
                "file_count": file_count,
                "total_mb": round(total_bytes / (1024.0 * 1024.0), 3),
            }
        )
    return inventory


def copy_path(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def source_snapshot_paths() -> list[Path]:
    snapshot_paths: list[Path] = []
    for relative in SOURCE_SNAPSHOT_FILES:
        path = ROOT / relative
        if path.exists():
            snapshot_paths.append(path)
    return snapshot_paths


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def dataset_metadata_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.dataset_slug:
        return None
    owner = args.dataset_owner or os.environ.get("KAGGLE_USERNAME", "").strip()
    dataset_id = f"{owner}/{args.dataset_slug}" if owner else args.dataset_slug
    return {
        "title": args.dataset_title or args.bundle_name.replace("-", " ").title(),
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }


def bundle_directory(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            handle.write(path, arcname=path.relative_to(source_dir))


def export_results_bundle(args: argparse.Namespace) -> dict[str, str]:
    output_dir = Path(args.output_dir)
    bundle_root = output_dir / args.bundle_name
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    history_paths = refresh_history_exports(output_dir=bundle_root / "history", runs_dir=RUNS_DIR, results_path=RESULTS_PATH)
    copy_path(RUNS_DIR, bundle_root / "runs")
    copy_path(RESULTS_PATH, bundle_root / "results.tsv")

    source_snapshot_dir = bundle_root / "source"
    for path in source_snapshot_paths():
        copy_path(path, source_snapshot_dir / path.relative_to(ROOT))

    environment = {
        "generated_at": utc_now(),
        "repo_root": str(ROOT),
        "git_commit": current_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "parallel_devices": args.parallel_devices,
        "rerank_batch_size": args.rerank_batch_size,
        "artifact_inventory": artifact_inventory(),
        "env_subset": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("KAGGLE_") or key.startswith("AUTORESEARCH_") or key in {"CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE"}
        },
    }

    try:
        import torch

        environment["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception as exc:  # pragma: no cover - best effort only
        environment["torch_error"] = str(exc)

    metadata = {
        "bundle_name": args.bundle_name,
        "generated_at": utc_now(),
        "history_paths": history_paths,
        "runs_dir": str(RUNS_DIR),
        "results_path": str(RESULTS_PATH),
    }
    write_json(bundle_root / "metadata.json", metadata)
    write_json(bundle_root / "environment.json", environment)

    dataset_metadata = dataset_metadata_payload(args)
    if dataset_metadata is not None:
        write_json(bundle_root / "dataset-metadata.json", dataset_metadata)

    archive_path = output_dir / f"{args.bundle_name}.zip"
    bundle_directory(bundle_root, archive_path)
    return {
        "bundle_root": str(bundle_root),
        "archive_path": str(archive_path),
    }


def command_prepare(args: argparse.Namespace) -> None:
    env = kaggle_env(args)
    python = sys.executable
    run_command([python, "prepare.py", "--dataset-id", "fast-fiqa-dev", "--artifact-dir", str(FAST_ARTIFACT_DIR)], env=env)
    run_command([python, "prepare.py", "--dataset-id", "promotion-scifact", "--artifact-dir", str(PROMOTION_ARTIFACT_DIR)], env=env)
    run_command([python, "prepare.py", "--dataset-id", "report-fiqa-test", "--artifact-dir", str(REPORT_ARTIFACT_DIR)], env=env)


def command_baseline(args: argparse.Namespace) -> None:
    env = kaggle_env(args)
    python = sys.executable
    run_command(
        [python, "run.py", "--artifact-dir", str(FAST_ARTIFACT_DIR), "--label", "fiqa-dev-bge-base-baseline-kaggle", "--split", "dev"],
        env=env,
    )
    run_command(
        [python, "run.py", "--artifact-dir", str(PROMOTION_ARTIFACT_DIR), "--label", "promotion-scifact-bge-base-baseline-kaggle", "--split", "dev"],
        env=env,
    )
    run_command(
        [python, "run.py", "--artifact-dir", str(REPORT_ARTIFACT_DIR), "--label", "fiqa-test-bge-base-baseline-kaggle", "--split", "dev"],
        env=env,
    )


def build_loop_command(args: argparse.Namespace) -> list[str]:
    python = sys.executable
    controller_model = resolve_model_name(args.controller_model, cache_dir=Path(args.cache_dir))
    proposal_model = resolve_model_name(args.proposal_model, cache_dir=Path(args.cache_dir))
    command = [
        python,
        "autoresearch_driver.py",
        "auto-loop",
        "--iterations",
        str(args.iterations),
        "--artifact-dir",
        str(ROOT / "artifacts" / "fast"),
        "--promotion-artifact-dir",
        str(PROMOTION_ARTIFACT_DIR),
        "--label-prefix",
        args.label_prefix,
        "--skip-git",
        "--brain-backend",
        args.brain_backend,
        "--controller-model",
        controller_model,
        "--proposal-model",
        proposal_model,
        "--controller-gpu",
        str(args.controller_gpu),
        "--proposal-gpu",
        str(args.proposal_gpu),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--explore-trials-per-family",
        str(args.explore_trials_per_family),
        "--exploit-top-families",
        str(args.exploit_top_families),
    ]
    if args.stop_after_keep:
        command.append("--stop-after-keep")
    return command


def command_loop(args: argparse.Namespace) -> None:
    env = kaggle_env(args)
    run_command(build_loop_command(args), env=env)


def command_once(args: argparse.Namespace) -> None:
    args.iterations = 1
    command_loop(args)


def command_status(args: argparse.Namespace) -> None:
    env = kaggle_env(args)
    run_command([sys.executable, "autoresearch_driver.py", "status"], env=env)


def command_package_source(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    source_files = source_snapshot_paths()
    with tarfile.open(output_path, "w:gz") as handle:
        for path in source_files:
            handle.add(path, arcname=path.relative_to(ROOT).as_posix())

    manifest = {
        "generated_at": utc_now(),
        "repo_root": str(ROOT),
        "git_commit": current_commit(),
        "file_count": len(source_files),
        "files": [str(path.relative_to(ROOT)) for path in source_files],
    }
    write_json(output_path.with_name(output_path.name + ".manifest.json"), manifest)


def command_package_artifacts(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    if not ARTIFACTS_DIR.exists():
        raise FileNotFoundError(f"missing artifacts directory: {ARTIFACTS_DIR}")

    with tarfile.open(output_path, "w:gz") as handle:
        handle.add(ARTIFACTS_DIR, arcname="artifacts")

    manifest = {
        "generated_at": utc_now(),
        "repo_root": str(ROOT),
        "artifact_inventory": artifact_inventory(),
    }
    write_json(output_path.with_name(output_path.name + ".manifest.json"), manifest)


def command_export_results(args: argparse.Namespace) -> None:
    bundle_paths = export_results_bundle(args)
    print(json.dumps(bundle_paths, indent=2))


def add_shared_kaggle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parallel-devices", default=DEFAULT_PARALLEL_DEVICES, help="Comma-separated CUDA device ordinals for the reranker, for example `0,1`.")
    parser.add_argument("--rerank-batch-size", type=int, default=DEFAULT_RERANK_BATCH_SIZE, help="Per-worker reranker batch size.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Hugging Face cache directory inside Kaggle working storage.")


def add_loop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--label-prefix", default=DEFAULT_LABEL_PREFIX)
    parser.add_argument("--brain-backend", default="auto", choices=["auto", "llama-cpp", "transformers"])
    parser.add_argument("--controller-model", default=DEFAULT_CONTROLLER_MODEL)
    parser.add_argument("--proposal-model", default=DEFAULT_PROPOSAL_MODEL)
    parser.add_argument("--controller-gpu", type=int, default=0)
    parser.add_argument("--proposal-gpu", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_BRAIN_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--explore-trials-per-family", type=int, default=1)
    parser.add_argument("--exploit-top-families", type=int, default=2)
    parser.add_argument("--stop-after-keep", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoresearch-reranking-kaggle",
        description="Kaggle T4x2 operator CLI for the reranking autoresearch loop.",
    )
    add_shared_kaggle_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare Kaggle artifacts and cache the fixed reranker.")
    prepare_parser.set_defaults(func=command_prepare)

    baseline_parser = subparsers.add_parser("baseline", help="Run the three standard baseline evaluations on Kaggle.")
    baseline_parser.set_defaults(func=command_baseline)

    loop_parser = subparsers.add_parser("loop", help="Run the autonomous loop on Kaggle.")
    loop_parser.add_argument("iterations", nargs="?", type=int, default=3)
    add_loop_args(loop_parser)
    loop_parser.set_defaults(func=command_loop)

    once_parser = subparsers.add_parser("once", help="Run exactly one autonomous iteration on Kaggle.")
    once_parser.add_argument("--iterations", type=int, default=1, help=argparse.SUPPRESS)
    add_loop_args(once_parser)
    once_parser.set_defaults(func=command_once)

    status_parser = subparsers.add_parser("status", help="Show the current benchmark status under the Kaggle env.")
    status_parser.set_defaults(func=command_status)

    package_parser = subparsers.add_parser("package-source", help="Build a source archive that can be attached to a Kaggle notebook as a dataset.")
    package_parser.add_argument("--output-path", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    package_parser.set_defaults(func=command_package_source)

    artifacts_parser = subparsers.add_parser("package-artifacts", help="Build an artifacts archive that can be attached to Kaggle as a second dataset.")
    artifacts_parser.add_argument("--output-path", type=Path, default=DEFAULT_ARTIFACTS_BUNDLE)
    artifacts_parser.set_defaults(func=command_package_artifacts)

    export_parser = subparsers.add_parser("export-results", help="Snapshot runs, history, and source into a portable Kaggle output bundle.")
    export_parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    export_parser.add_argument("--bundle-name", default=DEFAULT_RESULTS_BUNDLE)
    export_parser.add_argument("--dataset-slug", default="")
    export_parser.add_argument("--dataset-owner", default="")
    export_parser.add_argument("--dataset-title", default="")
    export_parser.set_defaults(func=command_export_results)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
