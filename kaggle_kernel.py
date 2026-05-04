from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Sequence


DEFAULT_WORKSPACE = Path("/kaggle/working/autoresearch-reranking")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/autoresearch-reranking-output")
DEFAULT_SOURCE_ARCHIVE = Path("/kaggle/input/autoresearch-reranking-source/autoresearch-reranking-source.tar.gz")
DEFAULT_ARTIFACTS_ARCHIVE = Path("/kaggle/input/autoresearch-reranking-artifacts/autoresearch-reranking-artifacts.tar.gz")
DEFAULT_REPO_URL = ""
DEFAULT_ACTION = "loop"
DEFAULT_LOOP_ITERATIONS = 3
DEFAULT_CONTROLLER_MODEL = "/kaggle/working/models/gemma4/gemma-4-E2B-it-Q3_K_M.gguf"
DEFAULT_PROPOSAL_MODEL = "/kaggle/working/models/qwen3/Qwen3-4B-Instruct-2507-F16.gguf"
DEFAULT_CONTROLLER_GPU = 0
DEFAULT_PROPOSAL_GPU = 1
KNOWN_ACTIONS = {"prepare", "baseline", "status", "once", "loop"}
SOURCE_SNAPSHOT_DIRNAME = "_source_snapshot"
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
MUTABLE_WORKSPACE_PATHS = {
    "rerank_strategy.py",
    "artifacts",
    "runs",
    "history",
    "results.tsv",
    ".venv",
}
SOURCE_MARKER_FILES = {"kaggle_t4_cli.py", "autoresearch_driver.py", "rerank_strategy.py", "train.py"}
DEFAULT_KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def run_command(args: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(list(args), check=True, cwd=cwd, env=env)


def is_kaggle_runtime_json(arg: str) -> bool:
    if not arg.endswith(".json"):
        return False
    name = Path(arg).name
    return name.startswith("tmp") or "papermill" in name.lower()


def is_notebook_runtime_flag(arg: str) -> bool:
    return arg in {"-f"} or arg.startswith("--HistoryManager.hist_file")


def normalize_entry_argv(argv: Sequence[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if is_notebook_runtime_flag(arg):
            if arg == "-f" and index + 1 < len(argv):
                skip_next = True
            continue
        if is_kaggle_runtime_json(arg):
            continue
        filtered.append(arg)
    if any(arg in KNOWN_ACTIONS for arg in filtered):
        return filtered

    action = (os.environ.get("AUTORESEARCH_KAGGLE_ACTION") or DEFAULT_ACTION).strip()
    if action not in KNOWN_ACTIONS:
        raise ValueError(f"unsupported AUTORESEARCH_KAGGLE_ACTION={action!r}")

    defaults = filtered[:]
    defaults.append(action)
    if action == "loop":
        defaults.append(str(int(os.environ.get("AUTORESEARCH_KAGGLE_ITERATIONS", str(DEFAULT_LOOP_ITERATIONS)))))
    return defaults


def looks_like_source_tree(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {child.name for child in path.iterdir()}
    return SOURCE_MARKER_FILES.issubset(names)


def looks_like_artifacts_tree(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "artifacts").is_dir():
        return True
    names = {child.name for child in path.iterdir()}
    return {"fast", "promotion"}.issubset(names)


def discover_kaggle_input_path(expected: Path, *, kind: str) -> Path:
    if expected.exists():
        return expected
    input_root = DEFAULT_KAGGLE_INPUT_ROOT
    if not input_root.exists():
        return expected

    archive_name = expected.name
    direct_matches = sorted(input_root.rglob(archive_name))
    if direct_matches:
        return direct_matches[0]

    if kind == "source":
        for path in sorted(input_root.rglob("*")):
            if looks_like_source_tree(path):
                return path
    if kind == "artifacts":
        for path in sorted(input_root.rglob("*")):
            if looks_like_artifacts_tree(path):
                return path
    return expected


def ensure_uv() -> None:
    try:
        run_command(["uv", "--version"])
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    run_command([sys.executable, "-m", "pip", "install", "-q", "uv"])


def safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:*") as handle:
        members = handle.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(str(destination.resolve())):
                raise ValueError(f"archive contains an unsafe path: {member.name}")
        handle.extractall(destination)


def copy_directory_contents(source_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def flatten_workspace_root(root: Path) -> None:
    top_level = [path for path in root.iterdir()]
    if len(top_level) != 1 or not top_level[0].is_dir():
        return
    root_dir = top_level[0]
    if root_dir.name in {"docs", "artifacts"}:
        return
    temp_dir = root / ".flattened"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for child in root_dir.iterdir():
        shutil.move(str(child), temp_dir / child.name)
    root_dir.rmdir()
    for child in temp_dir.iterdir():
        shutil.move(str(child), root / child.name)
    temp_dir.rmdir()


def clear_workspace(workspace: Path) -> None:
    if not workspace.exists():
        return

    def onerror(func, path, exc_info):  # type: ignore[no-untyped-def]
        target = Path(path)
        try:
            if target.parent.exists():
                target.parent.chmod(0o755)
            target.chmod(0o755 if target.is_dir() else 0o644)
        except OSError:
            pass
        func(path)

    shutil.rmtree(workspace, onerror=onerror)


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def lock_tree_read_only(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o555)
        else:
            path.chmod(0o444)
    root.chmod(0o555)


def snapshot_path(workspace: Path) -> Path:
    return workspace / SOURCE_SNAPSHOT_DIRNAME


def required_snapshot_paths() -> list[Path]:
    return [Path(relative) for relative in SOURCE_SNAPSHOT_FILES]


def build_live_workspace(workspace: Path) -> None:
    snapshot_root = snapshot_path(workspace)
    if not snapshot_root.exists():
        raise FileNotFoundError(f"missing source snapshot: {snapshot_root}")

    for relative in required_snapshot_paths():
        source_path = snapshot_root / relative
        if not source_path.exists():
            raise FileNotFoundError(f"source snapshot missing required path: {relative}")
        target_path = workspace / relative
        remove_path(target_path)
        ensure_parent(target_path)
        shutil.copy2(source_path, target_path)
        if relative.as_posix() == "rerank_strategy.py":
            target_path.chmod(0o644)
        else:
            target_path.chmod(0o444)

    for relative in MUTABLE_WORKSPACE_PATHS:
        mutable_path = workspace / relative
        if relative.endswith(".py") or relative.endswith(".tsv"):
            ensure_parent(mutable_path)
            if not mutable_path.exists():
                mutable_path.touch()
        else:
            mutable_path.mkdir(parents=True, exist_ok=True)

    lock_tree_read_only(snapshot_root)


def materialize_archive_snapshot(*, archive_path: Path, workspace: Path, rebuild: bool) -> None:
    if rebuild:
        clear_workspace(workspace)
    snapshot_root = snapshot_path(workspace)
    if snapshot_root.exists() and any(snapshot_root.iterdir()):
        return
    workspace.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)
    if archive_path.is_dir():
        copy_directory_contents(archive_path, snapshot_root)
    else:
        safe_extract(archive_path, snapshot_root)
    flatten_workspace_root(snapshot_root)


def materialize_git_snapshot(*, repo_url: str, repo_ref: str, workspace: Path, rebuild: bool) -> None:
    if not repo_url:
        raise ValueError("--repo-url is required for --source-mode git")
    if rebuild:
        clear_workspace(workspace)
    snapshot_root = snapshot_path(workspace)
    if snapshot_root.exists() and (snapshot_root / ".git").exists():
        return
    workspace.mkdir(parents=True, exist_ok=True)
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    run_command(["git", "clone", "--depth", "1", repo_url, str(snapshot_root)])
    if repo_ref:
        run_command(["git", "fetch", "--depth", "1", "origin", repo_ref], cwd=snapshot_root)
        run_command(["git", "checkout", repo_ref], cwd=snapshot_root)


def materialize_artifacts_archive(*, archive_path: Path, workspace: Path) -> None:
    if not archive_path.exists():
        return
    artifacts_root = workspace / "artifacts"
    if artifacts_root.exists():
        shutil.rmtree(artifacts_root)
    if archive_path.is_dir():
        source_root = archive_path / "artifacts" if (archive_path / "artifacts").exists() else archive_path
        shutil.copytree(source_root, artifacts_root, dirs_exist_ok=True)
    else:
        safe_extract(archive_path, workspace)
    extracted_artifacts = artifacts_root
    if not extracted_artifacts.exists():
        raise FileNotFoundError(f"artifact archive did not produce workspace/artifacts: {archive_path}")
    for path in extracted_artifacts.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


def sync_workspace(workspace: Path) -> None:
    ensure_uv()
    run_command(["uv", "sync"], cwd=workspace)


def kaggle_cli_command(args: argparse.Namespace) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "kaggle_t4_cli.py",
        "--parallel-devices",
        args.parallel_devices,
        "--rerank-batch-size",
        str(args.rerank_batch_size),
        "--cache-dir",
        str(args.cache_dir),
    ]
    if args.action == "prepare":
        command.append("prepare")
        return command
    if args.action == "baseline":
        command.append("baseline")
        return command
    if args.action == "status":
        command.append("status")
        return command

    command.append(args.action)
    if args.action == "loop":
        command.append(str(args.iterations))
    command.extend(
        [
            "--label-prefix",
            args.label_prefix,
            "--brain-backend",
            args.brain_backend,
            "--controller-model",
            args.controller_model,
            "--proposal-model",
            args.proposal_model,
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
    )
    if args.stop_after_keep:
        command.append("--stop-after-keep")
    return command


def export_results(workspace: Path, args: argparse.Namespace) -> None:
    export_command = [
        "uv",
        "run",
        "python",
        "kaggle_t4_cli.py",
        "--parallel-devices",
        args.parallel_devices,
        "--rerank-batch-size",
        str(args.rerank_batch_size),
        "--cache-dir",
        str(args.cache_dir),
        "export-results",
        "--output-dir",
        str(args.output_dir),
        "--bundle-name",
        args.bundle_name,
    ]
    if args.dataset_slug:
        export_command.extend(["--dataset-slug", args.dataset_slug])
    if args.dataset_owner:
        export_command.extend(["--dataset-owner", args.dataset_owner])
    if args.dataset_title:
        export_command.extend(["--dataset-title", args.dataset_title])
    run_command(export_command, cwd=workspace)


def cleanup_runtime_outputs(workspace: Path) -> None:
    # Kaggle persists anything left under the working directory as kernel output.
    remove_path(workspace / ".venv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_kernel.py",
        description="Single-file Kaggle bootstrap for the autoresearch reranking workflow.",
    )
    parser.add_argument("--source-mode", choices=["archive", "git"], default="archive")
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--artifacts-archive", type=Path, default=DEFAULT_ARTIFACTS_ARCHIVE)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default="")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=Path("/kaggle/working/.cache/huggingface"))
    parser.add_argument("--rebuild-workspace", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--unlock-source", action="store_true")
    parser.add_argument("--bundle-name", default="autoresearch-reranking-output")
    parser.add_argument("--dataset-slug", default="")
    parser.add_argument("--dataset-owner", default="")
    parser.add_argument("--dataset-title", default="")
    parser.add_argument("--parallel-devices", default="0,1")
    parser.add_argument("--rerank-batch-size", type=int, default=12)

    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "baseline", "status"):
        subparsers.add_parser(action)

    once_parser = subparsers.add_parser("once")
    once_parser.add_argument("--label-prefix", default="kaggle-fiqa-t4x2")
    once_parser.add_argument("--brain-backend", default="auto", choices=["auto", "llama-cpp", "transformers"])
    once_parser.add_argument("--controller-model", default=DEFAULT_CONTROLLER_MODEL)
    once_parser.add_argument("--proposal-model", default=DEFAULT_PROPOSAL_MODEL)
    once_parser.add_argument("--controller-gpu", type=int, default=DEFAULT_CONTROLLER_GPU)
    once_parser.add_argument("--proposal-gpu", type=int, default=DEFAULT_PROPOSAL_GPU)
    once_parser.add_argument("--max-tokens", type=int, default=1600)
    once_parser.add_argument("--temperature", type=float, default=0.2)
    once_parser.add_argument("--top-p", type=float, default=0.9)
    once_parser.add_argument("--explore-trials-per-family", type=int, default=1)
    once_parser.add_argument("--exploit-top-families", type=int, default=2)
    once_parser.add_argument("--stop-after-keep", action="store_true")

    loop_parser = subparsers.add_parser("loop")
    loop_parser.add_argument("iterations", nargs="?", type=int, default=3)
    loop_parser.add_argument("--label-prefix", default="kaggle-fiqa-t4x2")
    loop_parser.add_argument("--brain-backend", default="auto", choices=["auto", "llama-cpp", "transformers"])
    loop_parser.add_argument("--controller-model", default=DEFAULT_CONTROLLER_MODEL)
    loop_parser.add_argument("--proposal-model", default=DEFAULT_PROPOSAL_MODEL)
    loop_parser.add_argument("--controller-gpu", type=int, default=DEFAULT_CONTROLLER_GPU)
    loop_parser.add_argument("--proposal-gpu", type=int, default=DEFAULT_PROPOSAL_GPU)
    loop_parser.add_argument("--max-tokens", type=int, default=1600)
    loop_parser.add_argument("--temperature", type=float, default=0.2)
    loop_parser.add_argument("--top-p", type=float, default=0.9)
    loop_parser.add_argument("--explore-trials-per-family", type=int, default=1)
    loop_parser.add_argument("--exploit-top-families", type=int, default=2)
    loop_parser.add_argument("--stop-after-keep", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(normalize_entry_argv(sys.argv[1:]))
    args.source_archive = discover_kaggle_input_path(Path(args.source_archive), kind="source")
    args.artifacts_archive = discover_kaggle_input_path(Path(args.artifacts_archive), kind="artifacts")

    if args.source_mode == "archive":
        materialize_archive_snapshot(
            archive_path=args.source_archive,
            workspace=args.workspace,
            rebuild=args.rebuild_workspace,
        )
    else:
        materialize_git_snapshot(
            repo_url=args.repo_url,
            repo_ref=args.repo_ref,
            workspace=args.workspace,
            rebuild=args.rebuild_workspace,
        )

    build_live_workspace(args.workspace)
    materialize_artifacts_archive(archive_path=args.artifacts_archive, workspace=args.workspace)
    if args.unlock_source:
        for relative in required_snapshot_paths():
            path = args.workspace / relative
            if path.exists():
                path.chmod(0o644)

    if not args.skip_sync:
        sync_workspace(args.workspace)

    failure: BaseException | None = None
    try:
        run_command(kaggle_cli_command(args), cwd=args.workspace, env=os.environ.copy())
    except BaseException as exc:
        failure = exc
    finally:
        try:
            export_results(args.workspace, args)
        finally:
            cleanup_runtime_outputs(args.workspace)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
