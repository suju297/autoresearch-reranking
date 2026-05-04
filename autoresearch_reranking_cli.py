from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FAST_ARTIFACT_DIR = ROOT / "artifacts" / "fast"
FAST_FIQA_ARTIFACT_DIR = FAST_ARTIFACT_DIR / "fiqa-dev"
PROMOTION_ARTIFACT_DIR = ROOT / "artifacts" / "promotion"
REPORT_ARTIFACT_DIR = ROOT / "artifacts" / "report"
DEFAULT_LABEL_PREFIX = "fiqa-base-v1"


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT)


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def command_prepare(_: argparse.Namespace) -> None:
    python = sys.executable
    run_command([python, "prepare.py", "--dataset-id", "fast-fiqa-dev", "--artifact-dir", str(FAST_FIQA_ARTIFACT_DIR)])
    run_command([python, "prepare.py", "--dataset-id", "promotion-scifact", "--artifact-dir", str(PROMOTION_ARTIFACT_DIR)])
    run_command([python, "prepare.py", "--dataset-id", "report-fiqa-test", "--artifact-dir", str(REPORT_ARTIFACT_DIR)])


def command_baseline(_: argparse.Namespace) -> None:
    python = sys.executable
    commit = current_commit()
    run_command(
        [
            python,
            "run.py",
            "--artifact-dir",
            str(FAST_FIQA_ARTIFACT_DIR),
            "--label",
            "fiqa-dev-bge-base-baseline",
            "--split",
            "dev",
            "--run-id",
            f"fiqa-dev-bge-base-baseline-{commit}",
        ]
    )
    run_command(
        [
            python,
            "run.py",
            "--artifact-dir",
            str(PROMOTION_ARTIFACT_DIR),
            "--label",
            "promotion-scifact-bge-base-baseline",
            "--split",
            "dev",
            "--run-id",
            f"promotion-scifact-bge-base-baseline-{commit}",
        ]
    )
    run_command(
        [
            python,
            "run.py",
            "--artifact-dir",
            str(REPORT_ARTIFACT_DIR),
            "--label",
            "fiqa-test-bge-base-baseline",
            "--split",
            "dev",
            "--run-id",
            f"fiqa-test-bge-base-baseline-{commit}",
        ]
    )


def command_loop(args: argparse.Namespace) -> None:
    python = sys.executable
    run_command(
        [
            python,
            "autoresearch_driver.py",
            "auto-loop",
            "--iterations",
            str(args.iterations),
            "--artifact-dir",
            str(FAST_ARTIFACT_DIR),
            "--promotion-artifact-dir",
            str(PROMOTION_ARTIFACT_DIR),
            "--label-prefix",
            args.label_prefix,
        ]
    )


def command_once(args: argparse.Namespace) -> None:
    python = sys.executable
    run_command(
        [
            python,
            "autoresearch_driver.py",
            "auto-loop",
            "--iterations",
            "1",
            "--artifact-dir",
            str(FAST_ARTIFACT_DIR),
            "--promotion-artifact-dir",
            str(PROMOTION_ARTIFACT_DIR),
            "--label-prefix",
            args.label_prefix,
        ]
    )


def command_history(_: argparse.Namespace) -> None:
    run_command([sys.executable, "autoresearch_driver.py", "export-history"])


def command_status(_: argparse.Namespace) -> None:
    run_command([sys.executable, "autoresearch_driver.py", "status"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoresearch-reranking",
        description="Simple operator CLI for the FiQA-first autoresearch loop.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare the fast, promotion, and report artifacts.")
    prepare_parser.set_defaults(func=command_prepare)

    baseline_parser = subparsers.add_parser("baseline", help="Run the three standard baseline evaluations.")
    baseline_parser.set_defaults(func=command_baseline)

    loop_parser = subparsers.add_parser("loop", help="Run an autoresearch loop.")
    loop_parser.add_argument("iterations", nargs="?", type=int, default=5, help="Number of loop iterations to run.")
    loop_parser.add_argument("--label-prefix", default=DEFAULT_LABEL_PREFIX, help="Prefix for generated loop labels.")
    loop_parser.set_defaults(func=command_loop)

    once_parser = subparsers.add_parser("once", help="Run exactly one autoresearch iteration.")
    once_parser.add_argument("--label-prefix", default=DEFAULT_LABEL_PREFIX, help="Prefix for the generated loop label.")
    once_parser.set_defaults(func=command_once)

    history_parser = subparsers.add_parser("history", help="Refresh the exported history dashboard and CSVs.")
    history_parser.set_defaults(func=command_history)

    status_parser = subparsers.add_parser("status", help="Show the current benchmark and git status.")
    status_parser.set_defaults(func=command_status)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
