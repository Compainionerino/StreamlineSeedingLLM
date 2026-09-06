from __future__ import annotations

import argparse
from pathlib import Path

from rag.experiment_editor import serve_experiment_editor
from rag.paths import DEFAULT_EXPERIMENT_REPORT_DIR, DEFAULT_EXPERIMENT_SESSION_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the experiment dashboard with local JSON editing enabled."
    )
    parser.add_argument(
        "--session-root",
        default=DEFAULT_EXPERIMENT_SESSION_ROOT,
        help="Directory containing saved session JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_EXPERIMENT_REPORT_DIR,
        help="Directory where the dashboard and regenerated data are written.",
    )
    parser.add_argument(
        "--primary-strategy",
        choices=("latest", "best"),
        default="latest",
        help="How to choose one primary run when a condition has duplicate sessions.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local editor server.",
    )
    parser.add_argument(
        "--port",
        default=8765,
        type=int,
        help="Port for the local editor server.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        serve_experiment_editor(
            args.session_root,
            args.output_dir,
            primary_strategy=args.primary_strategy,
            repo_root=Path("."),
            host=args.host,
            port=args.port,
        )
    except KeyboardInterrupt:
        print("\nExperiment editor stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
