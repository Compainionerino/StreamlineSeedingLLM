from __future__ import annotations

import argparse
from pathlib import Path

from rag.experiment_analysis import build_experiment_report
from rag.paths import DEFAULT_EXPERIMENT_REPORT_DIR, DEFAULT_EXPERIMENT_SESSION_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate saved experiment sessions and build a local comparison dashboard."
    )
    parser.add_argument(
        "--session-root",
        default=DEFAULT_EXPERIMENT_SESSION_ROOT,
        help="Directory containing saved session JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_EXPERIMENT_REPORT_DIR,
        help="Directory where CSV/JSON summaries and the dashboard are written.",
    )
    parser.add_argument(
        "--primary-strategy",
        choices=("latest", "best"),
        default="latest",
        help="How to choose one primary run when a condition has duplicate sessions.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_experiment_report(
        args.session_root,
        args.output_dir,
        primary_strategy=args.primary_strategy,
    )

    output_dir = Path(args.output_dir).resolve()
    totals = report["totals"]
    primary = report["overview"]["primary_conditions"]

    print("Experiment analysis complete")
    print(f"  Raw runs: {totals['raw_runs']}")
    print(f"  Primary conditions: {totals['primary_conditions']}")
    print(f"  Datasets: {totals['datasets']}")
    print(f"  RAG pairs: {totals['rag_pairs']}")
    print(f"  Mode pairs: {totals['mode_pairs']}")
    print(f"  Missing conditions: {totals['missing_conditions']}")
    print(f"  Duplicate conditions: {totals['duplicate_conditions']}")
    print(f"  Primary success rate: {primary['success_rate']:.3f}")
    print(f"  Primary average features: {primary['avg_features_all']:.3f}")
    print(f"  Primary average seeding: {primary['avg_seeding_all']:.3f}")
    print(f"  Wrote dashboard: {output_dir / 'index.html'}")
    print(f"  Wrote data: {output_dir / 'data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
