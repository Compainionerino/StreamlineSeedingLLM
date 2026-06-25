from __future__ import annotations

import argparse
from pathlib import Path

from streamline_retrieval.audit import audit_applications, format_summary, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit structured streamline seeding application records."
    )
    parser.add_argument(
        "--data",
        default="tagged_applications.jsonl",
        help="Path to the application JSONL file.",
    )
    parser.add_argument(
        "--vocabulary",
        default="tag_vocabulary.json",
        help="Path to the controlled tag vocabulary JSON file. If missing, tags are derived from data.",
    )
    parser.add_argument(
        "--report",
        default="reports/data_audit.json",
        help="Where to write the JSON audit report.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Print the summary without writing a report file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    vocabulary_path = Path(args.vocabulary) if args.vocabulary else None
    report = audit_applications(args.data, vocabulary_path)

    print(format_summary(report))
    if not args.no_report:
        write_report(report, args.report)
        print(f"  Wrote report: {Path(args.report).resolve()}")

    issue_counts = report["issues"]["summary"]["by_severity"]
    return 1 if issue_counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
