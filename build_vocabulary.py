from __future__ import annotations

import argparse
import json
from pathlib import Path

from streamline_retrieval.loader import parse_jsonl
from streamline_retrieval.vocabulary import compare_vocabulary, derive_vocabulary, load_vocabulary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive or compare the tag vocabulary observed in application records."
    )
    parser.add_argument(
        "--data",
        default="tagged_applications.jsonl",
        help="Path to the application JSONL file.",
    )
    parser.add_argument(
        "--vocabulary",
        default="tag_vocabulary.json",
        help="Optional existing vocabulary to compare against.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for writing the derived vocabulary JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records, load_issues = parse_jsonl(args.data)
    derived = derive_vocabulary(records)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(derived, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Wrote derived vocabulary: {output_path.resolve()}")
    else:
        print(json.dumps(derived, ensure_ascii=False, indent=2))

    vocabulary_path = Path(args.vocabulary) if args.vocabulary else None
    if vocabulary_path and vocabulary_path.exists():
        vocabulary = load_vocabulary(vocabulary_path)
        unknown, unused = compare_vocabulary(vocabulary, derived)
        print("Vocabulary comparison")
        print(f"  Existing vocabulary: {vocabulary_path.resolve()}")
        print(f"  Unknown observed tags: {unknown}")
        print(f"  Unused controlled tags: {unused}")

    if load_issues:
        print(f"Load issues while deriving vocabulary: {len(load_issues)}")

    return 1 if any(issue.severity == "error" for issue in load_issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
