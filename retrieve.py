from __future__ import annotations

import argparse
import json

from streamline_retrieval.retrieval import retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve relevant streamline seeding application records."
    )
    parser.add_argument("query", help="Visualization/seeding problem to retrieve for.")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print a compact human-readable summary instead of full JSON.",
    )
    return parser


def print_pretty(payload: dict) -> None:
    print(f"Query: {payload['query']}")
    print(f"Query tags: {payload['query_tags']}")
    print(
        "Index: "
        f"{payload['index_metadata']['record_count']} records, "
        f"{payload['index_metadata']['embedding_backend_used']} "
        f"({payload['index_metadata']['embedding_model']})"
    )
    print()
    for result in payload["results"]:
        print(f"{result['rank']}. {result['score']:.3f} | {result['algorithm_name']}")
        print(f"   {result['paper_title']}")
        print(f"   tags: dim={result['dimension_tags']} feature={result['feature_tags']} task={result['task_tags']}")
        print(f"   why: {'; '.join(result['why_retrieved'])}")
        strategy = result.get("seed_placement_strategy") or ""
        if strategy:
            print(f"   strategy: {strategy[:240]}")
        print()


def main() -> int:
    args = build_parser().parse_args()
    payload = retrieve(args.query, index_dir=args.index_dir, top_k=args.top_k)
    if args.pretty:
        print_pretty(payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
