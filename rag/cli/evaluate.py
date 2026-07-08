from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.retrieval import HybridRetriever
from rag.paths import DEFAULT_EVAL_QUERIES_PATH, DEFAULT_EVALUATION_REPORT_PATH, DEFAULT_INDEX_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run simple retrieval checks over curated example queries."
    )
    parser.add_argument("--queries", default=DEFAULT_EVAL_QUERIES_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default=DEFAULT_EVALUATION_REPORT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with Path(args.queries).open("r", encoding="utf-8-sig") as handle:
        examples = json.load(handle)

    retriever = HybridRetriever(args.index_dir)
    results = []
    hit_at_5 = 0
    hit_at_k = 0

    for example in examples:
        payload = retriever.retrieve(example["query"], top_k=args.top_k)
        returned_ids = [item["application_id"] for item in payload["results"]]
        expected_ids = example.get("expected_application_ids", [])
        hits = [application_id for application_id in expected_ids if application_id in returned_ids]
        top_5_hits = [application_id for application_id in expected_ids if application_id in returned_ids[:5]]
        if top_5_hits:
            hit_at_5 += 1
        if hits:
            hit_at_k += 1
        results.append(
            {
                "id": example["id"],
                "query": example["query"],
                "query_tags": payload["query_tags"],
                "expected_tags": example.get("expected_tags", {}),
                "expected_application_ids": expected_ids,
                "returned_application_ids": returned_ids,
                "hit_at_5": bool(top_5_hits),
                f"hit_at_{args.top_k}": bool(hits),
                "hits": hits,
                "top_results": [
                    {
                        "rank": item["rank"],
                        "score": item["score"],
                        "application_id": item["application_id"],
                        "paper_title": item["paper_title"],
                        "algorithm_name": item["algorithm_name"],
                    }
                    for item in payload["results"][:5]
                ],
            }
        )

    report = {
        "query_count": len(examples),
        "top_k": args.top_k,
        "hit_at_5": hit_at_5 / len(examples) if examples else 0.0,
        f"hit_at_{args.top_k}": hit_at_k / len(examples) if examples else 0.0,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Evaluation complete")
    print(f"  Queries: {report['query_count']}")
    print(f"  hit@5: {report['hit_at_5']:.3f}")
    print(f"  hit@{args.top_k}: {report[f'hit_at_{args.top_k}']:.3f}")
    print(f"  Wrote report: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
