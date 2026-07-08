from __future__ import annotations

import argparse
from pathlib import Path

from rag.embeddings import DEFAULT_HASHING_DIMENSION, DEFAULT_SENTENCE_TRANSFORMER_MODEL
from rag.indexing import build_index
from rag.paths import DEFAULT_INDEX_DIR, DEFAULT_RECORDS_PATH, DEFAULT_VOCABULARY_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a local retrieval index for streamline seeding application records."
    )
    parser.add_argument("--data", default=DEFAULT_RECORDS_PATH)
    parser.add_argument("--vocabulary", default=DEFAULT_VOCABULARY_PATH)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--embedding-backend",
        choices=("auto", "sentence-transformers", "local-hashing"),
        default="auto",
        help="Use BGE through sentence-transformers when available, or local hashing fallback.",
    )
    parser.add_argument("--model-name", default=DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load already-cached sentence-transformer files.",
    )
    parser.add_argument("--hashing-dimension", type=int, default=DEFAULT_HASHING_DIMENSION)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = build_index(
        data_path=args.data,
        vocabulary_path=args.vocabulary,
        index_dir=args.index_dir,
        embedding_backend=args.embedding_backend,
        model_name=args.model_name,
        local_files_only=args.local_files_only,
        hashing_dimension=args.hashing_dimension,
    )
    print("Index built")
    print(f"  Records: {metadata['record_count']}")
    print(f"  Backend requested: {metadata['embedding_backend_requested']}")
    print(f"  Backend used: {metadata['embedding_backend_used']}")
    if metadata.get("embedding_backend_error"):
        print(f"  Backend fallback reason: {metadata['embedding_backend_error']}")
    print(f"  Records path: {Path(args.index_dir, 'records.jsonl').resolve()}")
    print(f"  Embeddings path: {Path(args.index_dir, 'embeddings.npy').resolve()}")
    print(f"  Metadata path: {Path(args.index_dir, 'metadata.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
