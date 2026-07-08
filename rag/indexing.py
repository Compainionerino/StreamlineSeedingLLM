from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .audit import file_sha256
from .embeddings import (
    DEFAULT_HASHING_DIMENSION,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    EmbeddingBackendError,
    LocalHashingEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
)
from .loader import load_applications
from .paths import DEFAULT_INDEX_DIR, DEFAULT_RECORDS_PATH, DEFAULT_VOCABULARY_PATH
from .text import EMBEDDING_TEXT_FIELDS, build_retrieval_text, json_dump_line, project_result_record
from .vocabulary import resolve_vocabulary


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json_dump_line(record))
            handle.write("\n")


def build_index(
    *,
    data_path: str | Path = DEFAULT_RECORDS_PATH,
    vocabulary_path: str | Path | None = DEFAULT_VOCABULARY_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    embedding_backend: str = "auto",
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    local_files_only: bool = False,
    hashing_dimension: int = DEFAULT_HASHING_DIMENSION,
) -> dict[str, Any]:
    """Build retrieval records and embeddings for local hybrid retrieval."""

    source_path = Path(data_path)
    destination = Path(index_dir)
    records = load_applications(source_path, strict=True)
    vocabulary_bundle = resolve_vocabulary(records, vocabulary_path)

    retrieval_records = [project_result_record(record.data) for record in records]
    retrieval_texts = [build_retrieval_text(record.data) for record in records]

    backend_name = embedding_backend
    backend_error: str | None = None
    if embedding_backend in {"auto", "sentence-transformers"}:
        try:
            sentence_backend = SentenceTransformerEmbeddingBackend(
                model_name=model_name,
                local_files_only=local_files_only,
            )
            embeddings, embedding_metadata = sentence_backend.fit_encode(retrieval_texts)
            backend_name = "sentence-transformers"
        except EmbeddingBackendError as exc:
            if embedding_backend == "sentence-transformers":
                raise
            backend_error = str(exc)
            local_backend = LocalHashingEmbeddingBackend(dimension=hashing_dimension)
            embeddings, embedding_metadata = local_backend.fit_encode(retrieval_texts)
            backend_name = "local-hashing"
    elif embedding_backend == "local-hashing":
        local_backend = LocalHashingEmbeddingBackend(dimension=hashing_dimension)
        embeddings, embedding_metadata = local_backend.fit_encode(retrieval_texts)
    else:
        raise ValueError(
            "embedding_backend must be one of: auto, sentence-transformers, local-hashing"
        )

    if len(retrieval_records) != embeddings.shape[0]:
        raise RuntimeError("Embedding row count does not match retrieval record count.")

    destination.mkdir(parents=True, exist_ok=True)
    records_path = destination / "records.jsonl"
    embeddings_path = destination / "embeddings.npy"
    metadata_path = destination / "metadata.json"

    _write_jsonl(records_path, retrieval_records)
    np.save(embeddings_path, embeddings.astype(np.float32, copy=False))

    metadata = {
        "created_at_utc": utc_now_iso(),
        "source_data_path": str(source_path.resolve()),
        "source_data_sha256": file_sha256(source_path),
        "record_count": len(retrieval_records),
        "records_path": str(records_path.resolve()),
        "embeddings_path": str(embeddings_path.resolve()),
        "vocabulary_path": str(Path(vocabulary_path).resolve()) if vocabulary_path else None,
        "vocabulary_mode": vocabulary_bundle.mode,
        "vocabulary": vocabulary_bundle.vocabulary,
        "embedding_text_fields": list(EMBEDDING_TEXT_FIELDS),
        "embedding_backend_requested": embedding_backend,
        "embedding_backend_used": backend_name,
        "embedding_backend_error": backend_error,
        "embedding": embedding_metadata.to_dict(),
    }
    _write_json(metadata_path, metadata)

    return metadata


def load_index(index_dir: str | Path = DEFAULT_INDEX_DIR) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    path = Path(index_dir)
    records_path = path / "records.jsonl"
    embeddings_path = path / "embeddings.npy"
    metadata_path = path / "metadata.json"

    with records_path.open("r", encoding="utf-8-sig") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    embeddings = np.load(embeddings_path)

    with metadata_path.open("r", encoding="utf-8-sig") as handle:
        metadata = json.load(handle)

    if len(records) != embeddings.shape[0]:
        raise RuntimeError("Index records and embeddings have different row counts.")

    return records, embeddings.astype(np.float32, copy=False), metadata
