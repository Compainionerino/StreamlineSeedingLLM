from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .text import tokenize


DEFAULT_SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_HASHING_DIMENSION = 2048
DEFAULT_NGRAM_RANGE = (1, 2)


class EmbeddingBackendError(RuntimeError):
    """Raised when an embedding backend cannot be loaded or used."""


@dataclass(frozen=True)
class EmbeddingMetadata:
    backend: str
    model_name: str
    dimension: int
    normalize: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "details": self.details,
        }


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def ngrams(tokens: Sequence[str], ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE) -> list[str]:
    min_n, max_n = ngram_range
    terms: list[str] = []
    for n in range(min_n, max_n + 1):
        if n <= 0 or len(tokens) < n:
            continue
        for index in range(0, len(tokens) - n + 1):
            terms.append("_".join(tokens[index:index + n]))
    return terms


def text_terms(text: str, ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE) -> list[str]:
    return ngrams(tokenize(text), ngram_range)


def fit_document_frequencies(
    texts: Iterable[str],
    *,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
) -> tuple[dict[str, int], int]:
    df: Counter[str] = Counter()
    document_count = 0

    for text in texts:
        document_count += 1
        df.update(set(text_terms(text, ngram_range)))

    return dict(sorted(df.items())), document_count


def _idf(term: str, document_frequencies: Mapping[str, int], document_count: int) -> float:
    return math.log((1 + document_count) / (1 + document_frequencies.get(term, 0))) + 1.0


def _hash_index(term: str, dimension: int) -> int:
    digest = hashlib.sha256(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % dimension


def _hash_sign(term: str) -> float:
    digest = hashlib.sha256(("sign:" + term).encode("utf-8")).digest()
    return 1.0 if digest[0] % 2 == 0 else -1.0


def encode_hashing_tfidf(
    texts: Sequence[str],
    *,
    document_frequencies: Mapping[str, int],
    document_count: int,
    dimension: int = DEFAULT_HASHING_DIMENSION,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    normalize: bool = True,
) -> np.ndarray:
    matrix = np.zeros((len(texts), dimension), dtype=np.float32)

    for row_index, text in enumerate(texts):
        term_counts = Counter(text_terms(text, ngram_range))
        for term, count in term_counts.items():
            tf = 1.0 + math.log(count)
            value = tf * _idf(term, document_frequencies, document_count)
            matrix[row_index, _hash_index(term, dimension)] += value * _hash_sign(term)

    if normalize:
        matrix = l2_normalize(matrix)

    return matrix.astype(np.float32, copy=False)


class LocalHashingEmbeddingBackend:
    backend_name = "local-hashing"

    def __init__(
        self,
        *,
        dimension: int = DEFAULT_HASHING_DIMENSION,
        ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
        document_frequencies: Mapping[str, int] | None = None,
        document_count: int | None = None,
    ) -> None:
        self.dimension = dimension
        self.ngram_range = ngram_range
        self.document_frequencies = dict(document_frequencies or {})
        self.document_count = int(document_count or 0)

    def fit_encode(self, texts: Sequence[str]) -> tuple[np.ndarray, EmbeddingMetadata]:
        self.document_frequencies, self.document_count = fit_document_frequencies(
            texts,
            ngram_range=self.ngram_range,
        )
        embeddings = self.encode(texts)
        metadata = EmbeddingMetadata(
            backend=self.backend_name,
            model_name=f"hashing-tfidf-{self.dimension}",
            dimension=self.dimension,
            normalize=True,
            details={
                "document_count": self.document_count,
                "document_frequencies": self.document_frequencies,
                "ngram_range": list(self.ngram_range),
            },
        )
        return embeddings, metadata

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self.document_frequencies or self.document_count <= 0:
            raise EmbeddingBackendError("Local hashing backend has not been fitted.")
        return encode_hashing_tfidf(
            texts,
            document_frequencies=self.document_frequencies,
            document_count=self.document_count,
            dimension=self.dimension,
            ngram_range=self.ngram_range,
            normalize=True,
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "LocalHashingEmbeddingBackend":
        details = metadata.get("details", {})
        ngram_range_raw = details.get("ngram_range", list(DEFAULT_NGRAM_RANGE))
        return cls(
            dimension=int(metadata.get("dimension", DEFAULT_HASHING_DIMENSION)),
            ngram_range=(int(ngram_range_raw[0]), int(ngram_range_raw[1])),
            document_frequencies=details.get("document_frequencies", {}),
            document_count=details.get("document_count", 0),
        )


class SentenceTransformerEmbeddingBackend:
    backend_name = "sentence-transformers"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        local_files_only: bool = False,
        batch_size: int = 32,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        passage_prefix: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingBackendError(
                "sentence-transformers is not installed. Install it to use the BGE backend."
            ) from exc

        try:
            self.model = SentenceTransformer(
                model_name,
                local_files_only=local_files_only,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise EmbeddingBackendError(
                f"Could not load sentence-transformer model {model_name!r}: {exc}"
            ) from exc

        self.model_name = model_name
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def _encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def fit_encode(self, texts: Sequence[str]) -> tuple[np.ndarray, EmbeddingMetadata]:
        prefixed = [self.passage_prefix + text for text in texts]
        embeddings = self._encode_raw(prefixed)
        metadata = EmbeddingMetadata(
            backend=self.backend_name,
            model_name=self.model_name,
            dimension=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            normalize=True,
            details={
                "batch_size": self.batch_size,
                "build_local_files_only": self.local_files_only,
                "retrieval_local_files_only": True,
                "query_prefix": self.query_prefix,
                "passage_prefix": self.passage_prefix,
            },
        )
        return embeddings, metadata

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode_raw([self.query_prefix + query])

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "SentenceTransformerEmbeddingBackend":
        details = metadata.get("details", {})
        return cls(
            model_name=str(metadata.get("model_name", DEFAULT_SENTENCE_TRANSFORMER_MODEL)),
            local_files_only=bool(details.get("retrieval_local_files_only", True)),
            batch_size=int(details.get("batch_size", 32)),
            query_prefix=str(details.get("query_prefix", "")),
            passage_prefix=str(details.get("passage_prefix", "")),
        )


def backend_from_metadata(metadata: Mapping[str, Any]) -> LocalHashingEmbeddingBackend | SentenceTransformerEmbeddingBackend:
    backend = metadata.get("backend")
    if backend == LocalHashingEmbeddingBackend.backend_name:
        return LocalHashingEmbeddingBackend.from_metadata(metadata)
    if backend == SentenceTransformerEmbeddingBackend.backend_name:
        return SentenceTransformerEmbeddingBackend.from_metadata(metadata)
    raise EmbeddingBackendError(f"Unknown embedding backend in metadata: {backend!r}")


def metadata_json_dumps(metadata: Mapping[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
