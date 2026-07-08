from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .embeddings import (
    EmbeddingBackendError,
    LocalHashingEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    backend_from_metadata,
)
from .indexing import DEFAULT_INDEX_DIR, load_index
from .loader import TAG_FIELDS
from .query_tagger import tag_query
from .text import compact_evidence


SCORE_WEIGHTS: dict[str, float] = {
    "embedding": 0.65,
    "dimension_tags": 0.15,
    "feature_tags": 0.10,
    "task_tags": 0.05,
    "input_tags": 0.05,
}


@dataclass(frozen=True)
class Penalty:
    code: str
    amount: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "amount": self.amount, "reason": self.reason}


def _as_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str)}


def _tag_group_score(query_tags: Sequence[str], record_tags: Sequence[str]) -> float:
    if not query_tags:
        return 0.0
    shared = set(query_tags).intersection(record_tags)
    return len(shared) / len(set(query_tags))


def _tag_details(
    query_tags: Mapping[str, Sequence[str]],
    record: Mapping[str, Any],
) -> dict[str, dict[str, list[str]]]:
    details: dict[str, dict[str, list[str]]] = {}
    for field in TAG_FIELDS:
        wanted = set(query_tags.get(field, []))
        actual = _as_set(record.get(field))
        details[field] = {
            "matched": sorted(wanted & actual),
            "missing": sorted(wanted - actual),
            "record_tags": sorted(actual),
        }
    return details


def _compatibility_penalties(
    query_tags: Mapping[str, Sequence[str]],
    record: Mapping[str, Any],
) -> list[Penalty]:
    penalties: list[Penalty] = []
    dimensions = _as_set(record.get("dimension_tags"))
    tasks = _as_set(record.get("task_tags"))
    methods = _as_set(record.get("method_tags"))
    q_dimensions = set(query_tags.get("dimension_tags", []))
    q_tasks = set(query_tags.get("task_tags", []))

    if "3d" in q_dimensions and "3d" not in dimensions and "2d" in dimensions:
        penalties.append(
            Penalty("dimension_3d_vs_2d", 0.10, "Query asks for 3D but record is only tagged as 2D.")
        )
    if "2d" in q_dimensions and "2d" not in dimensions and "3d" in dimensions:
        penalties.append(
            Penalty("dimension_2d_vs_3d", 0.10, "Query asks for 2D but record is only tagged as 3D.")
        )
    if "volume" in q_dimensions and "volume" not in dimensions and "surface" in dimensions:
        penalties.append(
            Penalty("volume_vs_surface", 0.08, "Query asks for volume data but record is surface-oriented.")
        )
    if "surface" in q_dimensions and "surface" not in dimensions and "volume" in dimensions:
        penalties.append(
            Penalty("surface_vs_volume", 0.08, "Query asks for surface data but record is volume-oriented.")
        )
    if {"unsteady", "time_dependent"} & q_dimensions and not ({"unsteady", "time_dependent"} & dimensions):
        penalties.append(
            Penalty("missing_temporal_dimension", 0.05, "Query asks for unsteady/time-dependent flow.")
        )

    query_wants_seeding = bool({"generate_representative_streamlines", "cover_domain"} & q_tasks)
    record_looks_like_filtering = (
        "selection" in str(record.get("algorithm_type", "")).lower()
        or "filtering" in str(record.get("algorithm_type", "")).lower()
        or "selection" in str(record.get("algorithm_name", "")).lower()
        or "filtering" in str(record.get("algorithm_name", "")).lower()
    )
    if query_wants_seeding and record_looks_like_filtering and "interactive" not in methods:
        penalties.append(
            Penalty("selection_filtering_for_seeding_query", 0.03, "Record appears closer to selection/filtering than direct seed generation.")
        )

    if "reduce_clutter" in q_tasks and "reduce_clutter" not in tasks:
        penalties.append(
            Penalty("missing_clutter_task", 0.03, "Query asks for low clutter, but record lacks the reduce_clutter tag.")
        )

    return penalties


def _encode_query_with_backend(query: str, backend: Any) -> np.ndarray:
    if isinstance(backend, SentenceTransformerEmbeddingBackend):
        return backend.encode_query(query)
    if isinstance(backend, LocalHashingEmbeddingBackend):
        return backend.encode([query])
    raise EmbeddingBackendError("Unsupported embedding backend.")


def _encode_query(query: str, metadata: Mapping[str, Any]) -> np.ndarray:
    return _encode_query_with_backend(query, backend_from_metadata(metadata["embedding"]))


def _why_retrieved(
    *,
    score_components: Mapping[str, Any],
    tag_details: Mapping[str, Mapping[str, Sequence[str]]],
    penalties: Sequence[Penalty],
) -> list[str]:
    reasons: list[str] = []
    embedding_backend = score_components.get("embedding_backend", "embedding")
    reasons.append(
        f"{embedding_backend} similarity {score_components['embedding_score']:.3f}"
    )

    for field in ("dimension_tags", "feature_tags", "task_tags", "input_tags", "method_tags"):
        matched = tag_details.get(field, {}).get("matched", [])
        if matched:
            reasons.append(f"matched {field}: {', '.join(matched)}")

    if penalties:
        reasons.append(
            "penalties: "
            + ", ".join(f"{penalty.code} (-{penalty.amount:.2f})" for penalty in penalties)
        )

    return reasons


def _retrieve_loaded(
    query: str,
    *,
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    metadata: Mapping[str, Any],
    embedding_backend: Any | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    query_tag_result = tag_query(query, metadata["vocabulary"])
    query_tags = query_tag_result.tags
    if embedding_backend is None:
        query_embedding = _encode_query(query, metadata).reshape(1, -1).astype(np.float32)
    else:
        query_embedding = _encode_query_with_backend(query, embedding_backend).reshape(1, -1).astype(np.float32)

    if embeddings.shape[1] != query_embedding.shape[1]:
        raise RuntimeError(
            f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {embeddings.shape[1]}."
        )

    embedding_scores = embeddings @ query_embedding[0]
    ranked: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        tag_group_scores = {
            field: _tag_group_score(query_tags.get(field, []), record.get(field, []))
            for field in TAG_FIELDS
        }
        weighted_tag_score = sum(
            SCORE_WEIGHTS.get(field, 0.0) * tag_group_scores.get(field, 0.0)
            for field in TAG_FIELDS
        )
        penalties = _compatibility_penalties(query_tags, record)
        penalty_total = sum(penalty.amount for penalty in penalties)
        embedding_score = float(embedding_scores[index])
        final_score = (
            SCORE_WEIGHTS["embedding"] * embedding_score
            + weighted_tag_score
            - penalty_total
        )
        details = _tag_details(query_tags, record)
        score_components = {
            "embedding_backend": metadata["embedding_backend_used"],
            "embedding_score": embedding_score,
            "embedding_weight": SCORE_WEIGHTS["embedding"],
            "tag_group_scores": tag_group_scores,
            "weighted_tag_score": weighted_tag_score,
            "penalty_total": penalty_total,
            "final_score": final_score,
        }

        ranked.append(
            {
                "score": round(final_score, 6),
                "embedding_score": round(embedding_score, 6),
                "tag_score": round(weighted_tag_score, 6),
                "penalty_total": round(penalty_total, 6),
                "score_components": score_components,
                "tag_details": details,
                "penalties": [penalty.to_dict() for penalty in penalties],
                "why_retrieved": _why_retrieved(
                    score_components=score_components,
                    tag_details=details,
                    penalties=penalties,
                ),
                "application_id": record.get("application_id"),
                "paper_id": record.get("paper_id"),
                "paper_title": record.get("paper_title"),
                "method_role": record.get("method_role"),
                "algorithm_name": record.get("algorithm_name"),
                "algorithm_type": record.get("algorithm_type"),
                "application_goal": record.get("application_goal"),
                "application_context": record.get("application_context"),
                "target_feature": record.get("target_feature"),
                "data_dimension": record.get("data_dimension"),
                "seed_placement_strategy": record.get("seed_placement_strategy"),
                "seed_input_information": record.get("seed_input_information"),
                "seed_density_or_number": record.get("seed_density_or_number"),
                "seed_spacing_or_filtering": record.get("seed_spacing_or_filtering"),
                "streamline_length_control": record.get("streamline_length_control"),
                "integration_direction": record.get("integration_direction"),
                "stopping_criteria": record.get("stopping_criteria"),
                "parameters_reported": record.get("parameters_reported"),
                "parameter_reasoning": record.get("parameter_reasoning"),
                "reported_result": record.get("reported_result"),
                "limitations_or_notes": record.get("limitations_or_notes"),
                "evidence": compact_evidence(record),
                "dimension_tags": record.get("dimension_tags", []),
                "feature_tags": record.get("feature_tags", []),
                "method_tags": record.get("method_tags", []),
                "task_tags": record.get("task_tags", []),
                "input_tags": record.get("input_tags", []),
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    results = []
    for rank, result in enumerate(ranked[:top_k], start=1):
        result["rank"] = rank
        results.append(result)

    return {
        "query": query,
        "query_tags": query_tags,
        "query_tag_matches": [match.to_dict() for match in query_tag_result.matches],
        "index_metadata": {
            "record_count": metadata.get("record_count"),
            "embedding_backend_used": metadata.get("embedding_backend_used"),
            "embedding_model": metadata.get("embedding", {}).get("model_name"),
            "created_at_utc": metadata.get("created_at_utc"),
            "source_data_sha256": metadata.get("source_data_sha256"),
        },
        "results": results,
    }


class HybridRetriever:
    """Reusable local retriever that keeps the embedding backend loaded."""

    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR) -> None:
        self.records, self.embeddings, self.metadata = load_index(index_dir)
        self.embedding_backend = backend_from_metadata(self.metadata["embedding"])

    def retrieve(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        return _retrieve_loaded(
            query,
            records=self.records,
            embeddings=self.embeddings,
            metadata=self.metadata,
            embedding_backend=self.embedding_backend,
            top_k=top_k,
        )


def retrieve(
    query: str,
    *,
    index_dir: str = DEFAULT_INDEX_DIR,
    top_k: int = 5,
) -> dict[str, Any]:
    records, embeddings, metadata = load_index(index_dir)
    return _retrieve_loaded(
        query,
        records=records,
        embeddings=embeddings,
        metadata=metadata,
        top_k=top_k,
    )
