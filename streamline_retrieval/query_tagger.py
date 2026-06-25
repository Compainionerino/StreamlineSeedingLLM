from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from .normalization_rules import (
    dimension_tags,
    feature_tags,
    input_tags,
    method_tags,
    normalize_text,
    task_tags,
)

from .loader import TAG_FIELDS


@dataclass(frozen=True)
class QueryTagMatch:
    field: str
    tag: str
    pattern: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class QueryTagResult:
    query: str
    tags: dict[str, list[str]]
    matches: list[QueryTagMatch]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "tags": self.tags,
            "matches": [match.to_dict() for match in self.matches],
        }


TAG_FUNCTIONS: dict[str, Callable[[str, set[str]], list[str]]] = {
    "dimension_tags": dimension_tags,
    "feature_tags": feature_tags,
    "method_tags": method_tags,
    "task_tags": task_tags,
    "input_tags": input_tags,
}


def _vocabulary_sets(vocabulary: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    return {
        field: {value for value in vocabulary.get(field, []) if isinstance(value, str)}
        for field in TAG_FIELDS
    }


def tag_query(
    query: str,
    vocabulary: Mapping[str, Sequence[str]],
) -> QueryTagResult:
    """Tag a query using the exact normalization rules used for corpus records."""

    text = normalize_text(query)
    vocabulary_sets = _vocabulary_sets(vocabulary)
    tags = {
        field: TAG_FUNCTIONS[field](text, vocabulary_sets[field])
        for field in TAG_FIELDS
    }

    return QueryTagResult(query=query, tags=tags, matches=[])
