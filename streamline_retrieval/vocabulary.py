from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .loader import TAG_FIELDS, LoadedRecord


class VocabularyError(ValueError):
    """Raised when a tag vocabulary file is malformed."""


@dataclass(frozen=True)
class VocabularyBundle:
    vocabulary: dict[str, list[str]]
    derived_vocabulary: dict[str, list[str]]
    mode: str
    path: str | None
    unknown_tags_by_field: dict[str, list[str]]
    unused_tags_by_field: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "path": self.path,
            "vocabulary": self.vocabulary,
            "derived_vocabulary": self.derived_vocabulary,
            "unknown_tags_by_field": self.unknown_tags_by_field,
            "unused_tags_by_field": self.unused_tags_by_field,
        }


def _record_data(record: LoadedRecord | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(record, LoadedRecord):
        return record.data
    return record


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def load_vocabulary(
    path: str | Path,
    *,
    tag_fields: Sequence[str] = TAG_FIELDS,
) -> dict[str, list[str]]:
    """Load and validate a controlled tag vocabulary JSON file."""

    vocab_path = Path(path)
    if not vocab_path.exists():
        raise VocabularyError(f"Vocabulary file does not exist: {vocab_path}")

    with vocab_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise VocabularyError("Vocabulary root must be a JSON object.")

    vocabulary: dict[str, list[str]] = {}
    for field in tag_fields:
        raw_values = payload.get(field, [])
        if not isinstance(raw_values, list):
            raise VocabularyError(f"Vocabulary field must be a list: {field}")

        values: list[str] = []
        for value in raw_values:
            if not isinstance(value, str) or not value.strip():
                raise VocabularyError(
                    f"Vocabulary values must be non-empty strings in {field}."
                )
            values.append(value.strip())

        vocabulary[field] = _dedupe_keep_order(values)

    return vocabulary


def derive_vocabulary(
    records: Iterable[LoadedRecord | Mapping[str, Any]],
    *,
    tag_fields: Sequence[str] = TAG_FIELDS,
) -> dict[str, list[str]]:
    """Derive the observed tag vocabulary from loaded application records."""

    values_by_field: dict[str, set[str]] = {field: set() for field in tag_fields}

    for record in records:
        data = _record_data(record)
        for field in tag_fields:
            value = data.get(field, [])
            if not isinstance(value, list):
                continue
            for tag in value:
                if isinstance(tag, str) and tag.strip():
                    values_by_field[field].add(tag.strip())

    return {field: sorted(values) for field, values in values_by_field.items()}


def compare_vocabulary(
    vocabulary: Mapping[str, Sequence[str]],
    derived_vocabulary: Mapping[str, Sequence[str]],
    *,
    tag_fields: Sequence[str] = TAG_FIELDS,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (unknown observed tags, unused controlled tags) by tag field."""

    unknown_tags_by_field: dict[str, list[str]] = {}
    unused_tags_by_field: dict[str, list[str]] = {}

    for field in tag_fields:
        controlled = set(vocabulary.get(field, []))
        observed = set(derived_vocabulary.get(field, []))
        unknown_tags_by_field[field] = sorted(observed - controlled)
        unused_tags_by_field[field] = sorted(controlled - observed)

    return unknown_tags_by_field, unused_tags_by_field


def resolve_vocabulary(
    records: Iterable[LoadedRecord | Mapping[str, Any]],
    vocabulary_path: str | Path | None = None,
    *,
    tag_fields: Sequence[str] = TAG_FIELDS,
) -> VocabularyBundle:
    """Load a vocabulary if present, otherwise derive one from the records."""

    records_list = list(records)
    derived = derive_vocabulary(records_list, tag_fields=tag_fields)

    mode = "derived"
    resolved_path: str | None = None
    vocabulary = derived

    if vocabulary_path is not None:
        path = Path(vocabulary_path)
        resolved_path = str(path)
        if path.exists():
            vocabulary = load_vocabulary(path, tag_fields=tag_fields)
            mode = "loaded"

    unknown, unused = compare_vocabulary(
        vocabulary, derived, tag_fields=tag_fields
    )

    return VocabularyBundle(
        vocabulary=vocabulary,
        derived_vocabulary=derived,
        mode=mode,
        path=resolved_path,
        unknown_tags_by_field=unknown,
        unused_tags_by_field=unused,
    )
