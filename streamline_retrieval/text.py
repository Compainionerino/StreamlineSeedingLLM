from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .loader import TAG_FIELDS


EMBEDDING_TEXT_FIELDS: tuple[str, ...] = (
    "algorithm_name",
    "algorithm_type",
    "application_goal",
    "application_context",
    "target_feature",
    "data_dimension",
    "seed_placement_strategy",
    "seed_input_information",
    "seed_density_or_number",
    "seed_spacing_or_filtering",
    "streamline_length_control",
    "integration_direction",
    "stopping_criteria",
    "parameters_reported",
    "parameter_reasoning",
    "reported_result",
    "limitations_or_notes",
)

RESULT_FIELDS: tuple[str, ...] = (
    "application_id",
    "paper_id",
    "paper_title",
    "method_role",
    "algorithm_name",
    "algorithm_type",
    "application_goal",
    "application_context",
    "target_feature",
    "data_dimension",
    "seed_placement_strategy",
    "seed_input_information",
    "seed_density_or_number",
    "seed_spacing_or_filtering",
    "streamline_length_control",
    "integration_direction",
    "stopping_criteria",
    "parameters_reported",
    "parameter_reasoning",
    "reported_result",
    "limitations_or_notes",
    "evidence",
    *TAG_FIELDS,
)

TOKEN_RE = re.compile(r"[a-z0-9_]+(?:[-+][a-z0-9_]+)*", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return normalize_whitespace(" ".join(stringify_value(item) for item in value))
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            item_text = stringify_value(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return normalize_whitespace("; ".join(parts))
    return normalize_whitespace(str(value))


def tag_text(record: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for field in TAG_FIELDS:
        tags = record.get(field, [])
        if isinstance(tags, list) and tags:
            parts.append(f"{field}: {' '.join(str(tag) for tag in tags)}")
    return " | ".join(parts)


def build_retrieval_text(record: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for field in EMBEDDING_TEXT_FIELDS:
        text = stringify_value(record.get(field))
        if text:
            parts.append(f"{field}: {text}")

    tags = tag_text(record)
    if tags:
        parts.append(f"tags: {tags}")

    return normalize_whitespace("\n".join(parts))


def project_result_record(record: Mapping[str, Any]) -> dict[str, Any]:
    projected = {field: record.get(field) for field in RESULT_FIELDS if field in record}
    projected["retrieval_text"] = build_retrieval_text(record)
    return projected


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def compact_evidence(record: Mapping[str, Any], *, max_items: int = 3) -> list[dict[str, Any]]:
    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        return []

    compact: list[dict[str, Any]] = []
    for item in evidence[:max_items]:
        if not isinstance(item, dict):
            continue
        output = {
            key: item.get(key)
            for key in ("page", "source", "quote")
            if item.get(key) is not None
        }
        if output:
            compact.append(output)
    return compact


def json_dump_line(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
