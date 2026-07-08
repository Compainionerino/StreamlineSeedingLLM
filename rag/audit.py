from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .loader import REQUIRED_FIELDS, TAG_FIELDS, JsonlIssue, LoadedRecord, parse_jsonl, validate_record
from .paths import DEFAULT_VOCABULARY_PATH
from .vocabulary import VocabularyError, resolve_vocabulary


MOJIBAKE_MARKERS: tuple[str, ...] = (
    "\u00c3",
    "\u00c2",
    "\u00e2\u20ac",
    "\u00e2\u20ac\u2122",
    "\u00e2\u20ac\u0153",
    "\u00e2\u20ac\u009d",
    "\u00ce",
    "\ufffd",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter_items(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _tag_counter_items(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"tag": tag, "count": count}
        for tag, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _flatten_strings(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from _flatten_strings(item, child_path)
    elif isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _flatten_strings(item, child_path)


def _snippet(text: str, marker: str, radius: int = 48) -> str:
    index = text.find(marker)
    if index < 0:
        return text[: radius * 2].strip()
    start = max(0, index - radius)
    end = min(len(text), index + len(marker) + radius)
    return text[start:end].strip()


def _find_mojibake_examples(
    records: Iterable[LoadedRecord],
    *,
    max_examples: int = 12,
) -> tuple[int, list[dict[str, Any]]]:
    record_count = 0
    examples: list[dict[str, Any]] = []

    for record in records:
        record_has_marker = False
        for field_path, text in _flatten_strings(record.data):
            marker = next((candidate for candidate in MOJIBAKE_MARKERS if candidate in text), None)
            if marker is None:
                continue

            record_has_marker = True
            if len(examples) < max_examples:
                examples.append(
                    {
                        "application_id": record.application_id,
                        "line_number": record.line_number,
                        "field": field_path,
                        "marker": marker,
                        "snippet": _snippet(text, marker),
                    }
                )

        if record_has_marker:
            record_count += 1

    return record_count, examples


def _issue_summary(issues: Iterable[JsonlIssue]) -> dict[str, Any]:
    by_code: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    for issue in issues:
        by_code[issue.code] += 1
        by_severity[issue.severity] += 1
    return {
        "by_code": dict(sorted(by_code.items())),
        "by_severity": dict(sorted(by_severity.items())),
    }


def audit_applications(
    data_path: str | Path,
    vocabulary_path: str | Path | None = DEFAULT_VOCABULARY_PATH,
) -> dict[str, Any]:
    """Audit the structured application JSONL and tag vocabulary."""

    path = Path(data_path)
    records, load_issues = parse_jsonl(path)
    validation_issues = [
        issue for record in records for issue in validate_record(record)
    ]

    application_ids = [
        record.application_id for record in records if record.application_id is not None
    ]
    application_id_counts = Counter(application_ids)
    duplicate_ids = {
        application_id: count
        for application_id, count in sorted(application_id_counts.items())
        if count > 1
    }
    duplicate_issues = [
        JsonlIssue(
            severity="error",
            code="duplicate_application_id",
            message=f"application_id appears {count} times.",
            application_id=application_id,
            field="application_id",
        )
        for application_id, count in duplicate_ids.items()
    ]

    all_issues = [*load_issues, *validation_issues, *duplicate_issues]

    paper_ids = {
        record.data.get("paper_id")
        for record in records
        if isinstance(record.data.get("paper_id"), str)
    }
    source_output_files = {
        record.data.get("source_output_file")
        for record in records
        if isinstance(record.data.get("source_output_file"), str)
    }
    source_pdf_files = {
        record.data.get("source_pdf_filename")
        for record in records
        if isinstance(record.data.get("source_pdf_filename"), str)
    }

    tag_counts: dict[str, Counter[str]] = {field: Counter() for field in TAG_FIELDS}
    tag_field_empty_counts: Counter[str] = Counter()
    tag_field_missing_counts: Counter[str] = Counter()
    records_per_tag_field: Counter[str] = Counter()

    for record in records:
        for field in TAG_FIELDS:
            value = record.data.get(field)
            if field not in record.data:
                tag_field_missing_counts[field] += 1
                continue
            if not isinstance(value, list):
                continue
            if not value:
                tag_field_empty_counts[field] += 1
                continue
            records_per_tag_field[field] += 1
            tag_counts[field].update(tag for tag in value if isinstance(tag, str))

    method_role_counts = Counter(
        str(record.data.get("method_role"))
        for record in records
        if record.data.get("method_role") is not None
    )

    missing_required_by_field: defaultdict[str, int] = defaultdict(int)
    wrong_list_shape_by_field: defaultdict[str, int] = defaultdict(int)
    invalid_tag_value_by_field: defaultdict[str, int] = defaultdict(int)
    for issue in validation_issues:
        if issue.code == "missing_required_field" and issue.field:
            missing_required_by_field[issue.field] += 1
        elif issue.code == "field_not_list" and issue.field:
            wrong_list_shape_by_field[issue.field] += 1
        elif issue.code == "invalid_tag_value" and issue.field:
            invalid_tag_value_by_field[issue.field] += 1

    try:
        vocabulary_bundle = resolve_vocabulary(records, vocabulary_path)
        vocabulary_payload = vocabulary_bundle.to_dict()
        vocabulary_error = None
    except VocabularyError as exc:
        derived_bundle = resolve_vocabulary(records, None)
        vocabulary_payload = derived_bundle.to_dict()
        vocabulary_error = str(exc)

    mojibake_record_count, mojibake_examples = _find_mojibake_examples(records)

    return {
        "data_path": str(path.resolve()),
        "data_sha256": file_sha256(path),
        "summary": {
            "record_count": len(records),
            "unique_application_id_count": len(application_id_counts),
            "unique_paper_id_count": len(paper_ids),
            "source_output_file_count": len(source_output_files),
            "source_pdf_filename_count": len(source_pdf_files),
            "duplicate_application_id_count": len(duplicate_ids),
        },
        "schema": {
            "required_fields": list(REQUIRED_FIELDS),
            "missing_required_by_field": dict(sorted(missing_required_by_field.items())),
            "wrong_list_shape_by_field": dict(sorted(wrong_list_shape_by_field.items())),
            "invalid_tag_value_by_field": dict(sorted(invalid_tag_value_by_field.items())),
        },
        "tags": {
            "tag_fields": list(TAG_FIELDS),
            "tag_counts": {
                field: _tag_counter_items(counter)
                for field, counter in tag_counts.items()
            },
            "records_with_tags_by_field": dict(sorted(records_per_tag_field.items())),
            "empty_tag_field_counts": dict(sorted(tag_field_empty_counts.items())),
            "missing_tag_field_counts": dict(sorted(tag_field_missing_counts.items())),
        },
        "method_roles": _counter_items(method_role_counts),
        "vocabulary": {
            **vocabulary_payload,
            "error": vocabulary_error,
        },
        "text_quality": {
            "possible_mojibake_record_count": mojibake_record_count,
            "possible_mojibake_examples": mojibake_examples,
        },
        "issues": {
            "summary": _issue_summary(all_issues),
            "items": [issue.to_dict() for issue in all_issues],
        },
    }


def write_report(report: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def format_summary(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    issue_summary = report["issues"]["summary"]
    vocabulary = report["vocabulary"]
    text_quality = report["text_quality"]

    unknown_counts = {
        field: len(values)
        for field, values in vocabulary["unknown_tags_by_field"].items()
    }
    unused_counts = {
        field: len(values)
        for field, values in vocabulary["unused_tags_by_field"].items()
    }

    return "\n".join(
        [
            "Data audit complete",
            f"  Records: {summary['record_count']}",
            f"  Unique application IDs: {summary['unique_application_id_count']}",
            f"  Unique paper IDs: {summary['unique_paper_id_count']}",
            f"  Source output files: {summary['source_output_file_count']}",
            f"  Duplicate application IDs: {summary['duplicate_application_id_count']}",
            f"  Vocabulary mode: {vocabulary['mode']} ({vocabulary['path']})",
            f"  Unknown observed tags by field: {unknown_counts}",
            f"  Unused controlled tags by field: {unused_counts}",
            f"  Possible mojibake records: {text_quality['possible_mojibake_record_count']}",
            f"  Issues by severity: {issue_summary['by_severity']}",
            f"  Issues by code: {issue_summary['by_code']}",
        ]
    )
