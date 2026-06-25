from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


TAG_FIELDS: tuple[str, ...] = (
    "dimension_tags",
    "feature_tags",
    "method_tags",
    "task_tags",
    "input_tags",
)

REQUIRED_FIELDS: tuple[str, ...] = (
    "paper_id",
    "application_id",
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
    "reported_result",
    "limitations_or_notes",
    "evidence",
    *TAG_FIELDS,
)

LIST_FIELDS: tuple[str, ...] = (
    "stopping_criteria",
    "parameters_reported",
    "evidence",
    *TAG_FIELDS,
)


class ApplicationDataError(ValueError):
    """Raised when application data cannot be loaded in strict mode."""


@dataclass(frozen=True)
class JsonlIssue:
    severity: str
    code: str
    message: str
    line_number: int | None = None
    application_id: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class LoadedRecord:
    line_number: int
    data: dict[str, Any]

    @property
    def application_id(self) -> str | None:
        value = self.data.get("application_id")
        return value if isinstance(value, str) else None


def parse_jsonl(path: str | Path) -> tuple[list[LoadedRecord], list[JsonlIssue]]:
    """Parse JSONL records and collect non-fatal line-level issues."""

    source_path = Path(path)
    records: list[LoadedRecord] = []
    issues: list[JsonlIssue] = []

    if not source_path.exists():
        raise ApplicationDataError(f"Input data file does not exist: {source_path}")

    with source_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                issues.append(
                    JsonlIssue(
                        severity="warning",
                        code="blank_line",
                        message="Blank lines are ignored.",
                        line_number=line_number,
                    )
                )
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(
                    JsonlIssue(
                        severity="error",
                        code="invalid_json",
                        message=exc.msg,
                        line_number=line_number,
                    )
                )
                continue

            if not isinstance(payload, dict):
                issues.append(
                    JsonlIssue(
                        severity="error",
                        code="record_not_object",
                        message="Each JSONL line must contain one JSON object.",
                        line_number=line_number,
                    )
                )
                continue

            records.append(LoadedRecord(line_number=line_number, data=payload))

    return records, issues


def validate_record(
    record: LoadedRecord,
    *,
    required_fields: Sequence[str] = REQUIRED_FIELDS,
    tag_fields: Sequence[str] = TAG_FIELDS,
    list_fields: Sequence[str] = LIST_FIELDS,
) -> list[JsonlIssue]:
    """Validate the fields needed by the retrieval pipeline."""

    data = record.data
    application_id = record.application_id
    issues: list[JsonlIssue] = []

    for field in required_fields:
        if field not in data:
            issues.append(
                JsonlIssue(
                    severity="error",
                    code="missing_required_field",
                    message=f"Required field is missing: {field}",
                    line_number=record.line_number,
                    application_id=application_id,
                    field=field,
                )
            )

    id_value = data.get("application_id")
    if not isinstance(id_value, str) or not id_value.strip():
        issues.append(
            JsonlIssue(
                severity="error",
                code="invalid_application_id",
                message="application_id must be a non-empty string.",
                line_number=record.line_number,
                field="application_id",
            )
        )

    for field in list_fields:
        if field not in data:
            continue

        value = data[field]
        if not isinstance(value, list):
            issues.append(
                JsonlIssue(
                    severity="error",
                    code="field_not_list",
                    message=f"Field must be a list: {field}",
                    line_number=record.line_number,
                    application_id=application_id,
                    field=field,
                )
            )
            continue

        if field in tag_fields:
            for tag in value:
                if not isinstance(tag, str) or not tag.strip():
                    issues.append(
                        JsonlIssue(
                            severity="error",
                            code="invalid_tag_value",
                            message=f"Tag values must be non-empty strings in {field}.",
                            line_number=record.line_number,
                            application_id=application_id,
                            field=field,
                        )
                    )

    return issues


def load_applications(
    path: str | Path,
    *,
    strict: bool = True,
    required_fields: Sequence[str] = REQUIRED_FIELDS,
) -> list[LoadedRecord]:
    """Load application records from JSONL.

    In strict mode, invalid JSON, missing required fields, invalid IDs, and
    invalid tag/list shapes raise ApplicationDataError. In non-strict mode,
    parseable object records are returned and callers can inspect issues through
    parse_jsonl/validate_record directly.
    """

    records, load_issues = parse_jsonl(path)
    validation_issues = [
        issue
        for record in records
        for issue in validate_record(record, required_fields=required_fields)
    ]
    blocking = [
        issue
        for issue in (*load_issues, *validation_issues)
        if issue.severity == "error"
    ]

    if strict and blocking:
        preview = "; ".join(
            f"line {issue.line_number}: {issue.code}" for issue in blocking[:5]
        )
        suffix = "" if len(blocking) <= 5 else f"; plus {len(blocking) - 5} more"
        raise ApplicationDataError(f"Application data validation failed: {preview}{suffix}")

    return records


def load_application_dicts(path: str | Path, *, strict: bool = True) -> list[dict[str, Any]]:
    """Load application records as plain dictionaries."""

    return [record.data for record in load_applications(path, strict=strict)]


def issue_dicts(issues: Iterable[JsonlIssue]) -> list[dict[str, Any]]:
    return [issue.to_dict() for issue in issues]
