from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RUN_CSV_FIELDS = [
    "id",
    "dataset",
    "provider",
    "model",
    "query_mode",
    "rag_enabled",
    "succeeded",
    "first_try_success",
    "features_recognized",
    "observed_feature_coverage",
    "seeding_score",
    "attempts",
    "colormap_used",
    "suggested_seeding_used",
    "has_image",
    "duplicate_count",
    "is_primary",
    "saved_at",
    "session_rel_path",
    "image_rel_path",
    "code_rel_path",
    "feature_notes",
    "seeding_notes",
]

PRIMARY_CSV_FIELDS = RUN_CSV_FIELDS

RAG_PAIR_CSV_FIELDS = [
    "dataset",
    "provider",
    "model",
    "query_mode",
    "rag_off_success",
    "rag_on_success",
    "delta_success",
    "success_effect",
    "rag_off_first_try_success",
    "rag_on_first_try_success",
    "delta_first_try_success",
    "first_try_effect",
    "rag_off_features",
    "rag_on_features",
    "delta_features",
    "rag_off_observed_feature_coverage",
    "rag_on_observed_feature_coverage",
    "delta_observed_feature_coverage",
    "rag_off_seeding",
    "rag_on_seeding",
    "delta_seeding",
    "rag_off_attempts",
    "rag_on_attempts",
    "rag_off_session",
    "rag_on_session",
]

MODE_PAIR_CSV_FIELDS = [
    "dataset",
    "provider",
    "model",
    "rag_enabled",
    "explorative_success",
    "feature_aware_success",
    "explorative_features",
    "feature_aware_features",
    "delta_features",
    "explorative_observed_feature_coverage",
    "feature_aware_observed_feature_coverage",
    "delta_observed_feature_coverage",
    "explorative_seeding",
    "feature_aware_seeding",
    "delta_seeding",
    "explorative_attempts",
    "feature_aware_attempts",
    "explorative_session",
    "feature_aware_session",
]

GROUP_CSV_FIELDS = [
    "scope",
    "group",
    "n",
    "success_count",
    "success_rate",
    "first_try_success_count",
    "first_try_success_rate",
    "avg_features_all",
    "avg_features_successful",
    "avg_observed_feature_coverage_all",
    "avg_observed_feature_coverage_successful",
    "avg_seeding_all",
    "avg_seeding_successful",
    "avg_attempts",
    "colormap_rate",
    "suggested_seeding_rate",
    "image_count",
]


def build_experiment_report(
    session_root: str | Path,
    output_dir: str | Path,
    *,
    primary_strategy: str = "latest",
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    repo_root_path = Path(repo_root).resolve()
    session_root_path = Path(session_root)
    if not session_root_path.is_absolute():
        session_root_path = repo_root_path / session_root_path
    session_root_path = session_root_path.resolve()

    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = repo_root_path / output_dir_path
    output_dir_path = output_dir_path.resolve()

    records = load_experiment_records(session_root_path, repo_root_path, output_dir_path)
    primary_records = select_primary_records(records, strategy=primary_strategy)
    annotate_records(records, primary_records)

    report = summarize_experiments(records, primary_records, primary_strategy=primary_strategy)
    write_report_outputs(report, output_dir_path, repo_root_path)
    return report


def load_experiment_records(
    session_root: str | Path,
    repo_root: str | Path,
    dashboard_dir: str | Path,
) -> list[dict[str, Any]]:
    session_root_path = Path(session_root)
    repo_root_path = Path(repo_root)
    dashboard_dir_path = Path(dashboard_dir)

    records = []
    for path in sorted(session_root_path.rglob("*.json")):
        payload = _load_json(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("experiment"), dict):
            continue
        records.append(_record_from_session(path, payload, session_root_path, repo_root_path, dashboard_dir_path))

    return records


def select_primary_records(records: Sequence[dict[str, Any]], *, strategy: str = "latest") -> list[dict[str, Any]]:
    if strategy not in {"latest", "best"}:
        raise ValueError("primary_strategy must be either 'latest' or 'best'.")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_condition_key(record)].append(record)

    selected = []
    for grouped_records in groups.values():
        ordered = sorted(grouped_records, key=_primary_sort_key)
        if strategy == "best":
            ordered = sorted(grouped_records, key=_best_sort_key)
        selected.append(ordered[-1])

    return sorted(selected, key=_run_sort_key)


def annotate_records(records: Sequence[dict[str, Any]], primary_records: Sequence[dict[str, Any]]) -> None:
    primary_ids = {record["id"] for record in primary_records}
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for record in records:
        counts[_condition_key(record)] += 1
    for record in records:
        record["duplicate_count"] = counts[_condition_key(record)]
        record["is_primary"] = record["id"] in primary_ids

    max_features_by_dataset: dict[str, int] = defaultdict(int)
    for record in primary_records:
        max_features_by_dataset[record["dataset"]] = max(
            max_features_by_dataset[record["dataset"]],
            int(record["features_recognized"]),
        )

    for record in records:
        max_features = max_features_by_dataset.get(record["dataset"], 0)
        record["observed_feature_max"] = max_features
        record["observed_feature_coverage"] = (
            round(record["features_recognized"] / max_features, 4) if max_features else None
        )


def summarize_experiments(
    records: Sequence[dict[str, Any]],
    primary_records: Sequence[dict[str, Any]],
    *,
    primary_strategy: str,
) -> dict[str, Any]:
    records_sorted = sorted(records, key=_run_sort_key)
    primary_sorted = sorted(primary_records, key=_run_sort_key)
    rag_pairs = build_rag_pairs(primary_sorted)
    mode_pairs = build_mode_pairs(primary_sorted)
    missing_conditions = build_missing_conditions(primary_sorted)
    duplicate_conditions = build_duplicate_conditions(records_sorted)
    group_summaries = build_group_summaries(records_sorted, primary_sorted)
    rubric_rows = build_rubric_seed(primary_sorted)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    provider_models = sorted({provider_model_key(record) for record in records_sorted})
    datasets = sorted({record["dataset"] for record in records_sorted}, key=str.lower)
    query_modes = sorted({record["query_mode"] for record in records_sorted})

    return {
        "generated_at": generated_at,
        "primary_strategy": primary_strategy,
        "totals": {
            "raw_runs": len(records_sorted),
            "primary_conditions": len(primary_sorted),
            "datasets": len(datasets),
            "provider_models": len(provider_models),
            "query_modes": len(query_modes),
            "rag_pairs": len(rag_pairs),
            "mode_pairs": len(mode_pairs),
            "missing_conditions": len(missing_conditions),
            "duplicate_conditions": len(duplicate_conditions),
            "runs_without_images": sum(1 for record in records_sorted if not record["has_image"]),
        },
        "dimensions": {
            "datasets": datasets,
            "provider_models": [
                {"provider": provider, "model": model, "label": format_provider_model(provider, model)}
                for provider, model in provider_models
            ],
            "query_modes": query_modes,
            "rag_states": [False, True],
        },
        "overview": {
            "all_runs": aggregate_records(records_sorted),
            "primary_conditions": aggregate_records(primary_sorted),
        },
        "records": records_sorted,
        "primary_records": primary_sorted,
        "rag_pairs": rag_pairs,
        "mode_pairs": mode_pairs,
        "missing_conditions": missing_conditions,
        "duplicate_conditions": duplicate_conditions,
        "group_summaries": group_summaries,
        "rubric_seed": rubric_rows,
    }


def build_rag_pairs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        groups[
            (
                record["dataset"],
                record["provider"],
                record["model"],
                record["query_mode"],
            )
        ][bool(record["rag_enabled"])] = record

    pairs = []
    for (dataset, provider, model, query_mode), by_rag in sorted(groups.items(), key=lambda item: item[0]):
        if True not in by_rag or False not in by_rag:
            continue
        off = by_rag[False]
        on = by_rag[True]
        pair = {
            "dataset": dataset,
            "provider": provider,
            "model": model,
            "provider_model": format_provider_model(provider, model),
            "query_mode": query_mode,
            "rag_off": slim_record(off),
            "rag_on": slim_record(on),
            "rag_off_success": off["succeeded"],
            "rag_on_success": on["succeeded"],
            "delta_success": int(bool(on["succeeded"])) - int(bool(off["succeeded"])),
            "success_effect": _success_effect(off["succeeded"], on["succeeded"]),
            "rag_off_first_try_success": off["first_try_success"],
            "rag_on_first_try_success": on["first_try_success"],
            "delta_first_try_success": int(bool(on["first_try_success"])) - int(bool(off["first_try_success"])),
            "first_try_effect": _success_effect(off["first_try_success"], on["first_try_success"]),
            "rag_off_features": off["features_recognized"],
            "rag_on_features": on["features_recognized"],
            "delta_features": _round(on["features_recognized"] - off["features_recognized"]),
            "rag_off_observed_feature_coverage": off["observed_feature_coverage"],
            "rag_on_observed_feature_coverage": on["observed_feature_coverage"],
            "delta_observed_feature_coverage": _delta_or_none(
                on["observed_feature_coverage"],
                off["observed_feature_coverage"],
            ),
            "rag_off_seeding": off["seeding_score"],
            "rag_on_seeding": on["seeding_score"],
            "delta_seeding": _round(on["seeding_score"] - off["seeding_score"]),
            "rag_off_attempts": off["attempts"],
            "rag_on_attempts": on["attempts"],
            "rag_off_session": off["session_rel_path"],
            "rag_on_session": on["session_rel_path"],
        }
        pairs.append(pair)
    return pairs


def build_mode_pairs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, bool], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        groups[
            (
                record["dataset"],
                record["provider"],
                record["model"],
                bool(record["rag_enabled"]),
            )
        ][record["query_mode"]] = record

    pairs = []
    for (dataset, provider, model, rag_enabled), by_mode in sorted(groups.items(), key=lambda item: item[0]):
        if "explorative" not in by_mode or "feature aware" not in by_mode:
            continue
        explorative = by_mode["explorative"]
        feature_aware = by_mode["feature aware"]
        pair = {
            "dataset": dataset,
            "provider": provider,
            "model": model,
            "provider_model": format_provider_model(provider, model),
            "rag_enabled": rag_enabled,
            "explorative": slim_record(explorative),
            "feature_aware": slim_record(feature_aware),
            "explorative_success": explorative["succeeded"],
            "feature_aware_success": feature_aware["succeeded"],
            "explorative_features": explorative["features_recognized"],
            "feature_aware_features": feature_aware["features_recognized"],
            "delta_features": _round(feature_aware["features_recognized"] - explorative["features_recognized"]),
            "explorative_observed_feature_coverage": explorative["observed_feature_coverage"],
            "feature_aware_observed_feature_coverage": feature_aware["observed_feature_coverage"],
            "delta_observed_feature_coverage": _delta_or_none(
                feature_aware["observed_feature_coverage"],
                explorative["observed_feature_coverage"],
            ),
            "explorative_seeding": explorative["seeding_score"],
            "feature_aware_seeding": feature_aware["seeding_score"],
            "delta_seeding": _round(feature_aware["seeding_score"] - explorative["seeding_score"]),
            "explorative_attempts": explorative["attempts"],
            "feature_aware_attempts": feature_aware["attempts"],
            "explorative_session": explorative["session_rel_path"],
            "feature_aware_session": feature_aware["session_rel_path"],
        }
        pairs.append(pair)
    return pairs


def build_missing_conditions(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = sorted({record["dataset"] for record in records}, key=str.lower)
    provider_models = sorted({provider_model_key(record) for record in records})
    modes = sorted({record["query_mode"] for record in records})
    existing = {_condition_key(record) for record in records}

    missing = []
    for dataset, (provider, model), mode, rag_enabled in product(datasets, provider_models, modes, [False, True]):
        key = (dataset, provider, model, mode, rag_enabled)
        if key not in existing:
            missing.append(
                {
                    "dataset": dataset,
                    "provider": provider,
                    "model": model,
                    "provider_model": format_provider_model(provider, model),
                    "query_mode": mode,
                    "rag_enabled": rag_enabled,
                }
            )
    return missing


def build_duplicate_conditions(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_condition_key(record)].append(record)

    duplicates = []
    for key, grouped_records in sorted(groups.items()):
        if len(grouped_records) <= 1:
            continue
        dataset, provider, model, query_mode, rag_enabled = key
        duplicates.append(
            {
                "dataset": dataset,
                "provider": provider,
                "model": model,
                "provider_model": format_provider_model(provider, model),
                "query_mode": query_mode,
                "rag_enabled": rag_enabled,
                "count": len(grouped_records),
                "primary_session": next(
                    (record["session_rel_path"] for record in grouped_records if record.get("is_primary")),
                    grouped_records[-1]["session_rel_path"],
                ),
                "sessions": [slim_record(record) for record in sorted(grouped_records, key=_run_sort_key)],
            }
        )
    return duplicates


def build_group_summaries(
    records: Sequence[dict[str, Any]],
    primary_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    group_specs = [
        ("primary_by_rag", primary_records, ("rag_enabled",)),
        ("primary_by_mode", primary_records, ("query_mode",)),
        ("primary_by_provider", primary_records, ("provider",)),
        ("primary_by_provider_model", primary_records, ("provider", "model")),
        ("primary_by_dataset", primary_records, ("dataset",)),
        ("primary_by_dataset_rag", primary_records, ("dataset", "rag_enabled")),
        ("primary_by_dataset_mode", primary_records, ("dataset", "query_mode")),
        ("all_by_rag", records, ("rag_enabled",)),
        ("all_by_mode", records, ("query_mode",)),
        ("all_by_provider", records, ("provider",)),
        ("all_by_dataset", records, ("dataset",)),
    ]

    for scope, scoped_records, fields in group_specs:
        for group, grouped_records in _group_by(scoped_records, fields).items():
            summary = aggregate_records(grouped_records)
            summary["scope"] = scope
            summary["group"] = group
            summaries.append(summary)

    return summaries


def build_rubric_seed(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)

    rows = []
    for dataset, grouped_records in sorted(by_dataset.items(), key=lambda item: item[0].lower()):
        best_records = sorted(
            grouped_records,
            key=lambda record: (
                int(bool(record["succeeded"])),
                record["features_recognized"],
                record["seeding_score"],
                record["saved_at_sort"],
            ),
            reverse=True,
        )
        feature_lines: list[str] = []
        for record in best_records:
            for line in _note_lines(record.get("feature_notes", "")):
                normalized = line.lower()
                if normalized and normalized not in {value.lower() for value in feature_lines}:
                    feature_lines.append(line)
            if len(feature_lines) >= max(1, best_records[0]["features_recognized"]):
                break

        if not feature_lines:
            rows.append(
                {
                    "dataset": dataset,
                    "expected_feature_count": best_records[0]["features_recognized"] if best_records else 0,
                    "feature_id": "",
                    "feature_description": "",
                    "source_session": best_records[0]["session_rel_path"] if best_records else "",
                    "notes": "Fill manually before using this as a thesis rubric.",
                }
            )
            continue

        for index, feature in enumerate(feature_lines, start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "expected_feature_count": max(best_records[0]["features_recognized"], len(feature_lines)),
                    "feature_id": f"f{index:02d}",
                    "feature_description": feature,
                    "source_session": best_records[0]["session_rel_path"],
                    "notes": "Seeded from high-scoring feature notes; review manually.",
                }
            )

    return rows


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["succeeded"]]
    first_try_successful = [record for record in records if record.get("first_try_success")]
    return {
        "n": len(records),
        "success_count": len(successful),
        "success_rate": _round(len(successful) / len(records)) if records else 0.0,
        "first_try_success_count": len(first_try_successful),
        "first_try_success_rate": _round(len(first_try_successful) / len(records)) if records else 0.0,
        "avg_features_all": _average(records, "features_recognized"),
        "avg_features_successful": _average(successful, "features_recognized"),
        "avg_observed_feature_coverage_all": _average(records, "observed_feature_coverage"),
        "avg_observed_feature_coverage_successful": _average(successful, "observed_feature_coverage"),
        "avg_seeding_all": _average(records, "seeding_score"),
        "avg_seeding_successful": _average(successful, "seeding_score"),
        "avg_attempts": _average(records, "attempts"),
        "colormap_rate": _rate(records, "colormap_used"),
        "suggested_seeding_rate": _rate(records, "suggested_seeding_used"),
        "image_count": sum(1 for record in records if record["has_image"]),
    }


def slim_record(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = [
        "id",
        "dataset",
        "provider",
        "model",
        "provider_model",
        "query_mode",
        "rag_enabled",
        "succeeded",
        "first_try_success",
        "features_recognized",
        "observed_feature_coverage",
        "observed_feature_max",
        "seeding_score",
        "attempts",
        "colormap_used",
        "suggested_seeding_used",
        "has_image",
        "image_url",
        "session_url",
        "code_url",
        "session_rel_path",
        "image_rel_path",
        "code_rel_path",
        "feature_notes",
        "seeding_notes",
        "visualization_goal",
        "target_feature",
        "saved_at",
        "duplicate_count",
        "is_primary",
    ]
    return {field: record.get(field) for field in fields}


def write_report_outputs(report: Mapping[str, Any], output_dir: str | Path, repo_root: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = output_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    records = list(report["records"])
    primary_records = list(report["primary_records"])
    rag_pairs = list(report["rag_pairs"])
    mode_pairs = list(report["mode_pairs"])
    group_summaries = list(report["group_summaries"])
    rubric_rows = list(report["rubric_seed"])

    _write_json(data_dir / "summary.json", _public_report(report, include_records=False))
    _write_json(data_dir / "runs.json", records)
    _write_json(data_dir / "primary_runs.json", primary_records)
    _write_json(data_dir / "rag_pairs.json", rag_pairs)
    _write_json(data_dir / "mode_pairs.json", mode_pairs)

    _write_csv(data_dir / "runs.csv", records, RUN_CSV_FIELDS)
    _write_csv(data_dir / "primary_runs.csv", primary_records, PRIMARY_CSV_FIELDS)
    _write_csv(data_dir / "rag_pairs.csv", rag_pairs, RAG_PAIR_CSV_FIELDS)
    _write_csv(data_dir / "mode_pairs.csv", mode_pairs, MODE_PAIR_CSV_FIELDS)
    _write_csv(data_dir / "group_summaries.csv", group_summaries, GROUP_CSV_FIELDS)
    _write_csv(
        data_dir / "dataset_feature_rubric_template.csv",
        rubric_rows,
        ["dataset", "expected_feature_count", "feature_id", "feature_description", "source_session", "notes"],
    )

    dashboard_payload = _dashboard_payload(report)
    data_js = (
        "window.EXPERIMENT_DASHBOARD_DATA = "
        + json.dumps(dashboard_payload, ensure_ascii=False, indent=2)
        + ";\n"
    )
    (output_path / "dashboard-data.js").write_text(data_js, encoding="utf-8")
    (output_path / "index.html").write_text(dashboard_html(), encoding="utf-8")
    (output_path / "styles.css").write_text(dashboard_css(), encoding="utf-8")
    (output_path / "app.js").write_text(dashboard_js(), encoding="utf-8")


def provider_model_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["provider"]), str(record["model"])


def format_provider_model(provider: str, model: str) -> str:
    provider_label = {
        "anthropic": "Anthropic",
        "blablador": "Blablador",
        "gemini": "Gemini",
        "openai": "OpenAI",
    }.get(provider, provider.title() if provider else "Unknown")
    display_model = model
    provider_prefix = f"{provider}/"
    if display_model.lower().startswith(provider_prefix.lower()):
        display_model = display_model[len(provider_prefix) :]
    return f"{provider_label} / {display_model}" if display_model else provider_label


def _record_from_session(
    session_path: Path,
    payload: Mapping[str, Any],
    session_root: Path,
    repo_root: Path,
    dashboard_dir: Path,
) -> dict[str, Any]:
    experiment = _mapping(payload.get("experiment"))
    llm_settings = _mapping(payload.get("llm_settings"))
    request = _mapping(payload.get("request"))
    rag = _mapping(payload.get("rag"))
    last_artifacts = _mapping(payload.get("last_artifacts"))

    dataset = _text(Path(str(payload.get("dataset_path", ""))).stem)
    if not dataset:
        dataset = session_path.parent.name

    provider = _text(llm_settings.get("provider")) or "unknown"
    model = _text(llm_settings.get("model")) or _text(llm_settings.get("model_name")) or _model_from_filename(session_path)
    query_mode = _text(request.get("query_mode")) or _query_mode_from_filename(session_path) or "unknown"
    rag_enabled = bool(_bool(rag.get("enabled")))
    succeeded = bool(_bool(experiment.get("succeeded")))
    attempts = int(_number(experiment.get("attempts")))

    image_path = _find_image_path(experiment.get("viewport_image"), session_path, repo_root)
    code_path = _path_from_reference(last_artifacts.get("code_path"), session_path.parent, repo_root)
    session_abs_path = session_path.resolve()

    saved_at = _text(payload.get("saved_at"))
    saved_at_sort = _timestamp_sort_value(saved_at, session_path)
    record_id = _stable_id(session_path, session_root)

    image_exists = bool(image_path and image_path.exists())
    code_exists = bool(code_path and code_path.exists())
    provider_model = format_provider_model(provider, model)

    return {
        "id": record_id,
        "dataset": dataset,
        "provider": provider,
        "model": model,
        "provider_model": provider_model,
        "query_mode": query_mode,
        "rag_enabled": rag_enabled,
        "succeeded": succeeded,
        "first_try_success": succeeded and attempts == 1,
        "features_recognized": int(_number(experiment.get("features_recognized"))),
        "observed_feature_coverage": None,
        "observed_feature_max": 0,
        "seeding_score": _number(experiment.get("seeding_score")),
        "attempts": attempts,
        "colormap_used": bool(_bool(experiment.get("colormap_used"))),
        "suggested_seeding_used": bool(_bool(experiment.get("suggested_seeding_used"))),
        "has_image": image_exists,
        "duplicate_count": 1,
        "is_primary": False,
        "saved_at": saved_at,
        "saved_at_sort": saved_at_sort,
        "session_path": str(session_abs_path),
        "session_rel_path": _relative_path(session_abs_path, repo_root),
        "session_url": _relative_url(session_abs_path, dashboard_dir) if session_abs_path.exists() else "",
        "image_path": str(image_path.resolve()) if image_path and image_exists else "",
        "image_rel_path": _relative_path(image_path.resolve(), repo_root) if image_path and image_exists else "",
        "image_url": _relative_url(image_path.resolve(), dashboard_dir) if image_path and image_exists else "",
        "code_path": str(code_path.resolve()) if code_path and code_exists else "",
        "code_rel_path": _relative_path(code_path.resolve(), repo_root) if code_path and code_exists else "",
        "code_url": _relative_url(code_path.resolve(), dashboard_dir) if code_path and code_exists else "",
        "feature_notes": _text(experiment.get("feature_notes")),
        "seeding_notes": _text(experiment.get("seeding_notes")),
        "visualization_goal": _text(request.get("visualization_goal")),
        "target_feature": _text(request.get("target_feature")),
        "constraints": _text(request.get("constraints")),
        "retrieved_algorithms": _retrieved_algorithms(payload),
        "session_file": session_path.name,
    }


def _dashboard_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": report["generated_at"],
        "primary_strategy": report["primary_strategy"],
        "totals": report["totals"],
        "dimensions": report["dimensions"],
        "overview": report["overview"],
        "primary_records": [slim_record(record) for record in report["primary_records"]],
        "records": [slim_record(record) for record in report["records"]],
        "rag_pairs": report["rag_pairs"],
        "mode_pairs": report["mode_pairs"],
        "group_summaries": report["group_summaries"],
        "missing_conditions": report["missing_conditions"],
        "duplicate_conditions": report["duplicate_conditions"],
    }


def _public_report(report: Mapping[str, Any], *, include_records: bool) -> dict[str, Any]:
    excluded = set() if include_records else {"records", "primary_records", "rag_pairs", "mode_pairs"}
    return {key: value for key, value in report.items() if key not in excluded}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _group_by(records: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        label_parts = []
        for field in fields:
            value = record[field]
            if field == "rag_enabled":
                label_parts.append("RAG on" if value else "RAG off")
            else:
                label_parts.append(str(value))
        groups[" / ".join(label_parts)].append(record)
    return groups


def _condition_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["dataset"],
        record["provider"],
        record["model"],
        record["query_mode"],
        bool(record["rag_enabled"]),
    )


def _primary_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return record["saved_at_sort"], record["session_rel_path"]


def _best_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(bool(record["succeeded"])),
        record["features_recognized"],
        record["seeding_score"],
        -record["attempts"],
        record["saved_at_sort"],
        record["session_rel_path"],
    )


def _run_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(record["dataset"]).lower(),
        str(record["provider"]).lower(),
        str(record["model"]).lower(),
        str(record["query_mode"]).lower(),
        bool(record["rag_enabled"]),
        record["saved_at_sort"],
        str(record["session_rel_path"]),
    )


def _average(records: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    if not values:
        return None
    return _round(sum(values) / len(values))


def _rate(records: Sequence[Mapping[str, Any]], field: str) -> float:
    if not records:
        return 0.0
    return _round(sum(1 for record in records if record.get(field)) / len(records))


def _round(value: float | int | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def _delta_or_none(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None:
        return None
    return _round(float(left) - float(right))


def _success_effect(rag_off_success: bool, rag_on_success: bool) -> str:
    delta = int(bool(rag_on_success)) - int(bool(rag_off_success))
    if delta > 0:
        return "helped"
    if delta < 0:
        return "hurt"
    return "same"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "rag", "enabled"}
    return bool(value)


def _timestamp_sort_value(saved_at: str, session_path: Path) -> float:
    if saved_at:
        normalized = saved_at.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            pass

    match = re.search(r"_(\d{8})_(\d{6})(?:_|$)", session_path.stem)
    if match:
        try:
            return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass

    try:
        return session_path.stat().st_mtime
    except OSError:
        return 0.0


def _stable_id(session_path: Path, session_root: Path) -> str:
    return _relative_path(session_path, session_root).replace("/", "__").replace("\\", "__").removesuffix(".json")


def _relative_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_url(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _find_image_path(value: Any, session_path: Path, repo_root: Path) -> Path | None:
    image_payload = _mapping(value)
    for key in ("absolute_path", "path"):
        candidate = _path_from_reference(image_payload.get(key), session_path.parent, repo_root)
        if candidate and candidate.exists():
            return candidate

    fallback = session_path.with_name(f"{session_path.stem}_viewport.png")
    if fallback.exists():
        return fallback
    return None


def _path_from_reference(value: Any, session_dir: Path, repo_root: Path) -> Path | None:
    text = _text(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path

    session_candidate = session_dir / path
    if session_candidate.exists():
        return session_candidate

    repo_candidate = repo_root / path
    if repo_candidate.exists():
        return repo_candidate

    return repo_candidate


def _model_from_filename(path: Path) -> str:
    stem = path.stem
    known_models = [
        "claude-opus-5",
        "gpt-5.6-terra",
        "gemini-3.7-flash",
        "alias-code",
    ]
    for model in known_models:
        if model in stem:
            return model
    return "unknown"


def _query_mode_from_filename(path: Path) -> str:
    stem = path.stem
    if "_feature_aware_" in stem:
        return "feature aware"
    if "_explorative_" in stem:
        return "explorative"
    return ""


def _retrieved_algorithms(payload: Mapping[str, Any]) -> list[str]:
    retrieval = _mapping(payload.get("retrieval_payload"))
    results = retrieval.get("results")
    if not isinstance(results, list):
        return []
    algorithms = []
    for item in results[:5]:
        if isinstance(item, Mapping):
            name = _text(item.get("algorithm_name")) or _text(item.get("paper_title"))
            if name:
                algorithms.append(name)
    return algorithms


def _note_lines(notes: str) -> list[str]:
    lines = []
    for raw_line in notes.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[-*]\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines


def dashboard_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Evaluation Dashboard</title>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<main class="shell">
  <header class="topbar">
    <div>
      <p class="eyebrow">Streamline seeding RAG evaluation</p>
      <h1>Experiment Evaluation Dashboard</h1>
    </div>
    <div class="report-meta" id="reportMeta"></div>
  </header>

  <section class="metrics" id="metrics"></section>

  <section class="controls" aria-label="Filters">
    <div class="filter-control multi-filter">
      <span>Dataset</span>
      <div class="multi-select" id="datasetMulti">
        <button id="datasetTrigger" class="multi-trigger" type="button" aria-haspopup="true" aria-expanded="false">All datasets</button>
        <div id="datasetMenu" class="multi-menu"></div>
      </div>
    </div>
    <div class="filter-control multi-filter">
      <span>Model</span>
      <div class="multi-select" id="modelMulti">
        <button id="modelTrigger" class="multi-trigger" type="button" aria-haspopup="true" aria-expanded="false">All models</button>
        <div id="modelMenu" class="multi-menu"></div>
      </div>
    </div>
    <label>
      <span>Mode</span>
      <select id="modeFilter"></select>
    </label>
    <label>
      <span>RAG</span>
      <select id="ragFilter">
        <option value="all">All</option>
        <option value="true">RAG on</option>
        <option value="false">RAG off</option>
      </select>
    </label>
    <label class="search">
      <span>Search</span>
      <input id="searchFilter" type="search" autocomplete="off">
    </label>
  </section>

  <nav class="tabs" aria-label="Report sections">
    <button type="button" class="active" data-tab="rag">RAG Pairs</button>
    <button type="button" data-tab="mode">Mode Pairs</button>
    <button type="button" data-tab="matrix">Matrix</button>
    <button type="button" data-tab="runs">Runs</button>
    <button type="button" data-tab="summary">Summary</button>
  </nav>

  <section class="tab-panel active" id="tab-rag">
    <div class="section-head">
      <h2>RAG Paired Comparison</h2>
      <div class="count" id="ragCount"></div>
    </div>
    <div class="pair-list" id="ragPairs"></div>
  </section>

  <section class="tab-panel" id="tab-mode">
    <div class="section-head">
      <h2>Prompt Mode Paired Comparison</h2>
      <div class="count" id="modeCount"></div>
    </div>
    <div class="pair-list" id="modePairs"></div>
  </section>

  <section class="tab-panel" id="tab-matrix">
    <div class="section-head">
      <h2>Condition Matrix</h2>
      <div class="count" id="matrixCount"></div>
    </div>
    <div class="matrix" id="matrix"></div>
  </section>

  <section class="tab-panel" id="tab-runs">
    <div class="section-head">
      <h2>Run Table</h2>
      <div class="count" id="runsCount"></div>
    </div>
    <div class="table-wrap">
      <table id="runsTable"></table>
    </div>
  </section>

  <section class="tab-panel" id="tab-summary">
    <div class="summary-grid">
      <div>
        <h2>Grouped Metrics</h2>
        <div id="summaryBars" class="bars"></div>
      </div>
      <div>
        <h2>Data Quality</h2>
        <div id="qualityPanel" class="quality"></div>
      </div>
    </div>
  </section>
</main>
<dialog id="detailDialog">
  <button class="dialog-close" type="button" id="closeDialog" aria-label="Close">x</button>
  <div id="detailContent"></div>
</dialog>
<script src="dashboard-data.js"></script>
<script src="app.js"></script>
</body>
</html>
"""


def dashboard_css() -> str:
    return """:root {
  color-scheme: light;
  --bg: #f5f7fb;
  --panel: #ffffff;
  --panel-soft: #f8fafc;
  --text: #18202f;
  --muted: #657085;
  --border: #d7deea;
  --accent: #2563eb;
  --accent-soft: #dbeafe;
  --good: #16803c;
  --good-soft: #dcfce7;
  --warn: #9a5a00;
  --warn-soft: #fef3c7;
  --bad: #b42318;
  --bad-soft: #fee4e2;
  --ink: #111827;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

button,
input,
select {
  font: inherit;
}

.shell {
  max-width: 1480px;
  margin: 0 auto;
  padding: 24px;
}

.topbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 18px;
}

.eyebrow {
  margin: 0 0 4px;
  color: var(--muted);
  font-size: 0.84rem;
  text-transform: uppercase;
  letter-spacing: 0;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  margin-bottom: 0;
  font-size: 2rem;
  line-height: 1.12;
}

h2 {
  margin-bottom: 0;
  font-size: 1.1rem;
}

h3 {
  margin-bottom: 8px;
  font-size: 0.98rem;
}

.report-meta,
.count {
  color: var(--muted);
  font-size: 0.9rem;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}

.metric {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
}

.metric .label {
  color: var(--muted);
  font-size: 0.78rem;
}

.metric .value {
  display: block;
  margin-top: 5px;
  color: var(--ink);
  font-size: 1.35rem;
  font-weight: 700;
}

.metric .note {
  display: block;
  margin-top: 3px;
  color: var(--muted);
  font-size: 0.74rem;
}

.controls {
  display: grid;
  grid-template-columns: 1.1fr 1.4fr 1fr 0.8fr 1.4fr;
  gap: 10px;
  margin-bottom: 12px;
  padding: 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
}

label,
.filter-control {
  display: grid;
  gap: 5px;
  min-width: 0;
}

label span,
.filter-control > span {
  color: var(--muted);
  font-size: 0.78rem;
}

select,
input {
  min-width: 0;
  width: 100%;
  min-height: 36px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: #fff;
  color: var(--text);
  padding: 7px 9px;
}

.multi-filter {
  position: relative;
}

.multi-select {
  position: relative;
  min-width: 0;
}

.multi-trigger {
  position: relative;
  width: 100%;
  min-height: 36px;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: #fff;
  color: var(--text);
  padding: 7px 30px 7px 9px;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: pointer;
}

.multi-trigger::after {
  content: "v";
  position: absolute;
  right: 10px;
  top: 50%;
  color: var(--muted);
  font-size: 0.72rem;
  transform: translateY(-50%);
}

.multi-select.open .multi-trigger {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgb(37 99 235 / 0.12);
}

.multi-menu {
  display: none;
  position: absolute;
  z-index: 30;
  top: calc(100% + 4px);
  left: 0;
  right: 0;
  max-height: 320px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 18px 40px rgb(15 23 42 / 0.16);
  padding: 6px;
}

.multi-select.open .multi-menu {
  display: grid;
  gap: 2px;
}

.multi-option {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 32px;
  border-radius: 6px;
  padding: 6px 7px;
  color: var(--text);
  cursor: pointer;
}

.multi-option:hover {
  background: var(--panel-soft);
}

.multi-option.all {
  border-bottom: 1px solid var(--border);
  border-radius: 6px 6px 0 0;
  margin-bottom: 4px;
  padding-bottom: 8px;
  color: var(--accent);
  font-weight: 600;
}

.multi-option input {
  width: 16px;
  min-width: 16px;
  min-height: 16px;
  margin: 0;
  padding: 0;
}

.multi-option span {
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 0.86rem;
}

.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 12px 0;
}

.tabs button,
.run-link,
.dialog-close {
  min-height: 34px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
  cursor: pointer;
}

.tabs button {
  padding: 7px 12px;
}

.tabs button.active {
  border-color: var(--accent);
  background: var(--accent-soft);
  color: #123d8a;
}

.tab-panel {
  display: none;
}

.tab-panel.active {
  display: block;
}

.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin: 10px 0;
}

.pair-list {
  display: grid;
  gap: 12px;
}

.pair {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

.pair-header {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 12px;
  align-items: center;
  padding: 12px;
  border-bottom: 1px solid var(--border);
}

.pair-title {
  min-width: 0;
}

.pair-title strong {
  display: block;
  overflow-wrap: anywhere;
}

.pair-title span {
  color: var(--muted);
  font-size: 0.88rem;
}

.delta {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}

.badge {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  border-radius: 999px;
  padding: 3px 8px;
  color: var(--muted);
  background: var(--panel-soft);
  border: 1px solid var(--border);
  font-size: 0.8rem;
  white-space: nowrap;
}

.badge.good {
  color: var(--good);
  background: var(--good-soft);
  border-color: #bbf7d0;
}

.badge.bad {
  color: var(--bad);
  background: var(--bad-soft);
  border-color: #fecaca;
}

.badge.warn {
  color: var(--warn);
  background: var(--warn-soft);
  border-color: #fde68a;
}

.pair-body {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
}

.result {
  min-width: 0;
  padding: 12px;
}

.result + .result {
  border-left: 1px solid var(--border);
}

.result-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-height: 30px;
  margin-bottom: 8px;
}

.result-head strong {
  overflow-wrap: anywhere;
}

.thumb {
  display: grid;
  place-items: center;
  width: 100%;
  aspect-ratio: 16 / 9;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  background: #101620;
}

.thumb img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.missing {
  color: #cbd5e1;
  font-size: 0.9rem;
}

.score-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 6px;
  margin-top: 8px;
}

.score {
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 7px;
  background: var(--panel-soft);
  min-width: 0;
}

.score span {
  display: block;
  color: var(--muted);
  font-size: 0.72rem;
}

.score strong {
  display: block;
  margin-top: 2px;
  overflow-wrap: anywhere;
  font-size: 0.92rem;
}

.notes {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 0.86rem;
  line-height: 1.42;
  white-space: pre-line;
}

.run-link {
  margin-top: 8px;
  padding: 6px 9px;
  background: #fff;
}

.matrix {
  display: grid;
  gap: 12px;
}

.dataset-band {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

.dataset-band h3 {
  margin: 0;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
}

.matrix-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 8px;
  padding: 10px;
}

.mini {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  background: var(--panel-soft);
}

.mini .thumb {
  border: 0;
  border-radius: 0;
}

.mini-meta {
  padding: 7px;
  font-size: 0.76rem;
}

.mini-meta strong,
.mini-meta span {
  display: block;
  overflow-wrap: anywhere;
}

.mini-meta span {
  color: var(--muted);
}

.table-wrap {
  overflow: auto;
  max-height: 72vh;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 980px;
}

th,
td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
  font-size: 0.86rem;
}

th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #eef2f7;
  color: #344054;
}

td {
  background: #fff;
}

.summary-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 14px;
}

.bars,
.quality {
  display: grid;
  gap: 10px;
  margin-top: 10px;
}

.bar-row,
.quality-item {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px;
}

.bar-head {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 0.88rem;
}

.bar-track {
  height: 9px;
  border-radius: 999px;
  background: #e5e7eb;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #2563eb, #16a34a);
}

dialog {
  width: min(980px, calc(100vw - 32px));
  max-height: calc(100vh - 32px);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0;
}

dialog::backdrop {
  background: rgb(15 23 42 / 0.54);
}

.dialog-close {
  position: sticky;
  top: 8px;
  float: right;
  margin: 8px;
  width: 34px;
  padding: 0;
  z-index: 2;
}

#detailContent {
  padding: 16px;
}

.detail-image {
  width: 100%;
  max-height: 62vh;
  object-fit: contain;
  background: #101620;
  border-radius: 6px;
}

.empty {
  padding: 18px;
  color: var(--muted);
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
}

@media (max-width: 1100px) {
  .metrics {
    grid-template-columns: repeat(3, 1fr);
  }

  .controls {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .search {
    grid-column: 1 / -1;
  }
}

@media (max-width: 760px) {
  .shell {
    padding: 14px;
  }

  .topbar,
  .section-head,
  .pair-header,
  .summary-grid {
    display: grid;
    grid-template-columns: 1fr;
  }

  .metrics,
  .controls,
  .pair-body,
  .score-row {
    grid-template-columns: 1fr;
  }

  .result + .result {
    border-left: 0;
    border-top: 1px solid var(--border);
  }

  h1 {
    font-size: 1.55rem;
  }
}
"""


def dashboard_js() -> str:
    return """const data = window.EXPERIMENT_DASHBOARD_DATA;

const state = {
  tab: "rag",
  datasets: new Set(),
  models: new Set(),
  mode: "all",
  rag: "all",
  search: ""
};

const multiSelects = [];

const els = {
  reportMeta: document.getElementById("reportMeta"),
  metrics: document.getElementById("metrics"),
  datasetMulti: document.getElementById("datasetMulti"),
  datasetTrigger: document.getElementById("datasetTrigger"),
  datasetMenu: document.getElementById("datasetMenu"),
  modelMulti: document.getElementById("modelMulti"),
  modelTrigger: document.getElementById("modelTrigger"),
  modelMenu: document.getElementById("modelMenu"),
  modeFilter: document.getElementById("modeFilter"),
  ragFilter: document.getElementById("ragFilter"),
  searchFilter: document.getElementById("searchFilter"),
  ragPairs: document.getElementById("ragPairs"),
  modePairs: document.getElementById("modePairs"),
  matrix: document.getElementById("matrix"),
  runsTable: document.getElementById("runsTable"),
  summaryBars: document.getElementById("summaryBars"),
  qualityPanel: document.getElementById("qualityPanel"),
  ragCount: document.getElementById("ragCount"),
  modeCount: document.getElementById("modeCount"),
  matrixCount: document.getElementById("matrixCount"),
  runsCount: document.getElementById("runsCount"),
  dialog: document.getElementById("detailDialog"),
  detailContent: document.getElementById("detailContent"),
  closeDialog: document.getElementById("closeDialog")
};

function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  return String(value);
}

function signedFmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

function signedPctPoints(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const points = value * 100;
  const sign = points > 0 ? "+" : "";
  return `${sign}${points.toFixed(1)} pp`;
}

function average(values) {
  const clean = values.map(Number).filter(value => Number.isFinite(value));
  if (!clean.length) return null;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function pct(value) {
  if (value === null || value === undefined) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function deltaBadge(label, value, digits = 2) {
  const number = Number(value || 0);
  const cls = number > 0 ? "good" : number < 0 ? "bad" : "";
  const sign = number > 0 ? "+" : "";
  return `<span class="badge ${cls}">${label}: ${sign}${number.toFixed(digits)}</span>`;
}

function effectBadge(label, effect) {
  const cls = effect === "helped" ? "good" : effect === "hurt" ? "bad" : "";
  return `<span class="badge ${cls}">${escapeHtml(label)}: ${escapeHtml(effect || "same")}</span>`;
}

function successEffectBadge(effect) {
  return effectBadge("success", effect);
}

function firstTryEffectBadge(effect) {
  return effectBadge("first try", effect);
}

function boolBadge(value, label) {
  return `<span class="badge ${value ? "good" : "bad"}">${label}: ${value ? "yes" : "no"}</span>`;
}

function textMatch(record) {
  const q = state.search.trim().toLowerCase();
  if (!q) return true;
  const haystack = [
    record.dataset,
    record.provider,
    record.model,
    record.provider_model,
    record.query_mode,
    record.feature_notes,
    record.seeding_notes,
    record.target_feature,
    record.visualization_goal
  ].join(" ").toLowerCase();
  return haystack.includes(q);
}

function recordPasses(record, includeMode = true, includeRag = true) {
  if (!setPasses(state.datasets, record.dataset)) return false;
  if (!setPasses(state.models, record.provider_model)) return false;
  if (includeMode && state.mode !== "all" && record.query_mode !== state.mode) return false;
  if (includeRag && state.rag !== "all" && String(record.rag_enabled) !== state.rag) return false;
  return textMatch(record);
}

function setPasses(selected, value) {
  return selected.size === 0 || selected.has(value);
}

function pairTextMatches(...records) {
  return records.some(record => textMatch(record));
}

function ragPairPasses(pair) {
  if (!setPasses(state.datasets, pair.dataset)) return false;
  if (!setPasses(state.models, pair.provider_model)) return false;
  if (state.mode !== "all" && pair.query_mode !== state.mode) return false;
  return pairTextMatches(pair.rag_off, pair.rag_on);
}

function modePairPasses(pair) {
  if (!setPasses(state.datasets, pair.dataset)) return false;
  if (!setPasses(state.models, pair.provider_model)) return false;
  if (state.rag !== "all" && String(pair.rag_enabled) !== state.rag) return false;
  return pairTextMatches(pair.explorative, pair.feature_aware);
}

function filteredRecords() {
  return data.records.filter(record => recordPasses(record));
}

function filteredPrimaryRecords() {
  return data.primary_records.filter(record => recordPasses(record));
}

function filteredRagPairs() {
  return data.rag_pairs.filter(pair => ragPairPasses(pair));
}

function filteredModePairs() {
  return data.mode_pairs.filter(pair => modePairPasses(pair));
}

function averageField(records, field) {
  return average(records.map(record => record[field]));
}

function successRate(records) {
  if (!records.length) return null;
  return records.filter(record => record.succeeded).length / records.length;
}

function successNote(records) {
  return `${records.filter(record => record.succeeded).length} / ${records.length} succeeded`;
}

function firstTryRate(records) {
  if (!records.length) return null;
  return records.filter(record => record.first_try_success).length / records.length;
}

function firstTryNote(records) {
  return `${records.filter(record => record.first_try_success).length} / ${records.length} first try`;
}

function effectDeltaNote(pairs, field) {
  const helped = pairs.filter(pair => pair[field] === "helped").length;
  const same = pairs.filter(pair => pair[field] === "same").length;
  const hurt = pairs.filter(pair => pair[field] === "hurt").length;
  return `${helped} helped / ${same} same / ${hurt} hurt`;
}

function populateFilters() {
  const datasets = data.dimensions.datasets;
  const models = data.dimensions.provider_models.map(item => item.label);
  const modes = ["all", ...data.dimensions.query_modes];
  buildMultiSelect({
    multi: els.datasetMulti,
    trigger: els.datasetTrigger,
    menu: els.datasetMenu,
    values: datasets,
    selected: state.datasets,
    allLabel: "All datasets",
    singularLabel: "dataset",
    pluralLabel: "datasets"
  });
  buildMultiSelect({
    multi: els.modelMulti,
    trigger: els.modelTrigger,
    menu: els.modelMenu,
    values: models,
    selected: state.models,
    allLabel: "All models",
    singularLabel: "model",
    pluralLabel: "models"
  });
  fillSelect(els.modeFilter, modes, "All modes");
}

function buildMultiSelect(config) {
  multiSelects.push(config);
  config.trigger.addEventListener("click", event => {
    event.stopPropagation();
    const shouldOpen = !config.multi.classList.contains("open");
    closeMultiSelects(config);
    config.multi.classList.toggle("open", shouldOpen);
    config.trigger.setAttribute("aria-expanded", String(shouldOpen));
  });
  config.menu.addEventListener("click", event => event.stopPropagation());
  renderMultiSelect(config);
}

function renderMultiSelect(config) {
  const selected = config.selected;
  const selectedValues = [...selected];
  const allChecked = selected.size === 0;
  config.trigger.textContent = allChecked
    ? config.allLabel
    : selected.size === 1
      ? selectedValues[0]
      : `${selected.size} ${config.pluralLabel} selected`;
  config.trigger.title = allChecked ? config.allLabel : selectedValues.join(", ");
  config.menu.innerHTML = [
    optionHtml("__all__", config.allLabel, allChecked, true),
    ...config.values.map(value => optionHtml(value, value, selected.has(value), false))
  ].join("");
  config.menu.querySelectorAll("input[type='checkbox']").forEach(input => {
    input.addEventListener("change", () => {
      const value = input.dataset.value;
      if (value === "__all__") {
        selected.clear();
      } else if (input.checked) {
        selected.add(value);
      } else {
        selected.delete(value);
      }
      renderMultiSelect(config);
      renderActive();
    });
  });
}

function optionHtml(value, label, checked, isAll) {
  return `
    <label class="multi-option ${isAll ? "all" : ""}">
      <input type="checkbox" data-value="${escapeAttr(value)}" ${checked ? "checked" : ""}>
      <span>${escapeHtml(label)}</span>
    </label>
  `;
}

function closeMultiSelects(except = null) {
  for (const config of multiSelects) {
    if (config === except) continue;
    config.multi.classList.remove("open");
    config.trigger.setAttribute("aria-expanded", "false");
  }
}

function fillSelect(select, values, allLabel) {
  select.innerHTML = values.map(value => {
    const label = value === "all" ? allLabel : value;
    return `<option value="${escapeAttr(value)}">${escapeHtml(label)}</option>`;
  }).join("");
}

function renderMeta() {
  els.reportMeta.textContent = `Generated ${new Date(data.generated_at).toLocaleString()} - primary: ${data.primary_strategy}`;
}

function renderMetrics() {
  const records = filteredRecords();
  const primaryRecords = filteredPrimaryRecords();
  const ragPairs = filteredRagPairs();
  const modePairs = filteredModePairs();
  const ragDeltaSuccess = average(ragPairs.map(pair => pair.delta_success));
  const ragDeltaFirstTry = average(ragPairs.map(pair => pair.delta_first_try_success));
  const ragDeltaSeeding = average(ragPairs.map(pair => pair.delta_seeding));
  const ragDeltaCoverage = average(ragPairs.map(pair => pair.delta_observed_feature_coverage));
  const modeDeltaSeeding = average(modePairs.map(pair => pair.delta_seeding));
  const metrics = [
    {label: "Raw runs", value: records.length, note: "matching filters"},
    {label: "Success", value: pct(successRate(primaryRecords)), note: successNote(primaryRecords)},
    {label: "First-try success", value: pct(firstTryRate(primaryRecords)), note: firstTryNote(primaryRecords)},
    {label: "Avg coverage", value: pct(averageField(primaryRecords, "observed_feature_coverage")), note: "selected conditions"},
    {label: "Avg seeding", value: fmt(averageField(primaryRecords, "seeding_score")), note: "selected conditions"},
    {label: "RAG Δ success", value: signedPctPoints(ragDeltaSuccess), note: effectDeltaNote(ragPairs, "success_effect")},
    {label: "RAG Δ first-try", value: signedPctPoints(ragDeltaFirstTry), note: effectDeltaNote(ragPairs, "first_try_effect")},
    {label: "RAG Δ coverage", value: signedPctPoints(ragDeltaCoverage), note: `${ragPairs.length} paired comparisons`},
    {label: "RAG Δ seeding", value: signedFmt(ragDeltaSeeding), note: `${ragPairs.length} paired comparisons`},
    {label: "Mode Δ seeding", value: signedFmt(modeDeltaSeeding), note: `${modePairs.length} paired comparisons`}
  ];
  els.metrics.innerHTML = metrics.map(metric => `
    <div class="metric">
      <span class="label">${escapeHtml(metric.label)}</span>
      <span class="value">${escapeHtml(metric.value)}</span>
      <span class="note">${escapeHtml(metric.note)}</span>
    </div>
  `).join("");
}

function renderRagPairs() {
  const pairs = filteredRagPairs();
  els.ragCount.textContent = `${pairs.length} pairs`;
  els.ragPairs.innerHTML = pairs.length ? pairs.map(pair => pairHtml({
    title: `${pair.dataset} - ${pair.provider_model}`,
    subtitle: pair.query_mode,
    deltas: [
      successEffectBadge(pair.success_effect),
      firstTryEffectBadge(pair.first_try_effect),
      deltaBadge("features", pair.delta_features, 1),
      deltaBadge("coverage", pair.delta_observed_feature_coverage || 0, 2),
      deltaBadge("seeding", pair.delta_seeding, 1)
    ],
    leftLabel: "RAG off",
    rightLabel: "RAG on",
    left: pair.rag_off,
    right: pair.rag_on
  })).join("") : emptyHtml();
}

function renderModePairs() {
  const pairs = filteredModePairs();
  els.modeCount.textContent = `${pairs.length} pairs`;
  els.modePairs.innerHTML = pairs.length ? pairs.map(pair => pairHtml({
    title: `${pair.dataset} - ${pair.provider_model}`,
    subtitle: pair.rag_enabled ? "RAG on" : "RAG off",
    deltas: [
      deltaBadge("features", pair.delta_features, 1),
      deltaBadge("coverage", pair.delta_observed_feature_coverage || 0, 2),
      deltaBadge("seeding", pair.delta_seeding, 1)
    ],
    leftLabel: "Explorative",
    rightLabel: "Feature aware",
    left: pair.explorative,
    right: pair.feature_aware
  })).join("") : emptyHtml();
}

function pairHtml(config) {
  return `
    <article class="pair">
      <div class="pair-header">
        <div class="pair-title">
          <strong>${escapeHtml(config.title)}</strong>
          <span>${escapeHtml(config.subtitle)}</span>
        </div>
        <div class="delta">${config.deltas.join("")}</div>
      </div>
      <div class="pair-body">
        ${resultHtml(config.leftLabel, config.left)}
        ${resultHtml(config.rightLabel, config.right)}
      </div>
    </article>
  `;
}

function resultHtml(label, record) {
  return `
    <div class="result">
      <div class="result-head">
        <strong>${escapeHtml(label)}</strong>
        ${boolBadge(record.succeeded, "success")}
      </div>
      ${thumbHtml(record)}
      <div class="score-row">
        ${scoreHtml("Features", `${fmt(record.features_recognized, 0)} / ${fmt(record.observed_feature_max, 0)}`)}
        ${scoreHtml("Coverage", pct(record.observed_feature_coverage))}
        ${scoreHtml("Seeding", fmt(record.seeding_score, 1))}
        ${scoreHtml("Attempts", fmt(record.attempts, 0))}
      </div>
      <p class="notes">${escapeHtml(firstUsefulNote(record))}</p>
      <button class="run-link" type="button" data-record="${escapeAttr(record.id)}">Details</button>
    </div>
  `;
}

function thumbHtml(record) {
  if (!record.has_image || !record.image_url) {
    return `<div class="thumb"><span class="missing">No viewport image</span></div>`;
  }
  return `<button class="thumb" type="button" data-record="${escapeAttr(record.id)}"><img src="${escapeAttr(record.image_url)}" alt="${escapeAttr(record.dataset)} ${escapeAttr(record.provider_model)} ${escapeAttr(record.query_mode)}"></button>`;
}

function scoreHtml(label, value) {
  return `<div class="score"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function firstUsefulNote(record) {
  return record.feature_notes || record.seeding_notes || record.target_feature || "";
}

function renderMatrix() {
  const filtered = data.primary_records.filter(record => recordPasses(record));
  const byDataset = groupBy(filtered, record => record.dataset);
  els.matrixCount.textContent = `${filtered.length} conditions`;
  els.matrix.innerHTML = Object.keys(byDataset).sort().map(dataset => `
    <section class="dataset-band">
      <h3>${escapeHtml(dataset)}</h3>
      <div class="matrix-grid">
        ${byDataset[dataset].map(record => miniHtml(record)).join("")}
      </div>
    </section>
  `).join("") || emptyHtml();
}

function miniHtml(record) {
  return `
    <article class="mini">
      ${thumbHtml(record)}
      <div class="mini-meta">
        <strong>${escapeHtml(record.provider_model)}</strong>
        <span>${escapeHtml(record.query_mode)} - ${record.rag_enabled ? "RAG on" : "RAG off"}</span>
        <span>F ${fmt(record.features_recognized, 0)} | S ${fmt(record.seeding_score, 1)}</span>
      </div>
    </article>
  `;
}

function renderRunsTable() {
  const filtered = data.records.filter(record => recordPasses(record));
  els.runsCount.textContent = `${filtered.length} runs`;
  const header = ["Dataset", "Model", "Mode", "RAG", "Success", "Features", "Seeding", "Attempts", "Image"];
  const rows = filtered.map(record => `
    <tr>
      <td>${escapeHtml(record.dataset)}</td>
      <td>${escapeHtml(record.provider_model)}</td>
      <td>${escapeHtml(record.query_mode)}</td>
      <td>${record.rag_enabled ? "on" : "off"}</td>
      <td>${record.succeeded ? "yes" : "no"}</td>
      <td>${fmt(record.features_recognized, 0)}</td>
      <td>${fmt(record.seeding_score, 1)}</td>
      <td>${fmt(record.attempts, 0)}</td>
      <td>${record.has_image ? "yes" : "no"}</td>
    </tr>
  `).join("");
  els.runsTable.innerHTML = `<thead><tr>${header.map(value => `<th>${value}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
}

function renderSummary() {
  const scopes = new Set(["primary_by_rag", "primary_by_mode", "primary_by_provider", "primary_by_dataset"]);
  const rows = data.group_summaries.filter(row => scopes.has(row.scope));
  els.summaryBars.innerHTML = rows.map(row => `
    <div class="bar-row">
      <div class="bar-head">
        <strong>${escapeHtml(cleanScope(row.scope))}: ${escapeHtml(row.group)}</strong>
        <span>${pct(row.success_rate)} success - ${fmt(row.avg_seeding_all, 1)} seeding</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width: ${Math.max(0, Math.min(100, row.avg_seeding_all * 10))}%"></div></div>
    </div>
  `).join("");

  els.qualityPanel.innerHTML = [
    qualityItem("Missing conditions", data.totals.missing_conditions),
    qualityItem("Duplicate conditions", data.totals.duplicate_conditions),
    qualityItem("Runs without images", data.totals.runs_without_images),
    qualityItem("RAG pairs", data.totals.rag_pairs),
    qualityItem("Mode pairs", data.totals.mode_pairs)
  ].join("");
}

function cleanScope(scope) {
  return scope.replace("primary_by_", "").replaceAll("_", " ");
}

function qualityItem(label, value) {
  const cls = value === 0 ? "good" : "warn";
  return `<div class="quality-item"><span class="badge ${cls}">${escapeHtml(label)}: ${escapeHtml(value)}</span></div>`;
}

function openDetail(id) {
  const record = data.records.find(item => item.id === id) || data.primary_records.find(item => item.id === id);
  if (!record) return;
  els.detailContent.innerHTML = `
    <h2>${escapeHtml(record.dataset)} - ${escapeHtml(record.provider_model)}</h2>
    <p class="report-meta">${escapeHtml(record.query_mode)} - ${record.rag_enabled ? "RAG on" : "RAG off"} - ${escapeHtml(record.saved_at || "")}</p>
    ${record.has_image ? `<img class="detail-image" src="${escapeAttr(record.image_url)}" alt="">` : `<div class="thumb"><span class="missing">No viewport image</span></div>`}
    <div class="score-row">
      ${scoreHtml("Success", record.succeeded ? "yes" : "no")}
      ${scoreHtml("Features", `${fmt(record.features_recognized, 0)} / ${fmt(record.observed_feature_max, 0)}`)}
      ${scoreHtml("Coverage", pct(record.observed_feature_coverage))}
      ${scoreHtml("Seeding", fmt(record.seeding_score, 1))}
    </div>
    <h3>Feature Notes</h3>
    <p class="notes">${escapeHtml(record.feature_notes || "-")}</p>
    <h3>Seeding Notes</h3>
    <p class="notes">${escapeHtml(record.seeding_notes || "-")}</p>
    <h3>Target Feature</h3>
    <p class="notes">${escapeHtml(record.target_feature || "-")}</p>
    <p>${record.session_url ? `<a href="${escapeAttr(record.session_url)}">Session JSON</a>` : ""} ${record.code_url ? `<a href="${escapeAttr(record.code_url)}">Generated code</a>` : ""}</p>
  `;
  els.dialog.showModal();
}

function groupBy(items, fn) {
  return items.reduce((acc, item) => {
    const key = fn(item);
    (acc[key] ||= []).push(item);
    return acc;
  }, {});
}

function emptyHtml() {
  return `<div class="empty">No matching records.</div>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function renderActive() {
  renderMetrics();
  renderRagPairs();
  renderModePairs();
  renderMatrix();
  renderRunsTable();
  renderSummary();
  document.querySelectorAll(".run-link, .thumb, .mini").forEach(button => {
    button.addEventListener("click", event => {
      const id = event.currentTarget.getAttribute("data-record");
      if (id) openDetail(id);
    });
  });
}

function wireEvents() {
  els.modeFilter.addEventListener("change", event => {
    state.mode = event.target.value;
    renderActive();
  });
  els.ragFilter.addEventListener("change", event => {
    state.rag = event.target.value;
    renderActive();
  });
  els.searchFilter.addEventListener("input", event => {
    state.search = event.target.value;
    renderActive();
  });
  document.querySelectorAll(".tabs button").forEach(button => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach(item => item.classList.toggle("active", item === button));
      document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.remove("active"));
      document.getElementById(`tab-${button.dataset.tab}`).classList.add("active");
    });
  });
  els.closeDialog.addEventListener("click", () => els.dialog.close());
  document.addEventListener("click", () => closeMultiSelects());
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeMultiSelects();
  });
}

populateFilters();
renderMeta();
renderActive();
wireEvents();
"""
