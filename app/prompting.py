from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rag.text import normalize_whitespace, stringify_value

from .query import UserRequest, build_retrieval_query


PROMPT_RECORD_FIELDS: tuple[str, ...] = (
    "algorithm_name",
    "algorithm_type",
    "application_goal",
    "application_context",
    "target_feature",
    "seed_placement_strategy",
    "seed_input_information",
    "seed_density_or_number",
    "seed_spacing_or_filtering",
    "streamline_length_control",
    "integration_direction",
    "stopping_criteria",
    "parameters_reported",
    "limitations_or_notes",
)


@dataclass(frozen=True)
class PromptBundle:
    retrieval_query: str
    final_prompt: str
    selected_records: list[dict[str, Any]]
    rag_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_query": self.retrieval_query,
            "final_prompt": self.final_prompt,
            "selected_records": self.selected_records,
            "rag_enabled": self.rag_enabled,
        }


def _json_block(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def record_for_prompt(record: Mapping[str, Any]) -> dict[str, Any]:
    selected = {
        field: record.get(field)
        for field in PROMPT_RECORD_FIELDS
        if record.get(field) not in (None, "", [])
    }
    selected["retrieval_rank"] = record.get("rank")
    selected["retrieval_score"] = record.get("score")
    selected["paper_title"] = record.get("paper_title")
    selected["why_retrieved"] = record.get("why_retrieved", [])
    selected["tags"] = {
        "dimension_tags": record.get("dimension_tags", []),
        "feature_tags": record.get("feature_tags", []),
        "method_tags": record.get("method_tags", []),
        "task_tags": record.get("task_tags", []),
        "input_tags": record.get("input_tags", []),
    }
    return selected


def _record_markdown(record: Mapping[str, Any]) -> str:
    lines = [
        f"### Record {record.get('retrieval_rank')}: {record.get('algorithm_name', 'Unknown algorithm')}",
        f"- paper_title: {stringify_value(record.get('paper_title'))}",
        f"- retrieval_score: {record.get('retrieval_score')}",
        f"- why_retrieved: {stringify_value(record.get('why_retrieved'))}",
    ]
    for field in PROMPT_RECORD_FIELDS:
        if field in record:
            lines.append(f"- {field}: {stringify_value(record[field])}")

    tags = record.get("tags")
    if tags:
        lines.append(f"- tags: {stringify_value(tags)}")
    return "\n".join(lines)


def records_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return "No retrieved seeding records were provided."
    return "\n\n".join(_record_markdown(record) for record in records)


def build_final_prompt(
    *,
    user_request: UserRequest,
    dataset_metadata: Mapping[str, Any],
    retrieval_payload: Mapping[str, Any],
    rag_enabled: bool | None = None,
    top_records: int = 3,
) -> PromptBundle:
    if rag_enabled is None:
        rag_enabled = retrieval_payload.get("rag_enabled", True) is not False
    retrieval_query = build_retrieval_query(user_request, dataset_metadata)
    raw_results = [] if not rag_enabled else list(retrieval_payload.get("results", []))
    records = [
        record_for_prompt(record)
        for record in raw_results[:top_records]
        if isinstance(record, Mapping)
    ]
    rag_instruction = (
        "RAG retrieval is enabled. Use the dataset metadata, the user request, and the retrieved "
        "literature records to choose a practical seeding technique and parameterization."
    )
    if not rag_enabled:
        rag_instruction = (
            "RAG retrieval is disabled for this run. Use only the dataset metadata and the user "
            "request; do not assume that retrieved literature records are available."
        )
    rag_hard_requirements = ""
    if rag_enabled:
        rag_hard_requirements = "\n".join(
            [
                "- If an exact paper technique is too specialized, implement the closest practical VTK version and document the approximation in code comments.",
                "- At the point where seeds are created or configured, add a concise code comment naming the seeding strategy being used.",
            ]
        )
    seed_source_guidance = "Select the seed source and density from the retrieved records and the dataset metadata."
    if not rag_enabled:
        seed_source_guidance = "Select the seed source and density from the dataset metadata and the user request."

    prompt = f"""You are generating pure Python VTK code for a local PySide6 + VTK desktop application.

Your task is to implement a visualization focused on streamline or pathline seeding. {rag_instruction}

Host application contract:
- The trusted visualization host process owns a PySide6 window with a QVTKRenderWindowInteractor viewport.
- The host process owns the vtkRenderWindow, interactor, event loop, and final Render() call.
- Your code must only create VTK pipeline objects and return a vtkRenderer.
- The host process will attach the returned vtkRenderer to the QVTK viewport.

Hard requirements:
- Produce only Python code, preferably in a single fenced code block.
- Use pure VTK Python imports from vtk. Preferably just import vtk and dont try to import modules one by one. Do not use PyVista, matplotlib, pandas, network access, subprocesses, or file writes.
- Define exactly this callable:

def create_visualization(dataset_path: str, metadata: dict, user_request: dict):
    \"\"\"Return a vtkRenderer containing the complete visualization.\"\"\"

- The function must load the dataset from dataset_path using the appropriate VTK reader.
- The function must return a vtkRenderer.
- Do not create a vtkRenderWindow, vtkRenderWindowInteractor, QVTK widget, QApplication, QWidget, or event loop.
- Do not call Start(), Initialize(), Render(), show(), open(), eval(), exec(), os, subprocess, socket, requests, or urllib.
- Do not import PySide6, Qt, QVTKRenderWindowInteractor, tkinter, or any GUI toolkit.
{rag_hard_requirements}

User request:
{_json_block(user_request.to_dict())}

Dataset metadata:
{_json_block(dataset_metadata)}

Retrieval query used for the local retriever:
{retrieval_query}

Retrieved seeding records:
{records_markdown(records)}

Implementation guidance:
- {seed_source_guidance}
- Use vector arrays when available. If active vectors are missing, look for a 3-component point or cell array.
- Add visible context geometry for the dataset when useful, such as outline, surface, or volume bounds.
- Color streamlines by scalar/vector magnitude where possible.
- Keep the code self-contained inside helper functions plus create_visualization.
"""

    return PromptBundle(
        retrieval_query=normalize_whitespace(retrieval_query),
        final_prompt=prompt.strip(),
        selected_records=records,
        rag_enabled=rag_enabled,
    )
