from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from streamline_retrieval.text import normalize_whitespace


@dataclass(frozen=True)
class UserRequest:
    visualization_goal: str = ""
    target_feature: str = ""
    data_dimension: str = ""
    data_type: str = ""
    seeding_behavior: str = ""
    density_clutter_preference: str = ""
    constraints: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _clean(value: Any) -> str:
    return normalize_whitespace(str(value)) if value is not None else ""


def _metadata_context(metadata: Mapping[str, Any] | None) -> str:
    if not metadata:
        return ""

    pieces: list[str] = []
    dataset_type = _clean(metadata.get("dataset_type") or metadata.get("dataset_class"))
    if dataset_type:
        pieces.append(f"dataset_type: {dataset_type}")

    hints = metadata.get("hints")
    if isinstance(hints, list) and hints:
        pieces.append("hints: " + " ".join(_clean(item) for item in hints))

    point_count = metadata.get("point_count")
    cell_count = metadata.get("cell_count")
    if point_count is not None or cell_count is not None:
        pieces.append(f"points: {point_count}; cells: {cell_count}")

    dimensions = metadata.get("dimensions")
    if dimensions:
        pieces.append(f"dimensions: {dimensions}")

    bounds = metadata.get("bounds")
    if bounds:
        pieces.append(f"bounds: {bounds}")

    active_vectors = [
        _clean(metadata.get("active_point_vectors")),
        _clean(metadata.get("active_cell_vectors")),
    ]
    active_vectors = [value for value in active_vectors if value]
    if active_vectors:
        pieces.append("active_vectors: " + ", ".join(active_vectors))

    point_arrays = metadata.get("point_arrays")
    cell_arrays = metadata.get("cell_arrays")
    array_names: list[str] = []
    for arrays in (point_arrays, cell_arrays):
        if not isinstance(arrays, list):
            continue
        for item in arrays:
            if isinstance(item, Mapping):
                name = _clean(item.get("name"))
                components = item.get("components")
                if name:
                    array_names.append(f"{name}({components} components)")
    if array_names:
        pieces.append("available_arrays: " + ", ".join(array_names[:20]))

    return normalize_whitespace(" ".join(pieces))


def build_retrieval_query(
    user_request: UserRequest,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Build record-shaped query text for the existing hybrid retriever."""

    metadata_context = _metadata_context(metadata)
    lines = [
        f"application_goal: {_clean(user_request.visualization_goal)}",
        f"application_context: {metadata_context}",
        f"target_feature: {_clean(user_request.target_feature)}",
        f"data_dimension: {_clean(user_request.data_dimension)} {_clean(user_request.data_type)}",
        f"seed_placement_strategy: {_clean(user_request.seeding_behavior)}",
        f"seed_input_information: {metadata_context}",
        f"seed_density_or_number: {_clean(user_request.density_clutter_preference)}",
        f"seed_spacing_or_filtering: {_clean(user_request.density_clutter_preference)}",
        f"limitations_or_notes: {_clean(user_request.constraints)} {_clean(user_request.notes)}",
    ]
    return normalize_whitespace("\n".join(line for line in lines if line.split(":", 1)[1].strip()))


def infer_request_defaults_from_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    hints = metadata.get("hints") if isinstance(metadata, Mapping) else []
    hints_set = {str(item) for item in hints} if isinstance(hints, list) else set()

    data_dimension = ""
    if "3d" in hints_set:
        data_dimension = "3D"
    elif "2d" in hints_set:
        data_dimension = "2D"

    data_type = ""
    if "surface" in hints_set:
        data_type = "surface flow"
    elif "volume" in hints_set:
        data_type = "volume flow"
    elif "unstructured_grid" in hints_set:
        data_type = "unstructured grid"

    return {"data_dimension": data_dimension, "data_type": data_type}

