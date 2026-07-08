from __future__ import annotations

from typing import Any, Mapping

from .code_validation import ensure_valid_code


class VisualizationExecutionError(RuntimeError):
    """Raised when generated visualization code cannot produce a renderer."""


def execute_visualization_code(
    code: str,
    *,
    dataset_path: str,
    metadata: Mapping[str, Any],
    user_request: Mapping[str, Any],
) -> Any:
    """Validate and run generated VTK code, returning its vtkRenderer."""

    ensure_valid_code(code)
    try:
        import vtk  # type: ignore
    except ImportError as exc:
        raise VisualizationExecutionError(
            "VTK is required to execute generated visualization code."
        ) from exc

    namespace: dict[str, Any] = {
        "__name__": "generated_vtk_visualization",
    }

    try:
        exec(code, namespace, namespace)
    except Exception as exc:
        raise VisualizationExecutionError(f"Generated code failed during import/execution: {exc}") from exc

    create_visualization = namespace.get("create_visualization")
    if not callable(create_visualization):
        raise VisualizationExecutionError("Generated code did not define a callable create_visualization.")

    try:
        renderer = create_visualization(
            dataset_path,
            dict(metadata),
            dict(user_request),
        )
    except Exception as exc:
        raise VisualizationExecutionError(f"create_visualization failed: {exc}") from exc

    if not isinstance(renderer, vtk.vtkRenderer):
        raise VisualizationExecutionError("create_visualization must return a vtkRenderer.")
    return renderer
