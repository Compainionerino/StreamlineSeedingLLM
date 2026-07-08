from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from .code_execution import execute_visualization_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run generated VTK code in an isolated subprocess and write a PNG preview."
    )
    parser.add_argument("--code", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--user-request", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    return parser


def _load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def render_preview(
    *,
    code: str,
    dataset_path: str,
    metadata: dict,
    user_request: dict,
    output_png: str,
    width: int,
    height: int,
) -> None:
    import vtk  # type: ignore

    renderer = execute_visualization_code(
        code,
        dataset_path=dataset_path,
        metadata=metadata,
        user_request=user_request,
    )

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(width, height)
    render_window.AddRenderer(renderer)
    renderer.ResetCamera()
    render_window.Render()

    image_filter = vtk.vtkWindowToImageFilter()
    image_filter.SetInput(render_window)
    image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_png)
    writer.SetInputConnection(image_filter.GetOutputPort())
    writer.Write()
    output_path = Path(output_png)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"VTK failed to write preview PNG: {output_png}")

    render_window.Finalize()


def main() -> int:
    args = build_parser().parse_args()
    try:
        render_preview(
            code=Path(args.code).read_text(encoding="utf-8"),
            dataset_path=args.dataset,
            metadata=_load_json(args.metadata),
            user_request=_load_json(args.user_request),
            output_png=args.output_png,
            width=args.width,
            height=args.height,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
