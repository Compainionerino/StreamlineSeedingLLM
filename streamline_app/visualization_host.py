from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .code_execution import execute_visualization_code


GUI_IMPORT_ERROR: Exception | None = None
try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except Exception as exc:  # pragma: no cover - depends on optional GUI dependencies
    GUI_IMPORT_ERROR = exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run generated VTK code in an isolated interactive Qt/VTK host."
    )
    parser.add_argument("--code", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--user-request", required=True)
    parser.add_argument("--title", default="Interactive VTK Visualization")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=850)
    return parser


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


if GUI_IMPORT_ERROR is None:

    class VisualizationHostWindow(QMainWindow):
        def __init__(
            self,
            *,
            code: str,
            dataset_path: str,
            metadata: dict[str, Any],
            user_request: dict[str, Any],
            title: str,
            width: int,
            height: int,
        ) -> None:
            super().__init__()
            self.setWindowTitle(title)
            self.resize(width, height)

            container = QWidget()
            layout = QVBoxLayout(container)
            self.status_label = QLabel("Loading generated VTK renderer...")
            layout.addWidget(self.status_label)
            self.vtk_widget = QVTKRenderWindowInteractor(container)
            layout.addWidget(self.vtk_widget, 1)
            self.setCentralWidget(container)

            self.renderer = execute_visualization_code(
                code,
                dataset_path=dataset_path,
                metadata=metadata,
                user_request=user_request,
            )
            self.attach_renderer()

        def attach_renderer(self) -> None:
            render_window = self.vtk_widget.GetRenderWindow()
            render_window.GetRenderers().RemoveAllItems()
            render_window.AddRenderer(self.renderer)
            self.renderer.ResetCamera()
            self.vtk_widget.Initialize()
            self.status_label.setText("Interactive VTK viewport ready")
            QTimer.singleShot(0, render_window.Render)


def main(argv: list[str] | None = None) -> int:
    if GUI_IMPORT_ERROR is not None:
        print(
            "PySide6 and VTK are required for the interactive visualization host.\n"
            f"Original import error: {GUI_IMPORT_ERROR}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    args = build_parser().parse_args(argv)
    try:
        code = Path(args.code).read_text(encoding="utf-8")
        metadata = _load_json(args.metadata)
        user_request = _load_json(args.user_request)
        app = QApplication.instance() or QApplication(sys.argv[:1])
        window = VisualizationHostWindow(
            code=code,
            dataset_path=args.dataset,
            metadata=metadata,
            user_request=user_request,
            title=args.title,
            width=args.width,
            height=args.height,
        )
        window.show()
        print("[streamline-rag] interactive viewport ready", flush=True)
        return int(app.exec())
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
