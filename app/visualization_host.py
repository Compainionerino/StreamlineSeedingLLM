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
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
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

    class OrbitCameraInteractorStyle(vtkInteractorStyleUser):
        """Simple camera controls: orbit, zoom, and reset without actor movement."""

        def __init__(self, renderer: Any) -> None:
            super().__init__()
            self.renderer = renderer
            self._left_dragging = False
            self._right_dragging = False
            self._last_position: tuple[int, int] | None = None
            self.AddObserver("LeftButtonPressEvent", self.left_button_press)
            self.AddObserver("LeftButtonReleaseEvent", self.left_button_release)
            self.AddObserver("RightButtonPressEvent", self.right_button_press)
            self.AddObserver("RightButtonReleaseEvent", self.right_button_release)
            self.AddObserver("MouseMoveEvent", self.mouse_move)
            self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward)
            self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward)
            self.AddObserver("KeyPressEvent", self.key_press)

        def _position(self) -> tuple[int, int]:
            interactor = self.GetInteractor()
            return tuple(interactor.GetEventPosition()) if interactor is not None else (0, 0)

        def _render(self) -> None:
            interactor = self.GetInteractor()
            self.renderer.ResetCameraClippingRange()
            if interactor is not None:
                interactor.Render()

        def left_button_press(self, obj: Any, event: str) -> None:
            self._left_dragging = True
            self._last_position = self._position()

        def left_button_release(self, obj: Any, event: str) -> None:
            self._left_dragging = False
            self._last_position = None

        def right_button_press(self, obj: Any, event: str) -> None:
            self._right_dragging = True
            self._last_position = self._position()

        def right_button_release(self, obj: Any, event: str) -> None:
            self._right_dragging = False
            self._last_position = None

        def mouse_move(self, obj: Any, event: str) -> None:
            if not self._left_dragging and not self._right_dragging:
                return
            position = self._position()
            if self._last_position is None:
                self._last_position = position
                return

            dx = position[0] - self._last_position[0]
            dy = position[1] - self._last_position[1]
            camera = self.renderer.GetActiveCamera()

            if self._left_dragging:
                camera.Azimuth(-0.45 * dx)
                camera.Elevation(0.45 * dy)
                camera.OrthogonalizeViewUp()
            elif self._right_dragging:
                self._dolly(1.0 + max(-0.8, min(0.8, -0.01 * dy)))

            self._last_position = position
            self._render()

        def mouse_wheel_forward(self, obj: Any, event: str) -> None:
            self._dolly(1.15)
            self._render()

        def mouse_wheel_backward(self, obj: Any, event: str) -> None:
            self._dolly(1.0 / 1.15)
            self._render()

        def key_press(self, obj: Any, event: str) -> None:
            interactor = self.GetInteractor()
            key = interactor.GetKeySym().lower() if interactor is not None else ""
            if key == "r":
                self.renderer.ResetCamera()
                self._render()

        def _dolly(self, factor: float) -> None:
            camera = self.renderer.GetActiveCamera()
            if camera.GetParallelProjection():
                camera.SetParallelScale(camera.GetParallelScale() / factor)
            else:
                camera.Dolly(factor)


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
            self.code = code
            self.dataset_path = dataset_path
            self.metadata = metadata
            self.user_request = user_request
            self.renderer: Any | None = None
            self.vtk_widget: Any | None = None
            self.render_window: Any | None = None
            self.interactor_style: OrbitCameraInteractorStyle | None = None

            container = QWidget()
            self.layout = QVBoxLayout(container)
            self.status_label = QLabel("Loading generated VTK renderer...")
            self.layout.addWidget(self.status_label)
            self.container = container
            self.setCentralWidget(container)

        def load_renderer(self) -> None:
            try:
                print("[streamline-rag] executing generated visualization code", flush=True)
                self.renderer = execute_visualization_code(
                    self.code,
                    dataset_path=self.dataset_path,
                    metadata=self.metadata,
                    user_request=self.user_request,
                )
                print("[streamline-rag] generated renderer returned", flush=True)
                self.status_label.setText("Creating interactive VTK viewport...")
                self.vtk_widget = QVTKRenderWindowInteractor(self.container)
                self.layout.addWidget(self.vtk_widget, 1)
                QTimer.singleShot(0, self.attach_renderer)
            except Exception:
                traceback.print_exc()
                app = QApplication.instance()
                if app is not None:
                    app.exit(1)

        def attach_renderer(self) -> None:
            if self.renderer is None or self.vtk_widget is None:
                return
            print("[streamline-rag] attaching renderer to interactive viewport", flush=True)
            self.render_window = self.vtk_widget.GetRenderWindow()
            self.render_window.GetRenderers().RemoveAllItems()
            self.render_window.AddRenderer(self.renderer)
            self.renderer.ResetCamera()
            self.interactor_style = OrbitCameraInteractorStyle(self.renderer)
            self.vtk_widget.SetInteractorStyle(self.interactor_style)
            self.vtk_widget.Initialize()
            QTimer.singleShot(0, self.render_once)

        def render_once(self) -> None:
            if self.render_window is None:
                return
            print("[streamline-rag] rendering interactive viewport", flush=True)
            self.render_window.Render()
            self.status_label.setText("Ready: left drag looks around, wheel/right drag zooms, R resets.")
            print("[streamline-rag] interactive viewport ready", flush=True)


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
        print("[streamline-rag] interactive host window shown", flush=True)
        QTimer.singleShot(0, window.load_renderer)
        return int(app.exec())
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
