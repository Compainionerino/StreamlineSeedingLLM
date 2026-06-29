from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class InteractiveViewportLaunch:
    program: str
    arguments: list[str]
    working_directory: str
    payload_dir: str


def prepare_interactive_viewport_launch(
    code: str,
    *,
    dataset_path: str,
    metadata: Mapping[str, Any],
    user_request: Mapping[str, Any],
    title: str = "Interactive VTK Visualization",
) -> InteractiveViewportLaunch:
    """Write a persistent child-process payload and return its launch command."""

    payload_dir = Path(tempfile.mkdtemp(prefix="streamline_vtk_interactive_"))
    code_path = payload_dir / "generated_visualization.py"
    metadata_path = payload_dir / "metadata.json"
    request_path = payload_dir / "user_request.json"

    code_path.write_text(code, encoding="utf-8")
    metadata_path.write_text(json.dumps(dict(metadata), ensure_ascii=False), encoding="utf-8")
    request_path.write_text(json.dumps(dict(user_request), ensure_ascii=False), encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    return InteractiveViewportLaunch(
        program=sys.executable,
        arguments=[
            "-m",
            "streamline_app.visualization_host",
            "--code",
            str(code_path),
            "--dataset",
            dataset_path,
            "--metadata",
            str(metadata_path),
            "--user-request",
            str(request_path),
            "--title",
            title,
        ],
        working_directory=str(project_root),
        payload_dir=str(payload_dir),
    )


def cleanup_interactive_payload(payload_dir: str | None) -> None:
    if not payload_dir:
        return
    shutil.rmtree(payload_dir, ignore_errors=True)
