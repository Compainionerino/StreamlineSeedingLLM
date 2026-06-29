from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SafeExecutionResult:
    output_png: str
    stdout: str
    stderr: str
    returncode: int


class SafeExecutionError(RuntimeError):
    """Raised when isolated generated-code execution fails."""


def run_generated_code_safely(
    code: str,
    *,
    dataset_path: str,
    metadata: Mapping[str, Any],
    user_request: Mapping[str, Any],
    timeout_seconds: int = 180,
) -> SafeExecutionResult:
    """Run generated VTK code in a child process and return a PNG preview path."""

    with tempfile.TemporaryDirectory(prefix="streamline_vtk_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        code_path = temp_dir / "generated_visualization.py"
        metadata_path = temp_dir / "metadata.json"
        request_path = temp_dir / "user_request.json"
        output_png = temp_dir / "preview.png"

        code_path.write_text(code, encoding="utf-8")
        metadata_path.write_text(json.dumps(dict(metadata), ensure_ascii=False), encoding="utf-8")
        request_path.write_text(json.dumps(dict(user_request), ensure_ascii=False), encoding="utf-8")

        command = [
            sys.executable,
            "-m",
            "streamline_app.subprocess_runner",
            "--code",
            str(code_path),
            "--dataset",
            dataset_path,
            "--metadata",
            str(metadata_path),
            "--user-request",
            str(request_path),
            "--output-png",
            str(output_png),
        ]
        project_root = Path(__file__).resolve().parents[1]
        try:
            completed = subprocess.run(
                command,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise SafeExecutionError(
                f"Generated visualization timed out after {timeout_seconds} seconds in the isolated subprocess.\n\n"
                f"STDOUT:\n{exc.stdout or ''}\n\nSTDERR:\n{exc.stderr or ''}"
            ) from exc

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode != 0:
            raise SafeExecutionError(
                "Generated visualization failed in isolated subprocess.\n\n"
                f"Exit code: {completed.returncode}\n\n"
                f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )
        if not output_png.exists() or output_png.stat().st_size == 0:
            raise SafeExecutionError(
                "Generated visualization subprocess succeeded but did not produce a preview PNG.\n\n"
                f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )

        stable_handle = tempfile.NamedTemporaryFile(
            prefix="streamline_vtk_preview_",
            suffix=".png",
            delete=False,
        )
        stable_png = Path(stable_handle.name)
        stable_handle.close()
        output_png.replace(stable_png)
        return SafeExecutionResult(
            output_png=str(stable_png),
            stdout=stdout,
            stderr=stderr,
            returncode=completed.returncode,
        )
