from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from app.safe_execution import run_generated_code_safely


VTK_AVAILABLE = importlib.util.find_spec("vtk") is not None


@unittest.skipUnless(VTK_AVAILABLE, "VTK is not installed")
class SafeExecutionTests(unittest.TestCase):
    def test_runs_generated_code_in_subprocess_and_writes_png(self) -> None:
        code = """
import vtk


def create_visualization(dataset_path: str, metadata: dict, user_request: dict):
    sphere = vtk.vtkSphereSource()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    return renderer
"""

        result = run_generated_code_safely(
            code,
            dataset_path="",
            metadata={},
            user_request={},
            timeout_seconds=60,
        )

        self.assertEqual(result.returncode, 0)
        self.assertTrue(Path(result.output_png).exists())
        self.assertGreater(Path(result.output_png).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

