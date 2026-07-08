from __future__ import annotations

import importlib.util
import unittest

from app.code_execution import execute_visualization_code


VTK_AVAILABLE = importlib.util.find_spec("vtk") is not None


@unittest.skipUnless(VTK_AVAILABLE, "VTK is not installed")
class CodeExecutionTests(unittest.TestCase):
    def test_executes_renderer_callable(self) -> None:
        import vtk  # type: ignore

        code = """
import vtk


def create_visualization(dataset_path: str, metadata: dict, user_request: dict):
    renderer = vtk.vtkRenderer()
    actor = vtk.vtkActor()
    renderer.AddActor(actor)
    return renderer
"""
        renderer = execute_visualization_code(
            code,
            dataset_path="dummy.vti",
            metadata={},
            user_request={},
        )

        self.assertIsInstance(renderer, vtk.vtkRenderer)
        self.assertEqual(renderer.GetActors().GetNumberOfItems(), 1)


if __name__ == "__main__":
    unittest.main()

