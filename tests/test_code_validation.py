from __future__ import annotations

import unittest

from streamline_app.code_validation import validate_generated_code


VALID_CODE = """
import vtk


def create_visualization(dataset_path: str, metadata: dict, user_request: dict):
    renderer = vtk.vtkRenderer()
    actor = vtk.vtkActor()
    renderer.AddActor(actor)
    return renderer
"""


class CodeValidationTests(unittest.TestCase):
    def test_accepts_required_vtk_callable(self) -> None:
        result = validate_generated_code(VALID_CODE)

        self.assertTrue(result.ok, result.errors)

    def test_rejects_missing_create_visualization(self) -> None:
        result = validate_generated_code("import vtk\n")

        self.assertFalse(result.ok)
        self.assertTrue(any("create_visualization" in error for error in result.errors))

    def test_rejects_forbidden_imports_and_calls(self) -> None:
        result = validate_generated_code(
            """
import os
import vtk

def create_visualization(dataset_path, metadata, user_request):
    open(dataset_path)
    os.system("echo bad")
    return vtk.vtkRenderer()
"""
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("Forbidden import: os" in error for error in result.errors))
        self.assertTrue(any("Forbidden call: open()" in error for error in result.errors))
        self.assertTrue(any("system()" in error for error in result.errors))

    def test_rejects_window_and_interactor_creation(self) -> None:
        result = validate_generated_code(
            """
import vtk

def create_visualization(dataset_path, metadata, user_request):
    window = vtk.vtkRenderWindow()
    interactor = vtk.vtkRenderWindowInteractor()
    return vtk.vtkRenderer()
"""
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("vtkRenderWindow" in error for error in result.errors))
        self.assertTrue(any("vtkRenderWindowInteractor" in error for error in result.errors))

    def test_rejects_vtk_writer_write_call(self) -> None:
        result = validate_generated_code(
            """
import vtk

def create_visualization(dataset_path, metadata, user_request):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.Write()
    return vtk.vtkRenderer()
"""
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("Write()" in error for error in result.errors))

    def test_rejects_host_viewport_lifecycle_calls(self) -> None:
        result = validate_generated_code(
            """
import vtk

def create_visualization(dataset_path, metadata, user_request):
    renderer = vtk.vtkRenderer()
    renderer.Render()
    return renderer
"""
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("Render()" in error for error in result.errors))

    def test_rejects_qt_imports(self) -> None:
        result = validate_generated_code(
            """
from PySide6.QtWidgets import QApplication
import vtk

def create_visualization(dataset_path, metadata, user_request):
    return vtk.vtkRenderer()
"""
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("PySide6" in error for error in result.errors))

    def test_rejects_vtkmodules_qt_imports(self) -> None:
        result = validate_generated_code(
            """
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

def create_visualization(dataset_path, metadata, user_request):
    return vtk.vtkRenderer()
"""
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("vtkmodules.qt" in error for error in result.errors))


if __name__ == "__main__":
    unittest.main()
