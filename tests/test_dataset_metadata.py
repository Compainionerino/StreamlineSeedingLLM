from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from streamline_app.dataset_metadata import extract_dataset_metadata


VTK_AVAILABLE = importlib.util.find_spec("vtk") is not None


@unittest.skipUnless(VTK_AVAILABLE, "VTK is not installed")
class DatasetMetadataTests(unittest.TestCase):
    def test_extracts_image_data_metadata(self) -> None:
        import vtk  # type: ignore

        image = vtk.vtkImageData()
        image.SetDimensions(3, 3, 3)
        image.SetSpacing(1.0, 1.0, 1.0)

        vectors = vtk.vtkDoubleArray()
        vectors.SetName("velocity")
        vectors.SetNumberOfComponents(3)
        vectors.SetNumberOfTuples(image.GetNumberOfPoints())
        for index in range(image.GetNumberOfPoints()):
            vectors.SetTuple3(index, 1.0, 0.0, 0.0)
        image.GetPointData().SetVectors(vectors)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.vti"
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(str(path))
            writer.SetInputData(image)
            self.assertEqual(writer.Write(), 1)

            metadata = extract_dataset_metadata(path)

        payload = metadata.to_dict()
        self.assertEqual(payload["dataset_class"], "vtkImageData")
        self.assertIn("3d", payload["hints"])
        self.assertIn("volume", payload["hints"])
        self.assertIn("vector_field", payload["hints"])
        self.assertEqual(payload["active_point_vectors"], "velocity")

    def test_extracts_legacy_structured_grid_dimensions(self) -> None:
        import vtk  # type: ignore

        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(2, 2, 2)
        points = vtk.vtkPoints()
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    points.InsertNextPoint(float(x), float(y), float(z))
        grid.SetPoints(points)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "structured.vtk"
            writer = vtk.vtkStructuredGridWriter()
            writer.SetFileName(str(path))
            writer.SetInputData(grid)
            self.assertEqual(writer.Write(), 1)

            metadata = extract_dataset_metadata(path)

        payload = metadata.to_dict()
        self.assertEqual(payload["dataset_class"], "vtkStructuredGrid")
        self.assertEqual(payload["dimensions"], (2, 2, 2))
        self.assertIn("3d", payload["hints"])
        self.assertIn("volume", payload["hints"])


if __name__ == "__main__":
    unittest.main()
