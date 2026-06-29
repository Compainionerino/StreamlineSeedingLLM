from __future__ import annotations

import json
import unittest
from pathlib import Path

from streamline_app.interactive_execution import (
    cleanup_interactive_payload,
    prepare_interactive_viewport_launch,
)


class InteractiveExecutionTests(unittest.TestCase):
    def test_prepares_child_process_payload_and_command(self) -> None:
        launch = prepare_interactive_viewport_launch(
            "import vtk\n",
            dataset_path="sample.vtk",
            metadata={"dataset_type": "vtkImageData"},
            user_request={"visualization_goal": "Show streamlines"},
            title="Test Viewport",
        )

        try:
            payload_dir = Path(launch.payload_dir)
            self.assertTrue(payload_dir.exists())
            self.assertEqual((payload_dir / "generated_visualization.py").read_text(), "import vtk\n")
            metadata = json.loads((payload_dir / "metadata.json").read_text(encoding="utf-8"))
            user_request = json.loads((payload_dir / "user_request.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset_type"], "vtkImageData")
            self.assertEqual(user_request["visualization_goal"], "Show streamlines")
            self.assertIn("streamline_app.visualization_host", launch.arguments)
            self.assertIn("--dataset", launch.arguments)
            self.assertIn("sample.vtk", launch.arguments)
            self.assertIn("--title", launch.arguments)
            self.assertIn("Test Viewport", launch.arguments)
            self.assertTrue(Path(launch.working_directory).exists())
        finally:
            cleanup_interactive_payload(launch.payload_dir)

        self.assertFalse(Path(launch.payload_dir).exists())


if __name__ == "__main__":
    unittest.main()
