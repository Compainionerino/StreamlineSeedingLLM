from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import app as app_module
from app.session_store import default_viewport_image_path


class ExperimentSessionTests(unittest.TestCase):
    def test_failed_experiment_omits_viewport_image_payload_for_save(self) -> None:
        image_payload = {"path": "preview.png"}

        self.assertIsNone(app_module.experiment_viewport_image_for_save(False, image_payload))

    def test_successful_experiment_keeps_viewport_image_payload_copy_for_save(self) -> None:
        image_payload = {"path": "preview.png"}

        saved_payload = app_module.experiment_viewport_image_for_save(True, image_payload)

        self.assertEqual(saved_payload, image_payload)
        self.assertIsNot(saved_payload, image_payload)

    @unittest.skipIf(app_module.GUI_IMPORT_ERROR is not None, "PySide6 is unavailable")
    def test_manual_save_does_not_copy_viewport_image_for_failed_experiment(self) -> None:
        class FakeWindow:
            def __init__(self) -> None:
                self.experiment_viewport_image = {"path": "existing.png"}
                self.experiment_viewport_image_source_path = "existing.png"
                self.label_updated = False

            def viewport_image_source_path(self) -> tuple[str, str] | None:
                raise AssertionError("Failed experiments should not inspect viewport image sources.")

            def update_experiment_viewport_image_label(self) -> None:
                self.label_updated = True

        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "failed_session.json"
            snapshot = {"experiment": {"succeeded": False, "viewport_image": {"path": "old.png"}}}
            fake_window = FakeWindow()

            app_module.MainWindow.attach_viewport_image_to_snapshot(fake_window, snapshot, session_path)

            self.assertIsNone(snapshot["experiment"]["viewport_image"])
            self.assertIsNone(fake_window.experiment_viewport_image)
            self.assertIsNone(fake_window.experiment_viewport_image_source_path)
            self.assertTrue(fake_window.label_updated)
            self.assertFalse(default_viewport_image_path(session_path).exists())


if __name__ == "__main__":
    unittest.main()
