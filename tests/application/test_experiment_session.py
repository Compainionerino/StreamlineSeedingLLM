from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app import app as app_module
from app.session_store import default_viewport_image_path


class ExperimentSessionTests(unittest.TestCase):
    def test_default_experiment_payload_starts_failed_and_without_image(self) -> None:
        payload = app_module.default_experiment_payload()

        self.assertFalse(payload["succeeded"])
        self.assertEqual(payload["attempts"], 1)
        self.assertEqual(payload["features_recognized"], 0)
        self.assertEqual(payload["feature_notes"], "")
        self.assertFalse(payload["colormap_used"])
        self.assertFalse(payload["suggested_seeding_used"])
        self.assertEqual(payload["seeding_score"], 0)
        self.assertEqual(payload["seeding_notes"], "")
        self.assertIsNone(payload["viewport_image"])

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

    @unittest.skipIf(app_module.GUI_IMPORT_ERROR is not None, "PySide6 is unavailable")
    def test_execution_attempt_resets_experiment_fields_and_preview_association(self) -> None:
        class FakeCheckbox:
            def __init__(self, checked: bool = True) -> None:
                self.checked = checked

            def setChecked(self, value: bool) -> None:
                self.checked = value

        class FakeSpinBox:
            def __init__(self, value: int = 7) -> None:
                self.value = value

            def setValue(self, value: int) -> None:
                self.value = value

        class FakeTextBox:
            def __init__(self, text: str = "notes") -> None:
                self.text = text

            def clear(self) -> None:
                self.text = ""

        class FakeWindow:
            def __init__(self) -> None:
                self.restoring_session = False
                self.experiment_succeeded = FakeCheckbox(True)
                self.experiment_attempts = FakeSpinBox(3)
                self.experiment_features_recognized = FakeSpinBox(5)
                self.experiment_feature_notes = FakeTextBox("feature notes")
                self.experiment_colormap_used = FakeCheckbox(True)
                self.experiment_suggested_seeding_used = FakeCheckbox(True)
                self.experiment_seeding_score = FakeSpinBox(87)
                self.experiment_seeding_notes = FakeTextBox("seed notes")
                self.experiment_viewport_image = {"path": "old.png"}
                self.experiment_viewport_image_source_path = "old.png"
                self.last_preview_png = "old_preview.png"
                self.preview_cleared = False
                self.label_updated = False
                self.autosaved = False

            def clear_safe_preview(self) -> None:
                self.last_preview_png = None
                self.preview_cleared = True

            def update_experiment_viewport_image_label(self) -> None:
                self.label_updated = True

            def schedule_session_autosave(self) -> None:
                self.autosaved = True

        fake_window = FakeWindow()

        app_module.MainWindow.reset_experiment_for_execution_attempt(fake_window)

        self.assertFalse(fake_window.experiment_succeeded.checked)
        self.assertEqual(fake_window.experiment_attempts.value, 1)
        self.assertEqual(fake_window.experiment_features_recognized.value, 0)
        self.assertEqual(fake_window.experiment_feature_notes.text, "")
        self.assertFalse(fake_window.experiment_colormap_used.checked)
        self.assertFalse(fake_window.experiment_suggested_seeding_used.checked)
        self.assertEqual(fake_window.experiment_seeding_score.value, 0)
        self.assertEqual(fake_window.experiment_seeding_notes.text, "")
        self.assertIsNone(fake_window.experiment_viewport_image)
        self.assertIsNone(fake_window.experiment_viewport_image_source_path)
        self.assertIsNone(fake_window.last_preview_png)
        self.assertTrue(fake_window.preview_cleared)
        self.assertTrue(fake_window.label_updated)
        self.assertTrue(fake_window.autosaved)


if __name__ == "__main__":
    unittest.main()
