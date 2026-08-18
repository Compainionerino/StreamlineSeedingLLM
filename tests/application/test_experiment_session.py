from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app as app_module
from app.llm import LLMResponse
from app.prompting import PromptBundle
from app.query import UserRequest
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
    def test_execution_attempt_resets_only_viewport_image_association(self) -> None:
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

        app_module.MainWindow.reset_experiment_viewport_image_for_execution_attempt(fake_window)

        self.assertTrue(fake_window.experiment_succeeded.checked)
        self.assertEqual(fake_window.experiment_attempts.value, 3)
        self.assertEqual(fake_window.experiment_features_recognized.value, 5)
        self.assertEqual(fake_window.experiment_feature_notes.text, "feature notes")
        self.assertTrue(fake_window.experiment_colormap_used.checked)
        self.assertTrue(fake_window.experiment_suggested_seeding_used.checked)
        self.assertEqual(fake_window.experiment_seeding_score.value, 87)
        self.assertEqual(fake_window.experiment_seeding_notes.text, "seed notes")
        self.assertIsNone(fake_window.experiment_viewport_image)
        self.assertIsNone(fake_window.experiment_viewport_image_source_path)
        self.assertIsNone(fake_window.last_preview_png)
        self.assertFalse(fake_window.preview_cleared)
        self.assertTrue(fake_window.label_updated)
        self.assertTrue(fake_window.autosaved)

    @unittest.skipIf(app_module.GUI_IMPORT_ERROR is not None, "PySide6 is unavailable")
    def test_generate_vtk_code_sends_edited_prompt_tab_text(self) -> None:
        class FakeText:
            def __init__(self, value: str) -> None:
                self.value = value

            def text(self) -> str:
                return self.value

            def toPlainText(self) -> str:
                return self.value

        class FakeSpinBox:
            def __init__(self, value: int | float) -> None:
                self._value = value

            def value(self) -> int | float:
                return self._value

        class FakeWindow:
            def __init__(self) -> None:
                self.retriever = None
                self.model_name = FakeText("fake-model")
                self.api_key = FakeText("secret")
                self.api_base = FakeText("")
                self.temperature = FakeSpinBox(0.2)
                self.max_tokens = FakeSpinBox(128)
                self.top_k = FakeSpinBox(3)
                self.prompt_text = FakeText("edited prompt from the Prompt tab")
                self.prompt_bundle = PromptBundle(
                    retrieval_query="streamlines",
                    final_prompt="original generated prompt",
                    selected_records=[],
                    rag_enabled=True,
                )
                self.retrieval_payload = {"results": []}
                self.background_result = None

            def current_request(self) -> UserRequest:
                return UserRequest(visualization_goal="Show streamlines.")

            def metadata_payload(self) -> dict:
                return {}

            def current_provider_id(self) -> str:
                return "custom"

            def rag_enabled(self) -> bool:
                return True

            def current_retrieval_payload_for_rag_state(self) -> dict:
                return self.retrieval_payload

            def current_prompt_bundle_for_rag_state(self) -> PromptBundle:
                return self.prompt_bundle

            def run_background_task(self, **kwargs) -> None:
                self.background_result = kwargs["work"](lambda _status: None)

            def apply_llm_result(self, result: dict) -> None:
                self.background_result = result

            def show_error(self, title: str, message: str) -> None:
                raise AssertionError(f"Unexpected error: {title}: {message}")

        fake_window = FakeWindow()
        llm_response = LLMResponse(
            content="import vtk",
            code="import vtk",
            model="fake-model",
        )

        with patch("app.app.generate_code", return_value=llm_response) as generate_code:
            app_module.MainWindow.generate_vtk_code(fake_window)

        sent_prompt = generate_code.call_args.args[0]
        self.assertEqual(sent_prompt, "edited prompt from the Prompt tab")
        self.assertEqual(
            fake_window.background_result["prompt_bundle"].final_prompt,
            "edited prompt from the Prompt tab",
        )


if __name__ == "__main__":
    unittest.main()
