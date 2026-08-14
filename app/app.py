from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rag.paths import DEFAULT_INDEX_DIR
from rag.retrieval import HybridRetriever

from .code_validation import validate_generated_code
from .dataset_metadata import extract_dataset_metadata, format_metadata_summary
from .interactive_execution import cleanup_interactive_payload, prepare_interactive_viewport_launch
from .llm import (
    BLABLADOR_API_BASE,
    DEFAULT_MAX_TOKENS,
    MAX_TOKEN_LIMIT,
    LLMSettings,
    generate_code,
    normalize_model_name,
)
from .prompting import PromptBundle, build_final_prompt
from .query import (
    DEFAULT_QUERY_MODE,
    QUERY_MODES,
    UserRequest,
    build_retrieval_query,
    infer_request_defaults_from_metadata,
)
from .run_store import RunArtifacts, write_run_artifacts
from .safe_execution import SafeExecutionResult, run_generated_code_safely
from .session_store import (
    DEFAULT_SESSION_ROOT,
    default_manual_session_path,
    default_viewport_image_path,
    load_last_session,
    load_session,
    write_last_session,
    write_manual_session,
)


GUI_IMPORT_ERROR: Exception | None = None
SAFE_PREVIEW_PLACEHOLDER = "Safe subprocess preview will appear here after validation/run."
try:
    from PySide6.QtCore import QEvent, QObject, QProcess, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QCheckBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QPlainTextEdit,
        QScrollArea,
        QSplitter,
        QSpinBox,
        QSlider,
        QDial,
        QTabBar,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QDoubleSpinBox,
        QComboBox,
    )
except Exception as exc:  # pragma: no cover - depends on optional GUI dependencies
    GUI_IMPORT_ERROR = exc


DEFAULT_LLM_PROVIDER_ID = "anthropic"
LLM_PROVIDER_OPTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "openai",
        "label": "OpenAI",
        "default_model": "openai/gpt-5.6-terra",
        "model_prefix": "openai/",
        "api_key_placeholder": "Optional OpenAI API key; OPENAI_API_KEY also works",
        "api_base_placeholder": "Optional OpenAI-compatible api_base",
    },
    {
        "id": "anthropic",
        "label": "Anthropic",
        "default_model": "anthropic/claude-opus-5",
        "model_prefix": "anthropic/",
        "api_key_placeholder": "Optional Anthropic API key; ANTHROPIC_API_KEY also works",
        "api_base_placeholder": "Optional Anthropic-compatible api_base",
    },
    {
        "id": "gemini",
        "label": "Google Gemini",
        "default_model": "gemini/gemini-3.7-flash",
        "model_prefix": "gemini/",
        "api_key_placeholder": "Optional Gemini API key; GEMINI_API_KEY or GOOGLE_API_KEY also works",
        "api_base_placeholder": "Optional Gemini-compatible api_base",
    },
    {
        "id": "blablador",
        "label": "Blablador",
        "default_model": "alias-code",
        "model_prefix": "openai/",
        "api_key_placeholder": "Optional Blablador token; BLABLADOR_API_KEY also works",
        "api_base_placeholder": f"Default: {BLABLADOR_API_BASE}",
        "always_use_default_on_switch": "true",
    },
    {
        "id": "custom",
        "label": "Custom LiteLLM",
        "default_model": "",
        "model_prefix": "",
        "api_key_placeholder": "Optional provider API key; environment variables also work",
        "api_base_placeholder": "Optional LiteLLM api_base",
    },
)
LLM_PROVIDER_BY_ID = {provider["id"]: provider for provider in LLM_PROVIDER_OPTIONS}


def _infer_provider_id_from_model(model_name: str) -> str:
    model = model_name.strip()
    for provider in LLM_PROVIDER_OPTIONS:
        prefix = provider.get("model_prefix", "")
        if prefix and model.startswith(prefix):
            return provider["id"]
    return "custom"


def _pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_experiment_payload() -> dict[str, Any]:
    return {
        "succeeded": False,
        "attempts": 1,
        "features_recognized": 0,
        "feature_notes": "",
        "colormap_used": False,
        "suggested_seeding_used": False,
        "seeding_score": 0,
        "seeding_notes": "",
        "viewport_image": None,
    }


def experiment_viewport_image_for_save(
    experiment_succeeded: bool,
    image_payload: Any,
) -> dict[str, Any] | None:
    if not experiment_succeeded or not isinstance(image_payload, dict):
        return None
    return dict(image_payload)


def _retrieval_summary(payload: dict[str, Any]) -> str:
    if payload.get("rag_enabled") is False:
        return "RAG retrieval disabled for this session.\n\nNo retrieved seeding records were used."
    lines = [
        f"Query tags: {_pretty_json(payload.get('query_tags', {}))}",
        "",
    ]
    for result in payload.get("results", []):
        lines.extend(
            [
                f"{result.get('rank')}. {result.get('score'):.3f} | {result.get('algorithm_name')}",
                f"   {result.get('paper_title')}",
                f"   tags: dim={result.get('dimension_tags')} feature={result.get('feature_tags')} method={result.get('method_tags')} task={result.get('task_tags')} input={result.get('input_tags')}",
                f"   why: {'; '.join(result.get('why_retrieved', []))}",
                "",
            ]
        )
    return "\n".join(lines)


if GUI_IMPORT_ERROR is None:

    class WheelScopedTextEdit(QTextEdit):
        def wheelEvent(self, event: Any) -> None:
            super().wheelEvent(event)
            event.accept()


    class WheelScopedPlainTextEdit(QPlainTextEdit):
        def wheelEvent(self, event: Any) -> None:
            super().wheelEvent(event)
            event.accept()


    class WheelSelectionGuard(QObject):
        def eventFilter(self, watched: Any, event: Any) -> bool:
            if event.type() == QEvent.Type.Wheel and isinstance(
                watched,
                (
                    QAbstractSpinBox,
                    QComboBox,
                    QDial,
                    QSlider,
                    QTabBar,
                ),
            ):
                event.ignore()
                return True
            return super().eventFilter(watched, event)


    class TaskSignals(QObject):
        status = Signal(str)
        finished = Signal(object)
        failed = Signal(str)


    class BackgroundTask(QRunnable):
        def __init__(self, work: Callable[[Callable[[str], None]], Any]) -> None:
            super().__init__()
            self.work = work
            self.signals = TaskSignals()

        @Slot()
        def run(self) -> None:
            try:
                result = self.work(self.signals.status.emit)
            except Exception as exc:
                self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
                return
            self.signals.finished.emit(result)


    class MainWindow(QMainWindow):
        def __init__(self, index_dir: str = DEFAULT_INDEX_DIR) -> None:
            super().__init__()
            self.setWindowTitle("Local VTK Seeding RAG")
            self.resize(1500, 920)
            self.wheel_selection_guard = WheelSelectionGuard(self)
            qt_app = QApplication.instance()
            if qt_app is not None:
                qt_app.installEventFilter(self.wheel_selection_guard)

            self.index_dir = index_dir
            self.retriever: HybridRetriever | None = None
            self.dataset_metadata: dict[str, Any] | None = None
            self.retrieval_payload: dict[str, Any] | None = None
            self.prompt_bundle: PromptBundle | None = None
            self.generated_code: str = ""
            self.llm_response: str = ""
            self.last_artifacts: RunArtifacts | None = None
            self.last_preview_png: str | None = None
            self.experiment_viewport_image: dict[str, Any] | None = None
            self.experiment_viewport_image_source_path: str | None = None
            self.visualization_process: QProcess | None = None
            self.visualization_payload_dir: str | None = None
            self.thread_pool = QThreadPool.globalInstance()
            self.active_workers: set[BackgroundTask] = set()
            self.busy = False
            self.restoring_session = False

            self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
            self.main_splitter.addWidget(self._build_left_panel())
            self.main_splitter.addWidget(self._build_viewport_panel())
            self.main_splitter.setStretchFactor(0, 0)
            self.main_splitter.setStretchFactor(1, 1)
            self.main_splitter.setSizes([560, 940])
            self.setCentralWidget(self.main_splitter)
            self._build_status_strip()
            self.session_autosave_timer = QTimer(self)
            self.session_autosave_timer.setSingleShot(True)
            self.session_autosave_timer.setInterval(1200)
            self.session_autosave_timer.timeout.connect(self.save_session)
            self._connect_session_autosave()
            if not self.restore_last_session():
                self.set_status("Ready")

        def _build_status_strip(self) -> None:
            self.busy_indicator = QProgressBar()
            self.busy_indicator.setRange(0, 0)
            self.busy_indicator.setFixedWidth(120)
            self.busy_indicator.setTextVisible(False)
            self.busy_indicator.hide()

            self.status_label = QLabel("Ready")
            self.status_label.setMinimumWidth(420)
            self.statusBar().addWidget(self.busy_indicator)
            self.statusBar().addWidget(self.status_label, 1)

        def set_status(self, message: str) -> None:
            self.status_label.setText(message)
            self.status_label.setToolTip(message)
            self.statusBar().clearMessage()

        def set_busy(self, message: str) -> None:
            self.busy = True
            self.busy_indicator.show()
            self.set_controls_enabled(False)
            self.set_status(message)

        def set_idle(self, message: str = "Ready") -> None:
            self.busy = False
            self.busy_indicator.hide()
            self.set_controls_enabled(True)
            self.set_status(message)

        def set_controls_enabled(self, enabled: bool) -> None:
            for button in (
                getattr(self, "browse_button", None),
                getattr(self, "analyze_button", None),
                getattr(self, "build_prompt_button", None),
                getattr(self, "generate_button", None),
                getattr(self, "run_button", None),
                getattr(self, "save_session_button", None),
                getattr(self, "load_session_button", None),
            ):
                if button is not None:
                    button.setEnabled(enabled)
            if getattr(self, "retrieve_button", None) is not None:
                self.retrieve_button.setEnabled(enabled and self.rag_enabled())
            if getattr(self, "stop_viewport_button", None) is not None:
                self.stop_viewport_button.setEnabled(self.viewport_process_running())

        def current_provider_id(self) -> str:
            provider_id = self.provider_combo.currentData()
            if isinstance(provider_id, str) and provider_id in LLM_PROVIDER_BY_ID:
                return provider_id
            return "custom"

        def set_provider_id(self, provider_id: str) -> None:
            index = self.provider_combo.findData(provider_id)
            if index < 0:
                index = self.provider_combo.findData("custom")
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)

        def current_query_mode(self) -> str:
            query_mode = self.query_mode_combo.currentData()
            if isinstance(query_mode, str) and query_mode in QUERY_MODES:
                return query_mode
            return DEFAULT_QUERY_MODE

        def set_query_mode(self, query_mode: str) -> None:
            index = self.query_mode_combo.findData(query_mode)
            if index < 0:
                index = self.query_mode_combo.findData(DEFAULT_QUERY_MODE)
            if index >= 0:
                self.query_mode_combo.setCurrentIndex(index)

        def rag_enabled(self) -> bool:
            return self.use_rag_checkbox.isChecked()

        def disabled_rag_payload(self, retrieval_query: str = "") -> dict[str, Any]:
            return {
                "rag_enabled": False,
                "retrieval_query": retrieval_query,
                "query_tags": {},
                "results": [],
            }

        def current_retrieval_payload_for_rag_state(self) -> dict[str, Any] | None:
            payload = self.retrieval_payload
            if not isinstance(payload, dict):
                return None
            payload_rag_enabled = payload.get("rag_enabled", True) is not False
            if payload_rag_enabled == self.rag_enabled():
                return payload
            return None

        def current_prompt_bundle_for_rag_state(self) -> PromptBundle | None:
            if self.prompt_bundle is None:
                return None
            if self.prompt_bundle.rag_enabled == self.rag_enabled():
                return self.prompt_bundle
            return None

        def handle_rag_toggled(self, *_: Any) -> None:
            if self.restoring_session:
                return
            self.prompt_bundle = None
            self.retrieval_payload = None
            if self.rag_enabled():
                self.retrieval_text.clear()
                self.set_status("RAG retrieval enabled")
            else:
                retrieval_query = build_retrieval_query(self.current_request(), self.metadata_payload())
                self.retrieval_payload = self.disabled_rag_payload(retrieval_query)
                self.retrieval_text.setPlainText(_retrieval_summary(self.retrieval_payload))
                self.set_status("RAG retrieval disabled")
            self.retrieve_button.setEnabled(not self.busy and self.rag_enabled())
            self.schedule_session_autosave()

        def current_provider_option(self) -> dict[str, str]:
            return LLM_PROVIDER_BY_ID.get(self.current_provider_id(), LLM_PROVIDER_BY_ID["custom"])

        def update_provider_placeholders(self) -> None:
            provider = self.current_provider_option()
            self.model_name.setPlaceholderText(provider.get("default_model") or "provider/model-name")
            self.api_key.setPlaceholderText(provider["api_key_placeholder"])
            self.api_base.setPlaceholderText(provider["api_base_placeholder"])

        def handle_provider_changed(self, *_: Any) -> None:
            provider = self.current_provider_option()
            previous_model = self.model_name.text().strip()
            known_defaults = {
                option["default_model"]
                for option in LLM_PROVIDER_OPTIONS
                if option.get("default_model")
            }
            known_prefixes = [
                option["model_prefix"]
                for option in LLM_PROVIDER_OPTIONS
                if option.get("model_prefix")
            ]
            should_replace_model = (
                provider["id"] != "custom"
                and (
                    provider.get("always_use_default_on_switch") == "true"
                    or not previous_model
                    or previous_model in known_defaults
                    or any(previous_model.startswith(prefix) for prefix in known_prefixes)
                )
            )
            if should_replace_model:
                self.model_name.setText(provider["default_model"])
            self.update_provider_placeholders()
            self.schedule_session_autosave()

        def _connect_session_autosave(self) -> None:
            for widget in (
                self.dataset_path,
                self.visualization_goal,
                self.target_feature,
                self.data_dimension,
                self.data_type,
                self.seeding_behavior,
                self.density_preference,
                self.constraints,
                self.notes,
                self.model_name,
                self.api_base,
                self.prompt_text,
                self.code_text,
                self.experiment_feature_notes,
                self.experiment_seeding_notes,
            ):
                widget.textChanged.connect(self.schedule_session_autosave)
            for widget in (
                self.temperature,
                self.max_tokens,
                self.top_k,
                self.experiment_attempts,
                self.experiment_features_recognized,
                self.experiment_seeding_score,
            ):
                widget.valueChanged.connect(self.schedule_session_autosave)
            self.query_mode_combo.currentIndexChanged.connect(self.schedule_session_autosave)
            self.use_rag_checkbox.stateChanged.connect(self.handle_rag_toggled)
            self.experiment_colormap_used.stateChanged.connect(self.schedule_session_autosave)
            self.experiment_succeeded.stateChanged.connect(self.handle_experiment_succeeded_changed)
            self.experiment_suggested_seeding_used.stateChanged.connect(self.schedule_session_autosave)
            self.main_splitter.splitterMoved.connect(self.schedule_session_autosave)
            self.workflow_tabs.currentChanged.connect(self.schedule_session_autosave)
            self.viewport_tabs.currentChanged.connect(self.schedule_session_autosave)

        def schedule_session_autosave(self, *_: Any) -> None:
            if self.restoring_session:
                return
            self.session_autosave_timer.start()

        def handle_experiment_succeeded_changed(self, *_: Any) -> None:
            self.update_experiment_viewport_image_label()
            self.schedule_session_autosave()

        def reset_experiment_viewport_image_for_execution_attempt(self) -> None:
            self.experiment_viewport_image = None
            self.experiment_viewport_image_source_path = None
            self.last_preview_png = None
            self.update_experiment_viewport_image_label()
            self.schedule_session_autosave()

        def experiment_snapshot(self) -> dict[str, Any]:
            succeeded = self.experiment_succeeded.isChecked()
            payload: dict[str, Any] = {
                "succeeded": succeeded,
                "attempts": int(self.experiment_attempts.value()),
                "features_recognized": int(self.experiment_features_recognized.value()),
                "feature_notes": self.experiment_feature_notes.toPlainText(),
                "colormap_used": self.experiment_colormap_used.isChecked(),
                "suggested_seeding_used": self.experiment_suggested_seeding_used.isChecked(),
                "seeding_score": int(self.experiment_seeding_score.value()),
                "seeding_notes": self.experiment_seeding_notes.toPlainText(),
                "viewport_image": None,
            }
            image_payload = experiment_viewport_image_for_save(succeeded, self.experiment_viewport_image)
            if image_payload is not None:
                payload["viewport_image"] = image_payload
            return payload

        def restore_experiment_payload(
            self,
            payload: Any,
            session_path: str | Path | None = None,
        ) -> None:
            if not isinstance(payload, dict):
                payload = {}

            def int_value(key: str, default: int = 0) -> int:
                if key not in payload:
                    return default
                try:
                    return int(payload.get(key) or 0)
                except (TypeError, ValueError):
                    return default

            self.experiment_succeeded.setChecked(bool(payload.get("succeeded", False)))
            self.experiment_attempts.setValue(int_value("attempts", 1))
            self.experiment_features_recognized.setValue(int_value("features_recognized"))
            self.experiment_feature_notes.setPlainText(str(payload.get("feature_notes") or ""))
            self.experiment_colormap_used.setChecked(bool(payload.get("colormap_used", False)))
            self.experiment_suggested_seeding_used.setChecked(bool(payload.get("suggested_seeding_used", False)))
            self.experiment_seeding_score.setValue(int_value("seeding_score"))
            self.experiment_seeding_notes.setPlainText(str(payload.get("seeding_notes") or ""))

            image_payload = payload.get("viewport_image")
            if self.experiment_succeeded.isChecked() and isinstance(image_payload, dict):
                self.experiment_viewport_image = dict(image_payload)
                self.experiment_viewport_image_source_path = self.resolve_experiment_viewport_image_path(
                    image_payload,
                    session_path,
                )
            else:
                self.experiment_viewport_image = None
                self.experiment_viewport_image_source_path = None
            self.update_experiment_viewport_image_label()

        def resolve_experiment_viewport_image_path(
            self,
            image_payload: dict[str, Any],
            session_path: str | Path | None = None,
        ) -> str | None:
            path_value = str(image_payload.get("path") or "")
            candidates: list[Path] = []
            if path_value:
                path = Path(path_value)
                if path.is_absolute():
                    candidates.append(path)
                elif session_path:
                    candidates.append(Path(session_path).parent / path)
            absolute_path = str(image_payload.get("absolute_path") or "")
            if absolute_path:
                candidates.append(Path(absolute_path))
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)
            return None

        def update_experiment_viewport_image_label(self) -> None:
            label = getattr(self, "experiment_viewport_image_label", None)
            if label is None:
                return
            succeeded_widget = getattr(self, "experiment_succeeded", None)
            if succeeded_widget is not None and not succeeded_widget.isChecked():
                label.setText("No viewport image associated for failed experiments.")
                label.setToolTip("")
                return
            if self.experiment_viewport_image is None:
                label.setText("No viewport image associated.")
                label.setToolTip("")
                return
            path_text = str(self.experiment_viewport_image.get("path") or "Viewport image associated.")
            if self.experiment_viewport_image_source_path:
                label.setText(path_text)
                label.setToolTip(self.experiment_viewport_image_source_path)
            else:
                label.setText(f"{path_text} (missing)")
                label.setToolTip("")

        def viewport_image_source_path(self) -> tuple[str, str] | None:
            if self.last_preview_png and Path(self.last_preview_png).exists():
                return self.last_preview_png, "safe_preview"
            if self.experiment_viewport_image_source_path and Path(self.experiment_viewport_image_source_path).exists():
                return self.experiment_viewport_image_source_path, "associated_viewport_image"
            return None

        def attach_viewport_image_to_snapshot(
            self,
            snapshot: dict[str, Any],
            session_path: str | Path,
        ) -> None:
            experiment = snapshot.get("experiment")
            if not isinstance(experiment, dict):
                experiment = {}
                snapshot["experiment"] = experiment

            if not bool(experiment.get("succeeded", False)):
                experiment["viewport_image"] = None
                self.experiment_viewport_image = None
                self.experiment_viewport_image_source_path = None
                self.update_experiment_viewport_image_label()
                return

            source = self.viewport_image_source_path()
            if source is None:
                experiment["viewport_image"] = None
                self.experiment_viewport_image = None
                self.experiment_viewport_image_source_path = None
                self.update_experiment_viewport_image_label()
                return

            source_path, source_label = source
            target_path = default_viewport_image_path(session_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if Path(source_path).resolve() != target_path.resolve():
                shutil.copy2(source_path, target_path)

            image_payload = {
                "path": target_path.name,
                "absolute_path": str(target_path.resolve()),
                "source": source_label,
                "captured_at": _utc_timestamp(),
            }
            experiment["viewport_image"] = image_payload
            self.experiment_viewport_image = image_payload
            self.experiment_viewport_image_source_path = str(target_path)
            self.update_experiment_viewport_image_label()

        def session_snapshot(self) -> dict[str, Any]:
            prompt_text = self.prompt_text.toPlainText()
            prompt_bundle = None
            if self.prompt_bundle is not None:
                prompt_bundle = self.prompt_bundle.to_dict()
                prompt_bundle["final_prompt"] = prompt_text
            return {
                "app": "local_vtk_seeding_rag",
                "dataset_path": self.dataset_path.text().strip(),
                "dataset_metadata": self.metadata_payload(),
                "request": self.current_request().to_dict(),
                "rag": {
                    "enabled": self.rag_enabled(),
                },
                "experiment": self.experiment_snapshot(),
                "llm_settings": {
                    "provider": self.current_provider_id(),
                    "model_name": self.model_name.text().strip(),
                    "api_base": self.api_base.text().strip(),
                    "temperature": float(self.temperature.value()),
                    "max_tokens": int(self.max_tokens.value()),
                    "top_k": int(self.top_k.value()),
                },
                "retrieval_payload": self.retrieval_payload,
                "prompt_bundle": prompt_bundle,
                "prompt_text": prompt_text,
                "llm_response": self.llm_response,
                "generated_code": self.code_text.toPlainText(),
                "last_artifacts": self.last_artifacts.to_dict() if self.last_artifacts is not None else None,
                "last_preview_png": self.last_preview_png,
                "window_size": [self.width(), self.height()],
                "splitter_sizes": self.main_splitter.sizes(),
                "workflow_tab_index": self.workflow_tabs.currentIndex(),
                "viewport_tab_index": self.viewport_tabs.currentIndex(),
            }

        def save_session(self) -> None:
            if self.restoring_session:
                return
            try:
                write_last_session(self.session_snapshot())
            except Exception as exc:
                print(f"[main] Could not save session: {exc}", file=sys.stderr, flush=True)

        def restore_last_session(self) -> bool:
            session = load_last_session()
            if session is None:
                return False
            return self.apply_session(
                session,
                success_status="Restored previous session",
                error_status="Could not restore previous session",
            )

        def apply_session(
            self,
            session: dict[str, Any],
            *,
            success_status: str,
            error_status: str,
            session_path: str | Path | None = None,
        ) -> bool:
            self.session_autosave_timer.stop()
            self.restoring_session = True
            try:
                self.dataset_path.setText(str(session.get("dataset_path") or ""))

                request = session.get("request")
                if not isinstance(request, dict):
                    request = {}
                self.set_query_mode(str(request.get("query_mode") or DEFAULT_QUERY_MODE))
                rag_settings = session.get("rag")
                rag_enabled = True
                if isinstance(rag_settings, dict) and rag_settings.get("enabled") is not None:
                    rag_enabled = bool(rag_settings["enabled"])
                self.use_rag_checkbox.setChecked(rag_enabled)
                self.visualization_goal.setPlainText(str(request.get("visualization_goal") or ""))
                self.target_feature.setText(str(request.get("target_feature") or ""))
                self.data_dimension.setText(str(request.get("data_dimension") or ""))
                self.data_type.setText(str(request.get("data_type") or ""))
                self.seeding_behavior.setPlainText(str(request.get("seeding_behavior") or ""))
                self.density_preference.setText(str(request.get("density_clutter_preference") or ""))
                self.constraints.setPlainText(str(request.get("constraints") or ""))
                self.notes.setPlainText(str(request.get("notes") or ""))
                self.restore_experiment_payload(session.get("experiment"), session_path)

                settings = session.get("llm_settings")
                if not isinstance(settings, dict):
                    settings = {}
                provider_id = str(settings.get("provider") or "")
                if not provider_id:
                    provider_id = _infer_provider_id_from_model(str(settings.get("model_name") or ""))
                self.set_provider_id(provider_id)
                if settings.get("model_name"):
                    self.model_name.setText(str(settings["model_name"]))
                self.api_base.setText(str(settings.get("api_base") or ""))
                if settings.get("temperature") is not None:
                    self.temperature.setValue(float(settings["temperature"]))
                if settings.get("max_tokens") is not None:
                    self.max_tokens.setValue(int(settings["max_tokens"]))
                if settings.get("top_k") is not None:
                    self.top_k.setValue(int(settings["top_k"]))

                metadata = session.get("dataset_metadata")
                if isinstance(metadata, dict) and metadata:
                    self.dataset_metadata = metadata
                    self.metadata_text.setPlainText(format_metadata_summary(metadata))
                else:
                    self.dataset_metadata = None
                    self.metadata_text.clear()

                retrieval_payload = session.get("retrieval_payload")
                if isinstance(retrieval_payload, dict):
                    self.retrieval_payload = retrieval_payload
                    self.retrieval_text.setPlainText(_retrieval_summary(retrieval_payload))
                else:
                    self.retrieval_payload = None
                    self.retrieval_text.clear()

                prompt_bundle = session.get("prompt_bundle")
                if isinstance(prompt_bundle, dict) and prompt_bundle.get("final_prompt"):
                    selected_records = prompt_bundle.get("selected_records")
                    prompt_rag_enabled = prompt_bundle.get("rag_enabled")
                    if not isinstance(prompt_rag_enabled, bool):
                        prompt_rag_enabled = (
                            retrieval_payload.get("rag_enabled", True) is not False
                            if isinstance(retrieval_payload, dict)
                            else True
                        )
                    self.prompt_bundle = PromptBundle(
                        retrieval_query=str(prompt_bundle.get("retrieval_query") or ""),
                        final_prompt=str(prompt_bundle.get("final_prompt") or ""),
                        selected_records=selected_records if isinstance(selected_records, list) else [],
                        rag_enabled=prompt_rag_enabled,
                    )
                    self.prompt_text.setPlainText(self.prompt_bundle.final_prompt)
                else:
                    self.prompt_bundle = None
                    self.prompt_text.setPlainText(str(session.get("prompt_text") or ""))

                self.llm_response = str(session.get("llm_response") or "")
                self.generated_code = str(session.get("generated_code") or "")
                self.code_text.setPlainText(self.generated_code)

                artifact_payload = session.get("last_artifacts")
                if isinstance(artifact_payload, dict):
                    self.last_artifacts = RunArtifacts(
                        run_dir=Path(str(artifact_payload.get("run_dir") or "")),
                        prompt_path=Path(str(artifact_payload.get("prompt_path") or "")),
                        retrieval_path=Path(str(artifact_payload.get("retrieval_path") or "")),
                        metadata_path=Path(str(artifact_payload.get("metadata_path") or "")),
                        llm_response_path=Path(str(artifact_payload.get("llm_response_path") or "")),
                        code_path=Path(str(artifact_payload.get("code_path") or "")),
                    )
                else:
                    self.last_artifacts = None

                self.last_preview_png = str(session.get("last_preview_png") or "") or None
                if self.last_preview_png and Path(self.last_preview_png).exists():
                    try:
                        self.show_preview_image(self.last_preview_png)
                    except Exception as exc:
                        print(f"[main] Could not restore preview image: {exc}", file=sys.stderr, flush=True)
                elif getattr(self, "preview_label", None) is not None:
                    self.preview_label.clear()
                    self.preview_label.setText("Safe subprocess preview will appear here after validation/run.")

                window_size = session.get("window_size")
                if isinstance(window_size, list) and len(window_size) == 2:
                    self.resize(int(window_size[0]), int(window_size[1]))
                splitter_sizes = session.get("splitter_sizes")
                if isinstance(splitter_sizes, list) and splitter_sizes:
                    self.main_splitter.setSizes([int(value) for value in splitter_sizes])
                workflow_tab_index = session.get("workflow_tab_index")
                if workflow_tab_index is not None:
                    self.workflow_tabs.setCurrentIndex(int(workflow_tab_index))
                viewport_tab_index = session.get("viewport_tab_index")
                if viewport_tab_index is not None:
                    self.viewport_tabs.setCurrentIndex(int(viewport_tab_index))
            except Exception as exc:
                print(f"[main] {error_status}: {exc}", file=sys.stderr, flush=True)
                self.set_status(error_status)
                return False
            finally:
                self.restoring_session = False

            self.set_status(success_status)
            self.retrieve_button.setEnabled(not self.busy and self.rag_enabled())
            return True

        def save_manual_session(self) -> None:
            snapshot = self.session_snapshot()
            settings = snapshot.get("llm_settings")
            if not isinstance(settings, dict):
                settings = {}
            request = snapshot.get("request")
            if not isinstance(request, dict):
                request = {}
            rag = snapshot.get("rag")
            if not isinstance(rag, dict):
                rag = {}
            DEFAULT_SESSION_ROOT.mkdir(parents=True, exist_ok=True)
            default_path = default_manual_session_path(
                snapshot.get("dataset_path"),
                model_name=str(settings.get("model_name") or ""),
                query_mode=str(request.get("query_mode") or ""),
                rag_enabled=bool(rag.get("enabled", True)),
            )
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Session",
                str(default_path.resolve()),
                "Session files (*.json);;All files (*.*)",
            )
            if not path:
                return
            try:
                self.attach_viewport_image_to_snapshot(snapshot, path)
                saved_path = write_manual_session(snapshot, path)
                write_last_session(snapshot)
            except Exception as exc:
                self.show_error("Session save failed", str(exc))
                return
            self.set_status(f"Saved session: {saved_path.name}")

        def load_manual_session(self) -> None:
            DEFAULT_SESSION_ROOT.mkdir(parents=True, exist_ok=True)
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Session",
                str(DEFAULT_SESSION_ROOT.resolve()),
                "Session files (*.json);;All files (*.*)",
            )
            if not path:
                return

            session = load_session(path)
            if session is None:
                self.show_error("Session load failed", "The selected file is not a valid session JSON object.")
                return

            if self.viewport_process_running():
                decision = QMessageBox.question(
                    self,
                    "Load session?",
                    "Loading a session will stop the current interactive viewport and replace the current UI state. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if decision != QMessageBox.StandardButton.Yes:
                    self.set_status("Session load cancelled")
                    return
            self.close_existing_viewport_process()

            loaded = self.apply_session(
                session,
                success_status=f"Loaded session: {Path(path).name}",
                error_status="Session load failed",
                session_path=path,
            )
            if loaded:
                self.save_session()

        def viewport_process_running(self) -> bool:
            process = self.visualization_process
            return process is not None and process.state() != QProcess.ProcessState.NotRunning

        def run_background_task(
            self,
            *,
            message: str,
            work: Callable[[Callable[[str], None]], Any],
            on_success: Callable[[Any], None],
            success_message: str,
            error_title: str,
        ) -> None:
            if self.busy:
                QMessageBox.information(self, "Task already running", "Please wait for the current task to finish.")
                return

            self.set_busy(message)
            worker = BackgroundTask(work)
            self.active_workers.add(worker)
            worker.signals.status.connect(self.set_status)
            worker.signals.finished.connect(
                lambda result, current_worker=worker: self.finish_background_task(
                    result,
                    current_worker,
                    on_success,
                    success_message,
                    error_title,
                )
            )
            worker.signals.failed.connect(
                lambda error, current_worker=worker: self.fail_background_task(
                    error_title,
                    error,
                    current_worker,
                )
            )
            self.thread_pool.start(worker)

        def finish_background_task(
            self,
            result: Any,
            worker: BackgroundTask,
            on_success: Callable[[Any], None],
            success_message: str,
            error_title: str,
        ) -> None:
            self.active_workers.discard(worker)
            try:
                on_success(result)
            except Exception as exc:
                self.set_idle(error_title)
                self.show_error(error_title, str(exc))
                return
            self.set_idle(success_message)

        def fail_background_task(self, error_title: str, error: str, worker: BackgroundTask) -> None:
            self.active_workers.discard(worker)
            self.set_idle(error_title)
            self.show_error(error_title, error)

        def _build_left_panel(self) -> QWidget:
            container = QWidget()
            layout = QVBoxLayout(container)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content = QWidget()
            content_layout = QVBoxLayout(content)

            content_layout.addWidget(self._dataset_group())
            content_layout.addWidget(self._request_group())
            content_layout.addWidget(self._llm_group())
            content_layout.addWidget(self._action_group())
            content_layout.addWidget(self._tabs())
            content_layout.addStretch(1)

            scroll.setWidget(content)
            layout.addWidget(scroll)
            return container

        def _dataset_group(self) -> QGroupBox:
            group = QGroupBox("Dataset")
            layout = QVBoxLayout(group)
            row = QHBoxLayout()
            self.dataset_path = QLineEdit()
            self.dataset_path.setPlaceholderText("Select .vtk, .vti, .vtu, .vtp, .vts, or .vtr")
            self.browse_button = QPushButton("Browse")
            self.browse_button.clicked.connect(self.browse_dataset)
            row.addWidget(self.dataset_path)
            row.addWidget(self.browse_button)
            layout.addLayout(row)

            self.analyze_button = QPushButton("Analyze Dataset Metadata")
            self.analyze_button.clicked.connect(self.analyze_dataset)
            layout.addWidget(self.analyze_button)

            self.metadata_text = WheelScopedTextEdit()
            self.metadata_text.setReadOnly(True)
            self.metadata_text.setMinimumHeight(130)
            layout.addWidget(self.metadata_text)
            return group

        def _request_group(self) -> QGroupBox:
            group = QGroupBox("Visualization Request")
            layout = QFormLayout(group)
            self.query_mode_combo = QComboBox()
            for query_mode in QUERY_MODES:
                self.query_mode_combo.addItem(query_mode, query_mode)
            self.use_rag_checkbox = QCheckBox("Use RAG")
            self.use_rag_checkbox.setChecked(True)
            self.visualization_goal = WheelScopedTextEdit()
            self.visualization_goal.setPlaceholderText("What should the visualization show or help you understand?")
            self.visualization_goal.setMinimumHeight(72)
            self.target_feature = QLineEdit()
            self.target_feature.setPlaceholderText("vortices, critical points, high-entropy regions, boundaries...")
            self.data_dimension = QLineEdit()
            self.data_type = QLineEdit()
            self.seeding_behavior = WheelScopedTextEdit()
            self.seeding_behavior.setPlaceholderText("Desired seeding technique or behavior")
            self.seeding_behavior.setMinimumHeight(72)
            self.density_preference = QLineEdit()
            self.density_preference.setPlaceholderText("dense, sparse, low clutter, even coverage...")
            self.constraints = WheelScopedTextEdit()
            self.constraints.setPlaceholderText("Implementation constraints, important arrays, camera/view needs...")
            self.constraints.setMinimumHeight(72)
            self.notes = WheelScopedTextEdit()
            self.notes.setPlaceholderText("Additional notes")
            self.notes.setMinimumHeight(56)

            layout.addRow("Query mode", self.query_mode_combo)
            layout.addRow("RAG", self.use_rag_checkbox)
            layout.addRow("Visualisation goal", self.visualization_goal)
            layout.addRow("Target feature", self.target_feature)
            layout.addRow("Data dimension", self.data_dimension)
            layout.addRow("Data type", self.data_type)
            layout.addRow("Seeding behavior", self.seeding_behavior)
            layout.addRow("Density/clutter", self.density_preference)
            layout.addRow("Constraints", self.constraints)
            layout.addRow("Notes", self.notes)
            return group

        def _llm_group(self) -> QGroupBox:
            group = QGroupBox("LLM")
            layout = QFormLayout(group)
            self.provider_combo = QComboBox()
            for provider in LLM_PROVIDER_OPTIONS:
                self.provider_combo.addItem(provider["label"], provider["id"])
            self.provider_combo.setCurrentIndex(self.provider_combo.findData(DEFAULT_LLM_PROVIDER_ID))
            self.model_name = QLineEdit(LLM_PROVIDER_BY_ID[DEFAULT_LLM_PROVIDER_ID]["default_model"])
            self.api_key = QLineEdit()
            self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.api_base = QLineEdit()
            self.temperature = QDoubleSpinBox()
            self.temperature.setDecimals(2)
            self.temperature.setRange(0.0, 2.0)
            self.temperature.setSingleStep(0.05)
            self.temperature.setValue(0.2)
            self.max_tokens = QSpinBox()
            self.max_tokens.setRange(512, MAX_TOKEN_LIMIT)
            self.max_tokens.setSingleStep(512)
            self.max_tokens.setValue(DEFAULT_MAX_TOKENS)
            self.top_k = QSpinBox()
            self.top_k.setRange(1, 20)
            self.top_k.setValue(5)
            self.provider_combo.currentIndexChanged.connect(self.handle_provider_changed)
            self.update_provider_placeholders()

            layout.addRow("Provider", self.provider_combo)
            layout.addRow("Model", self.model_name)
            layout.addRow("API key", self.api_key)
            layout.addRow("API base", self.api_base)
            layout.addRow("Temperature", self.temperature)
            layout.addRow("Max tokens", self.max_tokens)
            layout.addRow("Top K", self.top_k)
            return group

        def _action_group(self) -> QGroupBox:
            group = QGroupBox("Workflow")
            layout = QVBoxLayout(group)
            self.retrieve_button = QPushButton("Retrieve Seeding Records")
            self.retrieve_button.clicked.connect(self.retrieve_records)
            self.build_prompt_button = QPushButton("Build Final Prompt")
            self.build_prompt_button.clicked.connect(self.build_prompt)
            self.generate_button = QPushButton("Generate VTK Code")
            self.generate_button.clicked.connect(self.generate_vtk_code)
            self.run_button = QPushButton("Validate, Preview, And Open Viewport")
            self.run_button.clicked.connect(self.run_generated_code)
            self.stop_viewport_button = QPushButton("Stop Interactive Viewport")
            self.stop_viewport_button.clicked.connect(self.stop_interactive_viewport)
            self.stop_viewport_button.setEnabled(False)
            self.save_session_button = QPushButton("Save Session")
            self.save_session_button.clicked.connect(self.save_manual_session)
            self.load_session_button = QPushButton("Load Session")
            self.load_session_button.clicked.connect(self.load_manual_session)

            layout.addWidget(self.retrieve_button)
            layout.addWidget(self.build_prompt_button)
            layout.addWidget(self.generate_button)
            layout.addWidget(self.run_button)
            layout.addWidget(self.stop_viewport_button)
            session_row = QHBoxLayout()
            session_row.addWidget(self.save_session_button)
            session_row.addWidget(self.load_session_button)
            layout.addLayout(session_row)
            return group

        def _tabs(self) -> QTabWidget:
            tabs = QTabWidget()
            self.workflow_tabs = tabs
            self.retrieval_text = WheelScopedPlainTextEdit()
            self.retrieval_text.setReadOnly(True)
            self.prompt_text = WheelScopedPlainTextEdit()
            self.code_text = WheelScopedPlainTextEdit()
            tabs.addTab(self.retrieval_text, "Retrieval")
            tabs.addTab(self.prompt_text, "Prompt")
            tabs.addTab(self.code_text, "Code")
            tabs.addTab(self._experiment_tab(), "Experiment")
            tabs.setMinimumHeight(320)
            return tabs

        def _experiment_tab(self) -> QWidget:
            tab = QWidget()
            outer_layout = QVBoxLayout(tab)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            outer_layout.addWidget(scroll)

            container = QWidget()
            layout = QFormLayout(container)
            layout.setVerticalSpacing(8)
            layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

            self.experiment_succeeded = QCheckBox("Succeeded")
            self.experiment_attempts = QSpinBox()
            self.experiment_attempts.setRange(0, 100000)
            self.experiment_attempts.setValue(1)
            self.experiment_features_recognized = QSpinBox()
            self.experiment_features_recognized.setRange(0, 100000)
            self.experiment_feature_notes = WheelScopedPlainTextEdit()
            self.experiment_feature_notes.setFixedHeight(88)
            self.experiment_feature_notes.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
            self.experiment_feature_notes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.experiment_colormap_used = QCheckBox()
            self.experiment_suggested_seeding_used = QCheckBox()
            self.experiment_seeding_score = QSpinBox()
            self.experiment_seeding_score.setRange(0, 100)
            self.experiment_seeding_notes = WheelScopedPlainTextEdit()
            self.experiment_seeding_notes.setFixedHeight(88)
            self.experiment_seeding_notes.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
            self.experiment_seeding_notes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.experiment_viewport_image_label = QLabel("No viewport image associated.")
            self.experiment_viewport_image_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            layout.addRow("Result", self.experiment_succeeded)
            layout.addRow("Attempts", self.experiment_attempts)
            layout.addRow("Amount of Features Recognized", self.experiment_features_recognized)
            layout.addRow("Notes about Features", self.experiment_feature_notes)
            layout.addRow("Colormap used", self.experiment_colormap_used)
            layout.addRow("Suggested Seeding used", self.experiment_suggested_seeding_used)
            layout.addRow("Seeding Score", self.experiment_seeding_score)
            layout.addRow("Notes about Seeding Strategy", self.experiment_seeding_notes)
            layout.addRow("Viewport image", self.experiment_viewport_image_label)
            scroll.setWidget(container)
            return tab

        def _build_viewport_panel(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            self.viewport_tabs = QTabWidget()

            preview_panel = QWidget()
            preview_layout = QVBoxLayout(preview_panel)
            self.preview_label = QLabel(SAFE_PREVIEW_PLACEHOLDER)
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setMinimumSize(640, 420)
            self.preview_label.setStyleSheet("background: #15171d; color: #d8dce7;")
            preview_layout.addWidget(self.preview_label, 1)

            interactive_panel = QWidget()
            interactive_layout = QVBoxLayout(interactive_panel)
            self.interactive_status_label = QLabel("No interactive viewport process running.")
            self.viewport_log = WheelScopedPlainTextEdit()
            self.viewport_log.setReadOnly(True)
            self.viewport_log.setPlaceholderText("Interactive viewport process output")
            interactive_layout.addWidget(self.interactive_status_label)
            interactive_layout.addWidget(self.viewport_log, 1)

            self.viewport_tabs.addTab(preview_panel, "Safe Preview")
            self.viewport_tabs.addTab(interactive_panel, "Interactive Viewport")
            layout.addWidget(self.viewport_tabs, 1)
            return panel

        def browse_dataset(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select VTK dataset",
                str(Path.cwd()),
                "VTK datasets (*.vtk *.vti *.vtu *.vtp *.vts *.vtr);;All files (*.*)",
            )
            if path:
                self.dataset_path.setText(path)

        def current_request(self) -> UserRequest:
            return UserRequest(
                query_mode=self.current_query_mode(),
                visualization_goal=self.visualization_goal.toPlainText(),
                target_feature=self.target_feature.text(),
                data_dimension=self.data_dimension.text(),
                data_type=self.data_type.text(),
                seeding_behavior=self.seeding_behavior.toPlainText(),
                density_clutter_preference=self.density_preference.text(),
                constraints=self.constraints.toPlainText(),
                notes=self.notes.toPlainText(),
            )

        def analyze_dataset(self) -> None:
            dataset_path = self.dataset_path.text().strip()
            if not dataset_path:
                self.show_error("Dataset metadata error", "Select a VTK dataset first.")
                return

            def work(update_status: Callable[[str], None]) -> Any:
                update_status("Reading VTK dataset metadata...")
                return extract_dataset_metadata(dataset_path)

            def on_success(metadata: Any) -> None:
                self.dataset_metadata = metadata.to_dict()
                defaults = infer_request_defaults_from_metadata(self.dataset_metadata)
                if defaults["data_dimension"] and not self.data_dimension.text().strip():
                    self.data_dimension.setText(defaults["data_dimension"])
                if defaults["data_type"] and not self.data_type.text().strip():
                    self.data_type.setText(defaults["data_type"])
                self.metadata_text.setPlainText(format_metadata_summary(metadata))
                self.schedule_session_autosave()

            self.run_background_task(
                message="Reading dataset metadata...",
                work=work,
                on_success=on_success,
                success_message="Dataset metadata loaded",
                error_title="Dataset metadata error",
            )

        def retrieve_payload_for_request(
            self,
            request: UserRequest,
            metadata: dict[str, Any],
            top_k: int,
            update_status: Callable[[str], None],
        ) -> tuple[HybridRetriever, dict[str, Any]]:
            retriever = self.retriever
            if retriever is None:
                update_status("Loading retrieval index and embedding model...")
                retriever = HybridRetriever(self.index_dir)
            update_status("Embedding query and ranking seeding records...")
            retrieval_query = build_retrieval_query(request, metadata)
            return retriever, retriever.retrieve(retrieval_query, top_k=top_k)

        def apply_retrieval_payload(self, result: tuple[HybridRetriever, dict[str, Any]]) -> None:
            retriever, payload = result
            self.retriever = retriever
            self.retrieval_payload = payload
            self.retrieval_text.setPlainText(_retrieval_summary(payload))
            self.schedule_session_autosave()

        def apply_prompt_bundle(self, result: dict[str, Any]) -> None:
            retriever = result.get("retriever")
            if retriever is not None:
                self.retriever = retriever
            retrieval_payload = result.get("retrieval_payload")
            if retrieval_payload is not None:
                self.retrieval_payload = retrieval_payload
                self.retrieval_text.setPlainText(_retrieval_summary(retrieval_payload))
            self.prompt_bundle = result["prompt_bundle"]
            self.prompt_text.setPlainText(self.prompt_bundle.final_prompt)
            self.schedule_session_autosave()

        def apply_llm_result(self, result: dict[str, Any]) -> None:
            self.apply_prompt_bundle(result)
            response = result["response"]
            self.llm_response = response.content
            self.generated_code = response.code
            self.code_text.setPlainText(self.generated_code)
            self.save_run_artifacts()
            if response.validation_errors:
                QMessageBox.warning(
                    self,
                    "Generated code still failed validation",
                    "The app tried to repair the generated code twice, but validation still failed:\n\n"
                    + "\n".join(response.validation_errors),
                )

        def apply_safe_execution_result(self, result: SafeExecutionResult) -> None:
            self.generated_code = self.code_text.toPlainText().strip()
            self.last_preview_png = result.output_png
            self.experiment_viewport_image = None
            self.experiment_viewport_image_source_path = None
            self.update_experiment_viewport_image_label()
            self.show_preview_image(result.output_png)
            self.save_run_artifacts()
            self.launch_interactive_viewport()

        def ensure_retriever(self) -> HybridRetriever:
            if self.retriever is None:
                self.set_status("Loading retrieval index...")
                self.retriever = HybridRetriever(self.index_dir)
            return self.retriever

        def metadata_payload(self) -> dict[str, Any]:
            if self.dataset_metadata is None:
                return {}
            return self.dataset_metadata

        def retrieve_records(self) -> None:
            request = self.current_request()
            if not self.rag_enabled():
                retrieval_query = build_retrieval_query(request, self.metadata_payload())
                self.retrieval_payload = self.disabled_rag_payload(retrieval_query)
                self.prompt_bundle = None
                self.retrieval_text.setPlainText(_retrieval_summary(self.retrieval_payload))
                self.schedule_session_autosave()
                self.set_status("RAG retrieval disabled; no retrieval performed")
                return

            retrieval_query = build_retrieval_query(request, self.metadata_payload())
            if not retrieval_query:
                self.show_error("Missing request", "Fill in at least one visualization request field.")
                return

            metadata = self.metadata_payload()
            top_k = self.top_k.value()

            def work(update_status: Callable[[str], None]) -> Any:
                return self.retrieve_payload_for_request(request, metadata, top_k, update_status)

            self.run_background_task(
                message="Preparing retrieval query...",
                work=work,
                on_success=self.apply_retrieval_payload,
                success_message="Retrieved seeding records",
                error_title="Retrieval failed",
            )

        def build_prompt(self) -> None:
            request = self.current_request()
            metadata = self.metadata_payload()
            retrieval_query = build_retrieval_query(request, metadata)
            if not retrieval_query:
                self.show_error("Missing request", "Fill in at least one visualization request field.")
                return
            rag_enabled = self.rag_enabled()
            top_k = self.top_k.value()
            current_payload = self.current_retrieval_payload_for_rag_state()

            def work(update_status: Callable[[str], None]) -> dict[str, Any]:
                retriever: HybridRetriever | None = self.retriever if rag_enabled else None
                retrieval_payload = current_payload
                if rag_enabled and retrieval_payload is None:
                    retriever, retrieval_payload = self.retrieve_payload_for_request(
                        request,
                        metadata,
                        top_k,
                        update_status,
                    )
                if not rag_enabled:
                    retrieval_payload = self.disabled_rag_payload(retrieval_query)
                update_status("Combining request, metadata, and retrieval settings into final prompt...")
                prompt_bundle = build_final_prompt(
                    user_request=request,
                    dataset_metadata=metadata,
                    retrieval_payload=retrieval_payload,
                    rag_enabled=rag_enabled,
                )
                return {
                    "retriever": retriever,
                    "retrieval_payload": retrieval_payload,
                    "prompt_bundle": prompt_bundle,
                }

            self.run_background_task(
                message="Building final prompt...",
                work=work,
                on_success=self.apply_prompt_bundle,
                success_message="Prompt built",
                error_title="Prompt build failed",
            )

        def generate_vtk_code(self) -> None:
            request = self.current_request()
            metadata = self.metadata_payload()
            retrieval_query = build_retrieval_query(request, metadata)
            if not retrieval_query:
                self.show_error("Missing request", "Fill in at least one visualization request field.")
                return
            settings = LLMSettings(
                model=self.model_name.text().strip(),
                provider=self.current_provider_id(),
                api_key=self.api_key.text().strip(),
                api_base=self.api_base.text().strip(),
                temperature=float(self.temperature.value()),
                max_tokens=int(self.max_tokens.value()),
            )
            rag_enabled = self.rag_enabled()
            current_payload = self.current_retrieval_payload_for_rag_state()
            current_bundle = self.current_prompt_bundle_for_rag_state()
            top_k = self.top_k.value()

            def work(update_status: Callable[[str], None]) -> dict[str, Any]:
                retriever: HybridRetriever | None = self.retriever if rag_enabled else None
                retrieval_payload = current_payload
                prompt_bundle = current_bundle
                if prompt_bundle is None:
                    if rag_enabled and retrieval_payload is None:
                        retriever, retrieval_payload = self.retrieve_payload_for_request(
                            request,
                            metadata,
                            top_k,
                            update_status,
                        )
                    if not rag_enabled:
                        retrieval_payload = self.disabled_rag_payload(retrieval_query)
                    update_status("Combining request, metadata, and retrieval settings into final prompt...")
                    prompt_bundle = build_final_prompt(
                        user_request=request,
                        dataset_metadata=metadata,
                        retrieval_payload=retrieval_payload,
                        rag_enabled=rag_enabled,
                    )
                resolved_model = normalize_model_name(settings.model, settings.provider)
                update_status(f"Calling LLM: waiting for response from {resolved_model}...")
                response = generate_code(prompt_bundle.final_prompt, settings)
                update_status("Extracting generated Python code...")
                return {
                    "retriever": retriever,
                    "retrieval_payload": retrieval_payload,
                    "prompt_bundle": prompt_bundle,
                    "response": response,
                }

            self.run_background_task(
                message="Preparing LLM request...",
                work=work,
                on_success=self.apply_llm_result,
                success_message="Generated VTK code",
                error_title="LLM generation failed",
            )

        def run_generated_code(self) -> None:
            code = self.code_text.toPlainText().strip()
            if not code:
                self.show_error("No code", "Generate or paste VTK code before running.")
                return

            self.set_status("Validating generated code...")
            validation = validate_generated_code(code)
            if not validation.ok:
                self.show_error("Generated code failed validation", "\n".join(validation.errors))
                return

            if self.viewport_process_running():
                replace_decision = QMessageBox.question(
                    self,
                    "Replace interactive viewport?",
                    "An interactive viewport is already running. Stop it and launch a new one?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if replace_decision != QMessageBox.StandardButton.Yes:
                    self.set_status("Execution cancelled")
                    return

            decision = QMessageBox.question(
                self,
                "Run generated code?",
                "The generated code passed basic AST checks. The app will smoke-test it in a subprocess, then open an interactive VTK child window. Run it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if decision != QMessageBox.StandardButton.Yes:
                self.set_status("Execution cancelled")
                return

            self.reset_experiment_viewport_image_for_execution_attempt()
            dataset_path = self.dataset_path.text().strip()
            metadata = self.metadata_payload()
            user_request = self.current_request().to_dict()

            def work(update_status: Callable[[str], None]) -> SafeExecutionResult:
                update_status("Running generated VTK code in isolated subprocess...")
                result = run_generated_code_safely(
                    code,
                    dataset_path=dataset_path,
                    metadata=metadata,
                    user_request=user_request,
                )
                update_status("Loading safe preview image...")
                return result

            self.run_background_task(
                message="Smoke-testing generated VTK code safely...",
                work=work,
                on_success=self.apply_safe_execution_result,
                success_message="Safe preview rendered; launching interactive viewport",
                error_title="Visualization execution failed",
            )

        def show_preview_image(self, path: str) -> None:
            pixmap = QPixmap(path)
            if pixmap.isNull():
                raise RuntimeError(f"Could not load preview image: {path}")
            scaled = pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)
            self.viewport_tabs.setCurrentIndex(0)

        def clear_safe_preview(self) -> None:
            self.last_preview_png = None
            preview_label = getattr(self, "preview_label", None)
            if preview_label is not None:
                preview_label.clear()
                preview_label.setText(SAFE_PREVIEW_PLACEHOLDER)

        def launch_interactive_viewport(self) -> None:
            self.close_existing_viewport_process()
            self.viewport_log.clear()
            self.viewport_tabs.setCurrentIndex(1)
            self.interactive_status_label.setText("Launching interactive VTK viewport process...")

            launch = prepare_interactive_viewport_launch(
                self.code_text.toPlainText().strip(),
                dataset_path=self.dataset_path.text().strip(),
                metadata=self.metadata_payload(),
                user_request=self.current_request().to_dict(),
            )

            process = QProcess(self)
            process.setProgram(launch.program)
            process.setArguments(launch.arguments)
            process.setWorkingDirectory(launch.working_directory)
            process.readyReadStandardOutput.connect(self.read_viewport_stdout)
            process.readyReadStandardError.connect(self.read_viewport_stderr)
            process.started.connect(self.handle_viewport_started)
            process.errorOccurred.connect(self.handle_viewport_error)
            process.finished.connect(self.handle_viewport_finished)

            self.visualization_process = process
            self.visualization_payload_dir = launch.payload_dir
            self.stop_viewport_button.setEnabled(True)
            self.set_status("Launching interactive VTK viewport process...")
            process.start()

        def read_viewport_stdout(self) -> None:
            process = self.visualization_process
            if process is None:
                return
            text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
            if text:
                print(text, end="", file=sys.stdout, flush=True)
                self.append_viewport_log(text.rstrip())
            if "interactive viewport ready" in text:
                self.interactive_status_label.setText("Interactive VTK viewport running in a child window.")
                self.set_status("Interactive VTK viewport running")

        def read_viewport_stderr(self) -> None:
            process = self.visualization_process
            if process is None:
                return
            text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
            if text:
                print(text, end="", file=sys.stderr, flush=True)
                self.append_viewport_log("[stderr]\n" + text.rstrip())

        def append_viewport_log(self, text: str) -> None:
            if text:
                self.viewport_log.appendPlainText(text)

        def handle_viewport_started(self) -> None:
            self.interactive_status_label.setText("Interactive VTK viewport process started.")
            self.set_status("Interactive VTK viewport process started")

        def handle_viewport_error(self, error: Any) -> None:
            message = f"[main] QProcess error: {error}"
            print(message, file=sys.stderr, flush=True)
            self.append_viewport_log(message)
            self.interactive_status_label.setText("Interactive VTK viewport process error.")
            self.set_status("Interactive viewport process error")
            if error == QProcess.ProcessError.FailedToStart:
                cleanup_interactive_payload(self.visualization_payload_dir)
                self.visualization_payload_dir = None
                self.visualization_process = None
                self.stop_viewport_button.setEnabled(False)

        def handle_viewport_finished(self, exit_code: int, exit_status: Any) -> None:
            self.read_viewport_stdout()
            self.read_viewport_stderr()
            cleanup_interactive_payload(self.visualization_payload_dir)
            self.visualization_payload_dir = None
            self.visualization_process = None
            self.stop_viewport_button.setEnabled(False)

            crashed = exit_status == QProcess.ExitStatus.CrashExit or exit_code != 0
            if crashed:
                print(
                    f"[main] Interactive VTK viewport exited with code {exit_code}.",
                    file=sys.stderr,
                    flush=True,
                )
                self.interactive_status_label.setText(
                    f"Interactive VTK viewport exited with code {exit_code}."
                )
                self.set_status("Interactive VTK viewport exited unexpectedly")
                log_tail = self.viewport_log.toPlainText()[-3000:]
                QMessageBox.warning(
                    self,
                    "Interactive viewport exited",
                    "The interactive VTK viewport process exited, but the main app stayed open.\n\n"
                    f"Exit code: {exit_code}\n\n"
                    f"Recent process output:\n{log_tail}",
                )
            else:
                self.interactive_status_label.setText("Interactive VTK viewport closed.")
                self.set_status("Interactive VTK viewport closed")

        def stop_interactive_viewport(self) -> None:
            if not self.viewport_process_running():
                self.interactive_status_label.setText("No interactive viewport process running.")
                self.stop_viewport_button.setEnabled(False)
                return
            self.close_existing_viewport_process()
            self.interactive_status_label.setText("Interactive VTK viewport stopped.")
            self.append_viewport_log("[main] Interactive viewport stopped.")
            self.set_status("Interactive VTK viewport stopped")

        def close_existing_viewport_process(self) -> None:
            process = self.visualization_process
            self.visualization_process = None
            if process is not None:
                process.blockSignals(True)
                if process.state() != QProcess.ProcessState.NotRunning:
                    process.terminate()
                    if not process.waitForFinished(3000):
                        process.kill()
                        process.waitForFinished(1000)
                process.deleteLater()
            cleanup_interactive_payload(self.visualization_payload_dir)
            self.visualization_payload_dir = None
            if getattr(self, "stop_viewport_button", None) is not None:
                self.stop_viewport_button.setEnabled(False)

        def save_run_artifacts(self) -> None:
            if self.prompt_bundle is None or self.retrieval_payload is None:
                return
            self.last_artifacts = write_run_artifacts(
                prompt=self.prompt_bundle.final_prompt,
                retrieval_payload=self.retrieval_payload,
                dataset_metadata=self.metadata_payload(),
                user_request=self.current_request().to_dict(),
                llm_response=self.llm_response,
                code=self.code_text.toPlainText(),
            )
            self.schedule_session_autosave()

        def show_error(self, title: str, message: str) -> None:
            QMessageBox.critical(self, title, message)
            self.set_status(title)

        def closeEvent(self, event: Any) -> None:
            self.save_session()
            self.close_existing_viewport_process()
            super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    if GUI_IMPORT_ERROR is not None:
        raise SystemExit(
            "PySide6 and VTK are required for the desktop app. "
            "Install dependencies with `pip install -r requirements.txt`.\n"
            f"Original import error: {GUI_IMPORT_ERROR}"
        )

    app = QApplication(argv or sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
