from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from streamline_retrieval.retrieval import HybridRetriever

from .code_validation import validate_generated_code
from .dataset_metadata import extract_dataset_metadata, format_metadata_summary
from .llm import DEFAULT_MAX_TOKENS, LLMSettings, generate_code
from .prompting import PromptBundle, build_final_prompt
from .query import UserRequest, build_retrieval_query, infer_request_defaults_from_metadata
from .run_store import RunArtifacts, write_run_artifacts
from .safe_execution import SafeExecutionResult, run_generated_code_safely


GUI_IMPORT_ERROR: Exception | None = None
try:
    from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication,
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
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QDoubleSpinBox,
    )
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except Exception as exc:  # pragma: no cover - depends on optional GUI dependencies
    GUI_IMPORT_ERROR = exc


def _pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _retrieval_summary(payload: dict[str, Any]) -> str:
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
        def __init__(self, index_dir: str = "index") -> None:
            super().__init__()
            self.setWindowTitle("Local VTK Seeding RAG")
            self.resize(1500, 920)

            self.index_dir = index_dir
            self.retriever: HybridRetriever | None = None
            self.dataset_metadata: dict[str, Any] | None = None
            self.retrieval_payload: dict[str, Any] | None = None
            self.prompt_bundle: PromptBundle | None = None
            self.generated_code: str = ""
            self.llm_response: str = ""
            self.last_artifacts: RunArtifacts | None = None
            self.last_preview_png: str | None = None
            self.thread_pool = QThreadPool.globalInstance()
            self.active_workers: set[BackgroundTask] = set()
            self.busy = False

            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(self._build_left_panel())
            splitter.addWidget(self._build_viewport_panel())
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([560, 940])
            self.setCentralWidget(splitter)
            self._build_status_strip()
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
            self.statusBar().showMessage(message)

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
                getattr(self, "retrieve_button", None),
                getattr(self, "build_prompt_button", None),
                getattr(self, "generate_button", None),
                getattr(self, "run_button", None),
            ):
                if button is not None:
                    button.setEnabled(enabled)

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

            self.metadata_text = QTextEdit()
            self.metadata_text.setReadOnly(True)
            self.metadata_text.setMinimumHeight(130)
            layout.addWidget(self.metadata_text)
            return group

        def _request_group(self) -> QGroupBox:
            group = QGroupBox("Visualization Request")
            layout = QFormLayout(group)
            self.visualization_goal = QTextEdit()
            self.visualization_goal.setPlaceholderText("What should the visualization show or help you understand?")
            self.visualization_goal.setMinimumHeight(72)
            self.target_feature = QLineEdit()
            self.target_feature.setPlaceholderText("vortices, critical points, high-entropy regions, boundaries...")
            self.data_dimension = QLineEdit()
            self.data_type = QLineEdit()
            self.seeding_behavior = QTextEdit()
            self.seeding_behavior.setPlaceholderText("Desired seeding technique or behavior")
            self.seeding_behavior.setMinimumHeight(72)
            self.density_preference = QLineEdit()
            self.density_preference.setPlaceholderText("dense, sparse, low clutter, even coverage...")
            self.constraints = QTextEdit()
            self.constraints.setPlaceholderText("Implementation constraints, important arrays, camera/view needs...")
            self.constraints.setMinimumHeight(72)
            self.notes = QTextEdit()
            self.notes.setPlaceholderText("Additional notes")
            self.notes.setMinimumHeight(56)

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
            self.model_name = QLineEdit("openai/gpt-5.4-mini")
            self.api_key = QLineEdit()
            self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.api_key.setPlaceholderText("Optional; environment variables also work")
            self.api_base = QLineEdit()
            self.api_base.setPlaceholderText("Optional LiteLLM api_base")
            self.temperature = QDoubleSpinBox()
            self.temperature.setDecimals(2)
            self.temperature.setRange(0.0, 2.0)
            self.temperature.setSingleStep(0.05)
            self.temperature.setValue(0.2)
            self.max_tokens = QSpinBox()
            self.max_tokens.setRange(512, 32768)
            self.max_tokens.setSingleStep(512)
            self.max_tokens.setValue(DEFAULT_MAX_TOKENS)
            self.top_k = QSpinBox()
            self.top_k.setRange(1, 20)
            self.top_k.setValue(5)

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
            self.run_button = QPushButton("Validate And Run Code")
            self.run_button.clicked.connect(self.run_generated_code)

            layout.addWidget(self.retrieve_button)
            layout.addWidget(self.build_prompt_button)
            layout.addWidget(self.generate_button)
            layout.addWidget(self.run_button)
            return group

        def _tabs(self) -> QTabWidget:
            tabs = QTabWidget()
            self.retrieval_text = QPlainTextEdit()
            self.retrieval_text.setReadOnly(True)
            self.prompt_text = QPlainTextEdit()
            self.code_text = QPlainTextEdit()
            tabs.addTab(self.retrieval_text, "Retrieval")
            tabs.addTab(self.prompt_text, "Prompt")
            tabs.addTab(self.code_text, "Code")
            tabs.setMinimumHeight(320)
            return tabs

        def _build_viewport_panel(self) -> QWidget:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            self.viewport_tabs = QTabWidget()

            preview_panel = QWidget()
            preview_layout = QVBoxLayout(preview_panel)
            self.preview_label = QLabel("Safe subprocess preview will appear here after validation/run.")
            self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_label.setMinimumSize(640, 420)
            self.preview_label.setStyleSheet("background: #15171d; color: #d8dce7;")
            preview_layout.addWidget(self.preview_label, 1)

            interactive_panel = QWidget()
            interactive_layout = QVBoxLayout(interactive_panel)
            interactive_layout.addWidget(QLabel("Interactive VTK viewport"))
            self.vtk_widget = QVTKRenderWindowInteractor(interactive_panel)
            interactive_layout.addWidget(self.vtk_widget, 1)
            self.vtk_widget.Initialize()

            self.viewport_tabs.addTab(preview_panel, "Safe Preview")
            self.viewport_tabs.addTab(interactive_panel, "Interactive VTK")
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
            self.show_preview_image(result.output_png)
            self.save_run_artifacts()

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
            top_k = self.top_k.value()
            current_payload = self.retrieval_payload

            def work(update_status: Callable[[str], None]) -> dict[str, Any]:
                retriever: HybridRetriever | None = self.retriever
                retrieval_payload = current_payload
                if retrieval_payload is None:
                    retriever, retrieval_payload = self.retrieve_payload_for_request(
                        request,
                        metadata,
                        top_k,
                        update_status,
                    )
                update_status("Combining request, metadata, and retrieved records into final prompt...")
                prompt_bundle = build_final_prompt(
                    user_request=request,
                    dataset_metadata=metadata,
                    retrieval_payload=retrieval_payload,
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
                api_key=self.api_key.text().strip(),
                api_base=self.api_base.text().strip(),
                temperature=float(self.temperature.value()),
                max_tokens=int(self.max_tokens.value()),
            )
            current_payload = self.retrieval_payload
            current_bundle = self.prompt_bundle
            top_k = self.top_k.value()

            def work(update_status: Callable[[str], None]) -> dict[str, Any]:
                retriever: HybridRetriever | None = self.retriever
                retrieval_payload = current_payload
                prompt_bundle = current_bundle
                if prompt_bundle is None:
                    if retrieval_payload is None:
                        retriever, retrieval_payload = self.retrieve_payload_for_request(
                            request,
                            metadata,
                            top_k,
                            update_status,
                        )
                    update_status("Combining request, metadata, and retrieved records into final prompt...")
                    prompt_bundle = build_final_prompt(
                        user_request=request,
                        dataset_metadata=metadata,
                        retrieval_payload=retrieval_payload,
                    )
                update_status(f"Calling LLM: waiting for response from {settings.model}...")
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

            decision = QMessageBox.question(
                self,
                "Run generated code?",
                "The generated code passed basic AST checks, but it is still trusted local Python code. Run it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if decision != QMessageBox.StandardButton.Yes:
                self.set_status("Execution cancelled")
                return

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
                message="Running generated VTK code safely...",
                work=work,
                on_success=self.apply_safe_execution_result,
                success_message="Rendered safe subprocess preview",
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

        def attach_renderer(self, renderer: Any) -> None:
            render_window = self.vtk_widget.GetRenderWindow()
            render_window.GetRenderers().RemoveAllItems()
            render_window.AddRenderer(renderer)
            renderer.ResetCamera()
            render_window.Render()

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

        def show_error(self, title: str, message: str) -> None:
            QMessageBox.critical(self, title, message)
            self.set_status(title)


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
