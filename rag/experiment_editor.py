from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from rag.experiment_analysis import build_experiment_report


ALLOWED_EXPERIMENT_FIELD_SPECS: dict[str, dict[str, Any]] = {
    "succeeded": {"type": "boolean"},
    "attempts": {"type": "integer", "min": 0, "max": 100000},
    "features_recognized": {"type": "integer", "min": 0, "max": 100000},
    "feature_notes": {"type": "string"},
    "colormap_used": {"type": "boolean"},
    "suggested_seeding_used": {"type": "nullable_boolean"},
    "seeding_score": {"type": "number", "min": 0, "max": 100},
    "seeding_notes": {"type": "string"},
}


class ExperimentEditError(ValueError):
    """Raised when a requested session edit should not be written."""


def read_editable_experiment(
    session_reference: str,
    session_root: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    session_path = resolve_session_path(session_reference, session_root, repo_root=repo_root)
    payload = _load_session_payload(session_path)
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        raise ExperimentEditError("The selected session has no experiment section.")

    return {
        "session_rel_path": _relative_path(session_path, Path(repo_root).resolve()),
        "experiment": {field: experiment.get(field) for field in ALLOWED_EXPERIMENT_FIELD_SPECS},
    }


def update_session_experiment(
    session_reference: str,
    updates: Mapping[str, Any],
    session_root: str | Path,
    *,
    repo_root: str | Path = ".",
    backup_root: str | Path | None = None,
) -> dict[str, Any]:
    session_root_path = _resolved_path(session_root, Path(repo_root).resolve())
    repo_root_path = Path(repo_root).resolve()
    session_path = resolve_session_path(session_reference, session_root_path, repo_root=repo_root_path)
    payload = _load_session_payload(session_path)
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        raise ExperimentEditError("The selected session has no experiment section.")

    normalized_updates = normalize_experiment_updates(updates)
    if not normalized_updates:
        raise ExperimentEditError("No editable experiment fields were provided.")

    backup_base = Path(backup_root) if backup_root is not None else session_root_path / "_backups"
    if not backup_base.is_absolute():
        backup_base = repo_root_path / backup_base
    backup_path = _create_backup(session_path, session_root_path, backup_base.resolve())

    experiment.update(normalized_updates)
    temp_path = session_path.with_name(session_path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(session_path)

    return {
        "session_rel_path": _relative_path(session_path, repo_root_path),
        "backup_rel_path": _relative_path(backup_path, repo_root_path),
        "updated_fields": sorted(normalized_updates),
    }


def normalize_experiment_updates(updates: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(updates, Mapping):
        raise ExperimentEditError("Experiment updates must be a JSON object.")

    unknown = sorted(set(updates) - set(ALLOWED_EXPERIMENT_FIELD_SPECS))
    if unknown:
        raise ExperimentEditError(f"Only experiment fields can be edited here: {', '.join(unknown)}.")

    normalized: dict[str, Any] = {}
    for field, value in updates.items():
        field_type = ALLOWED_EXPERIMENT_FIELD_SPECS[field]["type"]
        if field_type == "boolean":
            normalized[field] = _strict_bool(value, field)
        elif field_type == "nullable_boolean":
            normalized[field] = None if value is None or value == "" else _strict_bool(value, field)
        elif field_type == "integer":
            normalized[field] = _strict_int(value, field, ALLOWED_EXPERIMENT_FIELD_SPECS[field])
        elif field_type == "number":
            normalized[field] = _strict_number(value, field, ALLOWED_EXPERIMENT_FIELD_SPECS[field])
        elif field_type == "string":
            normalized[field] = "" if value is None else str(value)
        else:
            raise ExperimentEditError(f"Unsupported editor field type for {field}.")
    return normalized


def resolve_session_path(
    session_reference: str,
    session_root: str | Path,
    *,
    repo_root: str | Path = ".",
) -> Path:
    if not session_reference:
        raise ExperimentEditError("No session was selected.")

    repo_root_path = Path(repo_root).resolve()
    session_root_path = _resolved_path(session_root, repo_root_path)
    reference = Path(str(session_reference))

    candidates = [reference] if reference.is_absolute() else [repo_root_path / reference, session_root_path / reference]
    for candidate in candidates:
        resolved = candidate.resolve()
        if not _is_relative_to(resolved, session_root_path):
            continue
        if "_backups" in resolved.relative_to(session_root_path).parts:
            raise ExperimentEditError("Backup sessions are read-only.")
        if resolved.name == "last_session.json":
            raise ExperimentEditError("The autosaved last session is read-only in the experiment editor.")
        if resolved.suffix.lower() != ".json":
            raise ExperimentEditError("Only saved session JSON files can be edited.")
        if not resolved.exists():
            raise ExperimentEditError("The selected session JSON file does not exist.")
        return resolved

    raise ExperimentEditError("The selected session must be inside the configured session directory.")


def serve_experiment_editor(
    session_root: str | Path,
    output_dir: str | Path,
    *,
    primary_strategy: str = "latest",
    repo_root: str | Path = ".",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    repo_root_path = Path(repo_root).resolve()
    session_root_path = _resolved_path(session_root, repo_root_path)
    output_dir_path = _resolved_path(output_dir, repo_root_path)
    dashboard_path = output_dir_path / "index.html"
    build_experiment_report(
        session_root_path,
        output_dir_path,
        primary_strategy=primary_strategy,
        repo_root=repo_root_path,
    )

    dashboard_url_path = "/" + dashboard_path.resolve().relative_to(repo_root_path).as_posix()

    class ExperimentEditorServer(ThreadingHTTPServer):
        pass

    ExperimentEditorServer.session_root = session_root_path
    ExperimentEditorServer.output_dir = output_dir_path
    ExperimentEditorServer.repo_root = repo_root_path
    ExperimentEditorServer.primary_strategy = primary_strategy
    ExperimentEditorServer.dashboard_url_path = dashboard_url_path

    class ExperimentEditorHandler(_ExperimentEditorHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(repo_root_path), **kwargs)

    with ExperimentEditorServer((host, port), ExperimentEditorHandler) as server:
        print(f"Experiment editor running at http://{host}:{port}{dashboard_url_path}")
        print("Press Ctrl+C to stop it.")
        server.serve_forever()


class _ExperimentEditorHandler(SimpleHTTPRequestHandler):
    server: Any

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/editor/status":
            self._send_json(
                200,
                {
                    "editor_enabled": True,
                    "allowed_fields": ALLOWED_EXPERIMENT_FIELD_SPECS,
                },
            )
            return

        if parsed.path == "/api/experiment":
            session_reference = parse_qs(parsed.query).get("session", [""])[0]
            try:
                payload = read_editable_experiment(
                    session_reference,
                    self.server.session_root,
                    repo_root=self.server.repo_root,
                )
            except ExperimentEditError as error:
                self._send_json(400, {"error": str(error)})
                return
            self._send_json(200, payload)
            return

        if parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", self.server.dashboard_url_path)
            self.end_headers()
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/experiment":
            self._send_json(404, {"error": "Unknown endpoint."})
            return

        try:
            payload = self._read_json()
            if not isinstance(payload, dict):
                raise ExperimentEditError("Request body must be a JSON object.")
            result = update_session_experiment(
                str(payload.get("session_rel_path") or ""),
                payload.get("experiment") or {},
                self.server.session_root,
                repo_root=self.server.repo_root,
            )
            build_experiment_report(
                self.server.session_root,
                self.server.output_dir,
                primary_strategy=self.server.primary_strategy,
                repo_root=self.server.repo_root,
            )
        except (ExperimentEditError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})
            return
        except OSError as error:
            self._send_json(500, {"error": f"Could not write session JSON: {error}"})
            return

        self._send_json(200, {"ok": True, **result})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _create_backup(session_path: Path, session_root: Path, backup_root: Path) -> Path:
    relative = session_path.relative_to(session_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_dir = backup_root / relative.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    backup_path = target_dir / f"{session_path.stem}__{timestamp}{session_path.suffix}"
    counter = 2
    while backup_path.exists():
        backup_path = target_dir / f"{session_path.stem}__{timestamp}_{counter}{session_path.suffix}"
        counter += 1
    shutil.copy2(session_path, backup_path)
    return backup_path


def _load_session_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ExperimentEditError(f"The selected session JSON is invalid: {error}") from error
    except OSError as error:
        raise ExperimentEditError(f"Could not read selected session JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ExperimentEditError("The selected session JSON must contain an object.")
    return payload


def _strict_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ExperimentEditError(f"{field} must be true or false.")


def _strict_int(value: Any, field: str, spec: Mapping[str, Any]) -> int:
    if isinstance(value, bool):
        raise ExperimentEditError(f"{field} must be an integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ExperimentEditError(f"{field} must be an integer.") from error
    if not number.is_integer():
        raise ExperimentEditError(f"{field} must be an integer.")
    integer = int(number)
    _validate_bounds(integer, field, spec)
    return integer


def _strict_number(value: Any, field: str, spec: Mapping[str, Any]) -> int | float:
    if isinstance(value, bool):
        raise ExperimentEditError(f"{field} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ExperimentEditError(f"{field} must be a number.") from error
    _validate_bounds(number, field, spec)
    return int(number) if number.is_integer() else number


def _validate_bounds(value: int | float, field: str, spec: Mapping[str, Any]) -> None:
    minimum = spec.get("min")
    maximum = spec.get("max")
    if minimum is not None and value < minimum:
        raise ExperimentEditError(f"{field} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ExperimentEditError(f"{field} must be at most {maximum}.")


def _resolved_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _relative_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
