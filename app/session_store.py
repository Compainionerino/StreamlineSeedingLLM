from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SESSION_ROOT = Path("local") / "sessions"
LAST_SESSION_PATH = DEFAULT_SESSION_ROOT / "last_session.json"
SESSION_SCHEMA_VERSION = 1


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_session(
    session: Mapping[str, Any],
    session_path: str | Path,
) -> Path:
    path = Path(session_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(session)
    payload["schema_version"] = SESSION_SCHEMA_VERSION
    payload["saved_at"] = _utc_timestamp()

    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)
    return path


def _session_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "session"


def default_manual_session_path(
    dataset_path: str | Path | None,
    *,
    model_name: str | None = None,
    query_mode: str | None = None,
    saved_at: datetime | None = None,
    session_root: str | Path = DEFAULT_SESSION_ROOT,
) -> Path:
    timestamp = (saved_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    dataset_name = ""
    if dataset_path:
        dataset_name = Path(dataset_path).stem
    filename_parts = [_session_filename_part(dataset_name)]
    for value in (model_name, query_mode):
        if value:
            filename_parts.append(_session_filename_part(str(value)))
    name = "_".join(filename_parts)
    return Path(session_root) / f"{name}_{timestamp}.json"


def write_last_session(
    session: Mapping[str, Any],
    session_path: str | Path = LAST_SESSION_PATH,
) -> Path:
    return _write_session(session, session_path)


def write_manual_session(
    session: Mapping[str, Any],
    session_path: str | Path,
) -> Path:
    return _write_session(session, session_path)


def load_session(session_path: str | Path) -> dict[str, Any] | None:
    path = Path(session_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def load_last_session(session_path: str | Path = LAST_SESSION_PATH) -> dict[str, Any] | None:
    return load_session(session_path)
