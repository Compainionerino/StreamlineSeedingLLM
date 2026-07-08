from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SESSION_ROOT = Path("runs") / "sessions"
LAST_SESSION_PATH = DEFAULT_SESSION_ROOT / "last_session.json"
SESSION_SCHEMA_VERSION = 1


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_last_session(
    session: Mapping[str, Any],
    session_path: str | Path = LAST_SESSION_PATH,
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


def load_last_session(session_path: str | Path = LAST_SESSION_PATH) -> dict[str, Any] | None:
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
