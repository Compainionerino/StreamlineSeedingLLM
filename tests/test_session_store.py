from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from streamline_app.session_store import SESSION_SCHEMA_VERSION, load_last_session, write_last_session


class SessionStoreTests(unittest.TestCase):
    def test_write_and_load_last_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last_session.json"

            write_last_session({"dataset_path": "sample.vti", "request": {"visualization_goal": "show vortices"}}, path)
            loaded = load_last_session(path)

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["schema_version"], SESSION_SCHEMA_VERSION)
            self.assertIn("saved_at", loaded)
            self.assertEqual(loaded["dataset_path"], "sample.vti")
            self.assertEqual(loaded["request"]["visualization_goal"], "show vortices")

    def test_missing_or_invalid_session_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.json"
            invalid_path = Path(temp_dir) / "invalid.json"
            invalid_path.write_text("{not json", encoding="utf-8")

            self.assertIsNone(load_last_session(missing_path))
            self.assertIsNone(load_last_session(invalid_path))

    def test_non_object_session_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last_session.json"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

            self.assertIsNone(load_last_session(path))


if __name__ == "__main__":
    unittest.main()
