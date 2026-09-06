from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.experiment_analysis import build_experiment_report
from rag.experiment_editor import ExperimentEditError, update_session_experiment


class ExperimentAnalysisTests(unittest.TestCase):
    def test_report_normalizes_runs_and_builds_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sessions = root / "local" / "sessions"
            dataset_dir = sessions / "datasetA"
            dataset_dir.mkdir(parents=True)
            image = dataset_dir / "datasetA_alias-code_feature_aware_rag_20260819_100000_viewport.png"
            image.write_bytes(b"fake-png")

            self._write_session(
                dataset_dir / "datasetA_alias-code_feature_aware_no_rag_20260819_090000.json",
                provider="blablador",
                model_name="alias-code",
                rag_enabled=False,
                succeeded=True,
                features=2,
                seeding=5,
            )
            self._write_session(
                dataset_dir / "datasetA_alias-code_feature_aware_rag_20260819_100000.json",
                provider="blablador",
                model_name="alias-code",
                rag_enabled=True,
                succeeded=True,
                features=4,
                seeding=8,
                image_path=image.name,
            )
            self._write_session(
                sessions / "last_session.json",
                provider="blablador",
                model_name="alias-code",
                rag_enabled=True,
                features=9,
                seeding=9,
            )
            backup_dir = sessions / "_backups" / "datasetA"
            backup_dir.mkdir(parents=True)
            self._write_session(
                backup_dir / "datasetA_alias-code_feature_aware_rag_backup.json",
                provider="blablador",
                model_name="alias-code",
                rag_enabled=True,
                features=99,
                seeding=99,
            )

            report = build_experiment_report(
                sessions,
                root / "local" / "reports" / "experiment_dashboard",
                repo_root=root,
            )

            self.assertEqual(report["totals"]["raw_runs"], 2)
            self.assertEqual(report["totals"]["primary_conditions"], 2)
            self.assertEqual(report["totals"]["rag_pairs"], 1)
            self.assertEqual(report["rag_pairs"][0]["delta_success"], 0)
            self.assertEqual(report["rag_pairs"][0]["success_effect"], "same")
            self.assertEqual(report["rag_pairs"][0]["delta_first_try_success"], 0)
            self.assertEqual(report["rag_pairs"][0]["first_try_effect"], "same")
            self.assertEqual(report["rag_pairs"][0]["rag_on_strategy_category"], "strategy_used")
            self.assertEqual(report["rag_pairs"][0]["rag_on_strategy_label"], "RAG: strategy used")
            self.assertEqual(report["rag_pairs"][0]["delta_features"], 2.0)
            self.assertEqual(report["rag_pairs"][0]["delta_seeding"], 3.0)
            self.assertTrue(report["records"][0]["first_try_success"])
            self.assertEqual(report["records"][0]["rag_strategy_category"], "no_rag")
            self.assertEqual(report["records"][0]["model"], "alias-code")

            output = root / "local" / "reports" / "experiment_dashboard"
            self.assertTrue((output / "index.html").exists())
            self.assertTrue((output / "dashboard-data.js").exists())
            self.assertTrue((output / "data" / "runs.csv").exists())
            self.assertTrue((output / "data" / "rag_pairs.csv").exists())

    def test_latest_primary_marks_duplicate_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sessions = root / "local" / "sessions"
            dataset_dir = sessions / "datasetA"
            dataset_dir.mkdir(parents=True)

            self._write_session(
                dataset_dir / "datasetA_openai_gpt-5.6-terra_explorative_rag_20260819_090000.json",
                provider="openai",
                model="gpt-5.6-terra",
                rag_enabled=True,
                saved_at="2026-08-19T07:00:00Z",
                features=1,
                seeding=2,
            )
            self._write_session(
                dataset_dir / "datasetA_openai_gpt-5.6-terra_explorative_rag_20260819_100000.json",
                provider="openai",
                model="gpt-5.6-terra",
                rag_enabled=True,
                saved_at="2026-08-19T08:00:00Z",
                features=3,
                seeding=7,
            )

            report = build_experiment_report(
                sessions,
                root / "local" / "reports" / "experiment_dashboard",
                repo_root=root,
            )

            self.assertEqual(report["totals"]["raw_runs"], 2)
            self.assertEqual(report["totals"]["primary_conditions"], 1)
            self.assertEqual(report["totals"]["duplicate_conditions"], 1)
            self.assertEqual(report["primary_records"][0]["features_recognized"], 3)

    def test_update_session_experiment_writes_allowed_fields_and_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sessions = root / "local" / "sessions"
            dataset_dir = sessions / "datasetA"
            session_path = dataset_dir / "datasetA_alias-code_explorative_rag_20260819_090000.json"
            dataset_dir.mkdir(parents=True)
            self._write_session(
                session_path,
                provider="blablador",
                model_name="alias-code",
                rag_enabled=True,
                succeeded=True,
                features=1,
                seeding=2,
            )

            result = update_session_experiment(
                "local/sessions/datasetA/datasetA_alias-code_explorative_rag_20260819_090000.json",
                {
                    "succeeded": False,
                    "attempts": 4,
                    "features_recognized": 3,
                    "feature_notes": "corrected features",
                    "colormap_used": False,
                    "suggested_seeding_used": None,
                    "seeding_score": 6.5,
                    "seeding_notes": "corrected seeding",
                },
                sessions,
                repo_root=root,
            )

            payload = json.loads(session_path.read_text(encoding="utf-8"))
            experiment = payload["experiment"]
            self.assertFalse(experiment["succeeded"])
            self.assertEqual(experiment["attempts"], 4)
            self.assertEqual(experiment["features_recognized"], 3)
            self.assertEqual(experiment["feature_notes"], "corrected features")
            self.assertFalse(experiment["colormap_used"])
            self.assertIsNone(experiment["suggested_seeding_used"])
            self.assertEqual(experiment["seeding_score"], 6.5)
            self.assertEqual(experiment["seeding_notes"], "corrected seeding")
            self.assertEqual(result["updated_fields"], sorted(result["updated_fields"]))
            self.assertTrue((root / result["backup_rel_path"]).exists())

    def test_update_session_experiment_rejects_non_experiment_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sessions = root / "local" / "sessions"
            dataset_dir = sessions / "datasetA"
            session_path = dataset_dir / "datasetA_alias-code_explorative_rag_20260819_090000.json"
            dataset_dir.mkdir(parents=True)
            self._write_session(
                session_path,
                provider="blablador",
                model_name="alias-code",
                rag_enabled=True,
                features=1,
                seeding=2,
            )

            with self.assertRaises(ExperimentEditError):
                update_session_experiment(
                    session_path,
                    {"model": "other-model"},
                    sessions,
                    repo_root=root,
                )

    def _write_session(
        self,
        path: Path,
        *,
        provider: str,
        rag_enabled: bool,
        model: str | None = None,
        model_name: str | None = None,
        succeeded: bool = True,
        features: int = 0,
        seeding: int = 0,
        image_path: str | None = None,
        saved_at: str = "2026-08-19T08:00:00Z",
    ) -> None:
        llm_settings = {"provider": provider}
        if model is not None:
            llm_settings["model"] = model
        if model_name is not None:
            llm_settings["model_name"] = model_name

        payload = {
            "dataset_path": "C:/data/datasetA.vti",
            "saved_at": saved_at,
            "llm_settings": llm_settings,
            "request": {
                "query_mode": "feature aware" if "feature_aware" in path.stem else "explorative",
                "visualization_goal": "Show streamlines",
                "target_feature": "Main flow feature",
            },
            "rag": {"enabled": rag_enabled},
            "experiment": {
                "attempts": 1,
                "colormap_used": True,
                "feature_notes": "- Main feature visible",
                "features_recognized": features,
                "seeding_notes": "good coverage",
                "seeding_score": seeding,
                "succeeded": succeeded,
                "suggested_seeding_used": rag_enabled,
                "viewport_image": {"path": image_path} if image_path else None,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
