from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag.experiment_analysis import build_experiment_report


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
            self.assertEqual(report["rag_pairs"][0]["delta_features"], 2.0)
            self.assertEqual(report["rag_pairs"][0]["delta_seeding"], 3.0)
            self.assertTrue(report["records"][0]["first_try_success"])
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
