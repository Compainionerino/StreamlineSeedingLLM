from __future__ import annotations

import unittest

from streamline_app.prompting import build_final_prompt
from streamline_app.query import UserRequest, build_retrieval_query, infer_request_defaults_from_metadata


class QueryPromptingTests(unittest.TestCase):
    def test_build_retrieval_query_uses_record_like_fields(self) -> None:
        request = UserRequest(
            visualization_goal="Show vortices with low clutter.",
            target_feature="vortex cores",
            data_dimension="3D",
            data_type="volume flow",
            seeding_behavior="seed around high-vorticity regions",
            density_clutter_preference="sparse and readable",
        )
        metadata = {
            "dataset_type": "ImageData",
            "hints": ["3d", "volume", "vector_field"],
            "point_arrays": [{"name": "velocity", "components": 3}],
        }

        query = build_retrieval_query(request, metadata)

        self.assertIn("application_goal: Show vortices", query)
        self.assertIn("target_feature: vortex cores", query)
        self.assertIn("data_dimension: 3D volume flow", query)
        self.assertIn("seed_placement_strategy: seed around high-vorticity regions", query)
        self.assertIn("velocity(3 components)", query)

    def test_metadata_defaults_infer_dimension_and_type(self) -> None:
        defaults = infer_request_defaults_from_metadata({"hints": ["3d", "surface", "vector_field"]})

        self.assertEqual(defaults["data_dimension"], "3D")
        self.assertEqual(defaults["data_type"], "surface flow")

    def test_prompt_uses_structured_record_fields_not_raw_retrieval_text(self) -> None:
        request = UserRequest(visualization_goal="Generate representative streamlines.")
        retrieval_payload = {
            "results": [
                {
                    "rank": 1,
                    "score": 0.7,
                    "paper_title": "Example",
                    "algorithm_name": "Entropy seeding",
                    "seed_placement_strategy": "Sample entropy field.",
                    "retrieval_text": "THIS SHOULD NOT APPEAR",
                    "dimension_tags": ["3d"],
                    "why_retrieved": ["matched dimension_tags: 3d"],
                }
            ]
        }

        bundle = build_final_prompt(
            user_request=request,
            dataset_metadata={"hints": ["3d"]},
            retrieval_payload=retrieval_payload,
        )

        self.assertIn("Entropy seeding", bundle.final_prompt)
        self.assertIn("seed_placement_strategy: Sample entropy field.", bundle.final_prompt)
        self.assertNotIn("THIS SHOULD NOT APPEAR", bundle.final_prompt)

    def test_prompt_describes_host_qvtk_viewport_contract(self) -> None:
        bundle = build_final_prompt(
            user_request=UserRequest(visualization_goal="Show streamlines."),
            dataset_metadata={},
            retrieval_payload={"results": []},
        )

        self.assertIn("QVTKRenderWindowInteractor viewport", bundle.final_prompt)
        self.assertIn("host process will attach the returned vtkRenderer", bundle.final_prompt)
        self.assertIn("Do not create a vtkRenderWindow", bundle.final_prompt)
        self.assertIn("Do not call Start(), Initialize(), Render()", bundle.final_prompt)


if __name__ == "__main__":
    unittest.main()
