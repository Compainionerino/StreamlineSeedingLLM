from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

from streamline_app.llm import (
    DEFAULT_MAX_TOKENS,
    MAX_TOKEN_LIMIT,
    LLMError,
    LLMSettings,
    extract_code_block,
    generate_code,
    normalize_model_name,
)


class FakeLiteLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake LiteLLM responses left.")
        return self.responses.pop(0)


class LLMTests(unittest.TestCase):
    def tearDown(self) -> None:
        sys.modules.pop("litellm", None)

    def test_extracts_python_fenced_code(self) -> None:
        content = """Here is the code:

```python
import vtk

def create_visualization(dataset_path, metadata, user_request):
    return vtk.vtkRenderer()
```
"""

        self.assertEqual(
            extract_code_block(content),
            "import vtk\n\ndef create_visualization(dataset_path, metadata, user_request):\n    return vtk.vtkRenderer()",
        )

    def test_returns_raw_text_when_unfenced(self) -> None:
        self.assertEqual(extract_code_block("print('x')"), "print('x')")

    def test_default_token_budget_is_large_enough_for_gui_start_value(self) -> None:
        self.assertEqual(DEFAULT_MAX_TOKENS, 50000)
        self.assertGreaterEqual(MAX_TOKEN_LIMIT, DEFAULT_MAX_TOKENS)

    def test_provider_model_prefix_is_added_when_missing(self) -> None:
        self.assertEqual(
            normalize_model_name("claude-sonnet-example", "anthropic"),
            "anthropic/claude-sonnet-example",
        )
        self.assertEqual(
            normalize_model_name("gemini-example", "gemini"),
            "gemini/gemini-example",
        )
        self.assertEqual(
            normalize_model_name("anthropic/claude-sonnet-example", "anthropic"),
            "anthropic/claude-sonnet-example",
        )
        self.assertEqual(normalize_model_name("local-model", "custom"), "local-model")

    def test_generate_code_passes_default_token_budget_to_litellm(self) -> None:
        fake = FakeLiteLLM(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "import vtk\n\ndef create_visualization(dataset_path, metadata, user_request):\n    return vtk.vtkRenderer()"
                            },
                        }
                    ]
                },
            ]
        )
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)

        generate_code("make vtk code", LLMSettings(model="fake", repair_attempts=0))

        self.assertEqual(fake.calls[0]["max_tokens"], 50000)

    def test_generate_code_passes_provider_model_and_key_to_litellm(self) -> None:
        fake = FakeLiteLLM(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "import vtk\n\ndef create_visualization(dataset_path, metadata, user_request):\n    return vtk.vtkRenderer()"
                            },
                        }
                    ]
                },
            ]
        )
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)

        response = generate_code(
            "make vtk code",
            LLMSettings(
                model="claude-sonnet-example",
                provider="anthropic",
                api_key="secret",
                repair_attempts=0,
            ),
        )

        self.assertEqual(response.model, "anthropic/claude-sonnet-example")
        self.assertEqual(fake.calls[0]["model"], "anthropic/claude-sonnet-example")
        self.assertEqual(fake.calls[0]["api_key"], "secret")

    def test_openai_model_does_not_require_gemini_key(self) -> None:
        fake = FakeLiteLLM(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "import vtk\n\ndef create_visualization(dataset_path, metadata, user_request):\n    return vtk.vtkRenderer()"
                            },
                        }
                    ]
                },
            ]
        )
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-secret"}, clear=True):
            response = generate_code(
                "make vtk code",
                LLMSettings(model="gpt-test", provider="openai", repair_attempts=0),
            )

        self.assertEqual(response.model, "openai/gpt-test")
        self.assertEqual(fake.calls[0]["model"], "openai/gpt-test")
        self.assertNotIn("api_key", fake.calls[0])

    def test_missing_gemini_key_errors_only_for_gemini_model(self) -> None:
        fake = FakeLiteLLM([])
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMError) as context:
                generate_code(
                    "make vtk code",
                    LLMSettings(model="gemini-test", provider="gemini", repair_attempts=0),
                )

        self.assertIn("Google Gemini API key is required", str(context.exception))
        self.assertIn("gemini/gemini-test", str(context.exception))
        self.assertEqual(fake.calls, [])

    def test_generate_code_continues_truncated_response(self) -> None:
        fake = FakeLiteLLM(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "import vtk\n\ndef create_visualization("},
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "dataset_path, metadata, user_request):\n    return vtk.vtkRenderer()"
                            },
                        }
                    ]
                },
            ]
        )
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)

        response = generate_code(
            "make vtk code",
            LLMSettings(model="fake", max_tokens=128, max_continuations=2, repair_attempts=0),
        )

        self.assertEqual(response.continuation_count, 1)
        self.assertEqual(len(fake.calls), 2)
        self.assertIn("return vtk.vtkRenderer()", response.code)
        self.assertIsNone(response.validation_errors)

    def test_generate_code_repairs_validation_failures_twice_at_most(self) -> None:
        fake = FakeLiteLLM(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "import os\nimport vtk\n\ndef create_visualization(dataset_path, metadata, user_request):\n    return vtk.vtkRenderer()"
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "import vtk\n\ndef create_visualization(dataset_path, metadata, user_request):\n    window = vtk.vtkRenderWindow()\n    return vtk.vtkRenderer()"
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "import vtk\n\ndef create_visualization(dataset_path, metadata, user_request):\n    return vtk.vtkRenderer()"
                            },
                        }
                    ]
                },
            ]
        )
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)

        response = generate_code(
            "make vtk code",
            LLMSettings(model="fake", max_tokens=128, max_continuations=0, repair_attempts=2),
        )

        self.assertEqual(response.repair_attempts, 2)
        self.assertEqual(len(fake.calls), 3)
        self.assertIsNone(response.validation_errors)
        self.assertNotIn("vtkRenderWindow", response.code)


if __name__ == "__main__":
    unittest.main()
