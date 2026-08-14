from __future__ import annotations

import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from app.llm import (
    BLABLADOR_API_BASE,
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
        self.assertEqual(
            normalize_model_name("alias-code", "blablador"),
            "openai/alias-code",
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

    def test_generate_code_logs_compact_token_usage_without_prompt_or_api_key(self) -> None:
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
                    ],
                    "model": "fake-response-model",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                },
            ]
        )
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)
        output = io.StringIO()

        with redirect_stdout(output):
            response = generate_code(
                "make vtk code",
                LLMSettings(model="fake", api_key="secret", repair_attempts=0),
            )

        self.assertEqual(response.token_usage["total"]["input_tokens"], 10)
        self.assertEqual(response.token_usage["total"]["output_tokens"], 20)
        self.assertEqual(response.token_usage["total"]["total_tokens"], 30)
        printed = output.getvalue()
        self.assertIn("===== LLM USAGE SUMMARY =====", printed)
        self.assertIn("Provider: custom", printed)
        self.assertIn("Requested model: fake", printed)
        self.assertIn("LiteLLM model: fake", printed)
        self.assertIn("API base: not configured", printed)
        self.assertIn("Provider-reported model(s): fake-response-model", printed)
        self.assertIn("Response stack:", printed)
        self.assertIn("1. initial generation", printed)
        self.assertIn("response_model: fake-response-model", printed)
        self.assertIn("finish_reason: stop", printed)
        self.assertIn("input_tokens: 10", printed)
        self.assertIn("output_tokens: 20", printed)
        self.assertIn("total_tokens: 30", printed)
        self.assertNotIn("make vtk code", printed)
        self.assertNotIn("api_key", printed)
        self.assertNotIn("secret", printed)

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

    def test_openai_terra_omits_temperature_parameter(self) -> None:
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
                LLMSettings(model="gpt-5.6-terra", provider="openai", temperature=0.7, repair_attempts=0),
            )

        self.assertEqual(response.model, "openai/gpt-5.6-terra")
        self.assertEqual(fake.calls[0]["model"], "openai/gpt-5.6-terra")
        self.assertNotIn("temperature", fake.calls[0])

    def test_anthropic_opus_omits_temperature_parameter(self) -> None:
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

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-secret"}, clear=True):
            response = generate_code(
                "make vtk code",
                LLMSettings(model="claude-opus-5", provider="anthropic", temperature=0.7, repair_attempts=0),
            )

        self.assertEqual(response.model, "anthropic/claude-opus-5")
        self.assertEqual(fake.calls[0]["model"], "anthropic/claude-opus-5")
        self.assertNotIn("temperature", fake.calls[0])

    def test_gemini_3_models_omit_deprecated_temperature_parameter(self) -> None:
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

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-secret"}, clear=True):
            response = generate_code(
                "make vtk code",
                LLMSettings(model="gemini-3.7-flash", provider="gemini", temperature=0.7, repair_attempts=0),
            )

        self.assertEqual(response.model, "gemini/gemini-3.7-flash")
        self.assertEqual(fake.calls[0]["model"], "gemini/gemini-3.7-flash")
        self.assertNotIn("temperature", fake.calls[0])

    def test_missing_gemini_key_errors_only_for_gemini_model(self) -> None:
        fake = FakeLiteLLM([])
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)

        with patch.dict(os.environ, {}, clear=True), patch("app.llm._windows_persistent_env_var", return_value=""):
            with self.assertRaises(LLMError) as context:
                generate_code(
                    "make vtk code",
                    LLMSettings(model="gemini-test", provider="gemini", repair_attempts=0),
                )

        self.assertIn("Google Gemini API key is required", str(context.exception))
        self.assertIn("gemini/gemini-test", str(context.exception))
        self.assertEqual(fake.calls, [])

    def test_blablador_provider_uses_env_key_and_default_api_base(self) -> None:
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

        with patch.dict(os.environ, {"BLABLADOR_API_KEY": "blablador-secret"}, clear=True):
            response = generate_code(
                "make vtk code",
                LLMSettings(model="alias-code", provider="blablador", repair_attempts=0),
            )

        self.assertEqual(response.model, "openai/alias-code")
        self.assertEqual(fake.calls[0]["model"], "openai/alias-code")
        self.assertEqual(fake.calls[0]["api_key"], "blablador-secret")
        self.assertEqual(fake.calls[0]["api_base"], BLABLADOR_API_BASE)

    def test_blablador_usage_log_shows_provider_and_api_base(self) -> None:
        fake = FakeLiteLLM(
            [
                {
                    "model": "provider-specific-code-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": "import vtk\n\ndef create_visualization(dataset_path, metadata, user_request):\n    return vtk.vtkRenderer()"
                            },
                        }
                    ],
                },
            ]
        )
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)
        output = io.StringIO()

        with patch.dict(os.environ, {"BLABLADOR_API_KEY": "blablador-secret"}, clear=True), redirect_stdout(output):
            generate_code(
                "make vtk code",
                LLMSettings(model="alias-code", provider="blablador", repair_attempts=0),
            )

        printed = output.getvalue()
        self.assertIn("Provider: blablador", printed)
        self.assertIn("Requested model: alias-code", printed)
        self.assertIn("LiteLLM model: openai/alias-code", printed)
        self.assertIn(f"API base: {BLABLADOR_API_BASE}", printed)
        self.assertIn("Provider-reported model(s): provider-specific-code-model", printed)

    def test_blablador_provider_accepts_double_o_env_key_alias(self) -> None:
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

        with patch.dict(os.environ, {"BLABLADOOR_API_KEY": "double-o-secret"}, clear=True), patch(
            "app.llm._windows_persistent_env_var", return_value=""
        ):
            response = generate_code(
                "make vtk code",
                LLMSettings(model="alias-code", provider="blablador", repair_attempts=0),
            )

        self.assertEqual(response.model, "openai/alias-code")
        self.assertEqual(fake.calls[0]["api_key"], "double-o-secret")

    def test_missing_blablador_key_mentions_blablador_env_var(self) -> None:
        fake = FakeLiteLLM([])
        sys.modules["litellm"] = types.SimpleNamespace(completion=fake.completion)

        with patch.dict(os.environ, {}, clear=True), patch("app.llm._windows_persistent_env_var", return_value=""):
            with self.assertRaises(LLMError) as context:
                generate_code(
                    "make vtk code",
                    LLMSettings(model="alias-code", provider="blablador", repair_attempts=0),
                )

        self.assertIn("Blablador API key is required", str(context.exception))
        self.assertIn("openai/alias-code", str(context.exception))
        self.assertIn("BLABLADOR_API_KEY", str(context.exception))
        self.assertIn("BLABLADOOR_API_KEY", str(context.exception))
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
