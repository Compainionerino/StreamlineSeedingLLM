from __future__ import annotations

import unittest

from app.app import DEFAULT_LLM_PROVIDER_ID, LLM_PROVIDER_BY_ID


class AppProviderOptionsTests(unittest.TestCase):
    def test_openai_switch_defaults_to_terra(self) -> None:
        openai = LLM_PROVIDER_BY_ID["openai"]

        self.assertEqual(openai["default_model"], "openai/gpt-5.6-terra")

    def test_gemini_switch_defaults_to_current_flash(self) -> None:
        gemini = LLM_PROVIDER_BY_ID["gemini"]

        self.assertEqual(gemini["default_model"], "gemini/gemini-3.7-flash")

    def test_anthropic_opus_5_is_default_provider_model(self) -> None:
        anthropic = LLM_PROVIDER_BY_ID["anthropic"]

        self.assertEqual(DEFAULT_LLM_PROVIDER_ID, "anthropic")
        self.assertEqual(anthropic["default_model"], "anthropic/claude-opus-5")

    def test_blablador_switch_defaults_to_coding_alias(self) -> None:
        blablador = LLM_PROVIDER_BY_ID["blablador"]

        self.assertEqual(blablador["default_model"], "alias-code")
        self.assertEqual(blablador["always_use_default_on_switch"], "true")


if __name__ == "__main__":
    unittest.main()
