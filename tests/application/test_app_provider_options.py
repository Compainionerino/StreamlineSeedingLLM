from __future__ import annotations

import unittest

from app.app import LLM_PROVIDER_BY_ID


class AppProviderOptionsTests(unittest.TestCase):
    def test_blablador_switch_defaults_to_coding_alias(self) -> None:
        blablador = LLM_PROVIDER_BY_ID["blablador"]

        self.assertEqual(blablador["default_model"], "alias-code")
        self.assertEqual(blablador["always_use_default_on_switch"], "true")


if __name__ == "__main__":
    unittest.main()
