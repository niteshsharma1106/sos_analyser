from __future__ import annotations

import os
import unittest

from osp_sos_analyser.env_config import get_llm_settings
from osp_sos_analyser.llm_client import MissingLLMConfiguration


class EnvConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = {
            key: os.environ.pop(key, None)
            for key in (
                "GROQ_API_KEY",
                "GROK_API_KEY",
                "OPENAI_API_KEY",
                "GOOGLE_API_KEY",
                "OSP_SOS_MODEL",
                "OSP_SOS_MODEL_PROVIDER",
                "OSP_SOS_SKIP_DOTENV",
            )
        }
        os.environ["OSP_SOS_SKIP_DOTENV"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("OSP_SOS_SKIP_DOTENV", None)
        for key, value in self._previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_requires_provider_from_env(self) -> None:
        os.environ["OSP_SOS_MODEL"] = "gemini-2.5-pro"
        os.environ["GOOGLE_API_KEY"] = "test-key"
        with self.assertRaises(MissingLLMConfiguration) as ctx:
            get_llm_settings()
        self.assertIn("OSP_SOS_MODEL_PROVIDER", str(ctx.exception))

    def test_requires_model_from_env(self) -> None:
        os.environ["OSP_SOS_MODEL_PROVIDER"] = "google_genai"
        os.environ["GOOGLE_API_KEY"] = "test-key"
        with self.assertRaises(MissingLLMConfiguration) as ctx:
            get_llm_settings()
        self.assertIn("OSP_SOS_MODEL", str(ctx.exception))

    def test_requires_api_key_from_env(self) -> None:
        os.environ["OSP_SOS_MODEL"] = "gemini-2.5-pro"
        os.environ["OSP_SOS_MODEL_PROVIDER"] = "google_genai"
        with self.assertRaises(MissingLLMConfiguration) as ctx:
            get_llm_settings()
        self.assertIn("GOOGLE_API_KEY", str(ctx.exception))

    def test_loads_model_provider_and_key_from_env(self) -> None:
        os.environ["OSP_SOS_MODEL"] = "gemini-2.5-flash"
        os.environ["OSP_SOS_MODEL_PROVIDER"] = "google_genai"
        os.environ["GOOGLE_API_KEY"] = "test-key"
        settings = get_llm_settings()
        self.assertEqual(settings.model, "gemini-2.5-flash")
        self.assertEqual(settings.provider, "google_genai")
        self.assertEqual(settings.api_key_env, "GOOGLE_API_KEY")

    def test_empty_override_keeps_env_values(self) -> None:
        os.environ["OSP_SOS_MODEL"] = "openai/gpt-oss-120b"
        os.environ["OSP_SOS_MODEL_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = "test-key"
        settings = get_llm_settings(model="", model_provider="")
        self.assertEqual(settings.model, "openai/gpt-oss-120b")
        self.assertEqual(settings.provider, "groq")

    def test_normalizes_google_alias(self) -> None:
        os.environ["OSP_SOS_MODEL"] = "gemini-2.5-pro"
        os.environ["OSP_SOS_MODEL_PROVIDER"] = "google"
        os.environ["GOOGLE_API_KEY"] = "test-key"
        settings = get_llm_settings()
        self.assertEqual(settings.provider, "google_genai")


if __name__ == "__main__":
    unittest.main()
