from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from osp_sos_analyser.env_config import describe_llm_settings, get_llm_settings, load_app_env
from osp_sos_analyser.llm_client import MissingLLMConfiguration


class EnvConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = {
            key: os.environ.pop(key, None)
            for key in (
                "GROQ_API_KEY",
                "OPENAI_API_KEY",
                "GOOGLE_API_KEY",
                "OSP_SOS_MODEL",
                "OSP_SOS_MODEL_PROVIDER",
                "OSP_SOS_SKIP_DOTENV",
                "OSP_SOS_CA_BUNDLE",
                "SSL_CERT_FILE",
                "OSP_SOS_API_KEY",
                "OSP_SOS_DOTENV",
                "REQUESTS_CA_BUNDLE",
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
        self.assertEqual(settings.api_key, "test-key")

    def test_uses_model_exactly_as_written(self) -> None:
        os.environ["OSP_SOS_MODEL"] = "groq/compound-mini"
        os.environ["OSP_SOS_MODEL_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = "test-key"
        settings = get_llm_settings()
        self.assertEqual(settings.model, "groq/compound-mini")
        self.assertEqual(settings.provider, "groq")

    def test_rejects_unknown_provider_without_rewriting(self) -> None:
        os.environ["OSP_SOS_MODEL"] = "gemini-2.5-pro"
        os.environ["OSP_SOS_MODEL_PROVIDER"] = "google"
        os.environ["GOOGLE_API_KEY"] = "test-key"
        with self.assertRaises(MissingLLMConfiguration) as ctx:
            get_llm_settings()
        self.assertIn("Unsupported OSP_SOS_MODEL_PROVIDER", str(ctx.exception))

    def test_unified_api_key(self) -> None:
        os.environ["OSP_SOS_MODEL"] = "llama-3.3-70b-versatile"
        os.environ["OSP_SOS_MODEL_PROVIDER"] = "groq"
        os.environ["OSP_SOS_API_KEY"] = "unified-secret"
        settings = get_llm_settings()
        self.assertEqual(settings.provider, "groq")
        self.assertEqual(settings.api_key_env, "GROQ_API_KEY")
        self.assertEqual(settings.api_key, "unified-secret")
        self.assertEqual(os.environ.get("GROQ_API_KEY"), "unified-secret")

    def test_empty_override_keeps_env_values(self) -> None:
        os.environ["OSP_SOS_MODEL"] = "openai/gpt-oss-120b"
        os.environ["OSP_SOS_MODEL_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = "test-key"
        settings = get_llm_settings(model="", model_provider="")
        self.assertEqual(settings.model, "openai/gpt-oss-120b")
        self.assertEqual(settings.provider, "groq")

    def test_dotenv_llm_keys_override_stale_process_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv = Path(tmpdir) / ".env"
            dotenv.write_text(
                "OSP_SOS_MODEL_PROVIDER=groq\n"
                'OSP_SOS_MODEL="groq/compound-mini"\n'
                "GROQ_API_KEY=from-file\n",
                encoding="utf-8",
            )
            os.environ.pop("OSP_SOS_SKIP_DOTENV", None)
            os.environ["OSP_SOS_DOTENV"] = str(dotenv)
            os.environ["OSP_SOS_MODEL"] = "openai/gpt-oss-120b"
            os.environ["OSP_SOS_MODEL_PROVIDER"] = "groq"
            os.environ["GROQ_API_KEY"] = "stale-shell-key"

            load_app_env()
            settings = get_llm_settings()
            self.assertEqual(settings.model, "groq/compound-mini")
            self.assertEqual(settings.api_key, "from-file")
            self.assertIn("groq/compound-mini", describe_llm_settings())

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = Path(tmpdir) / "company-ca.pem"
            bundle.write_text("-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n")
            os.environ["OSP_SOS_CA_BUNDLE"] = str(bundle)
            os.environ.pop("OSP_SOS_SKIP_DOTENV", None)
            load_app_env()
            self.assertEqual(os.environ["SSL_CERT_FILE"], str(bundle))
            self.assertEqual(os.environ["REQUESTS_CA_BUNDLE"], str(bundle))


if __name__ == "__main__":
    unittest.main()
