# env_config.py — load LLM settings exclusively from the environment / .env file.
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .llm_client import MissingLLMConfiguration

# Provider aliases people commonly put in .env → LangChain init_chat_model id.
_PROVIDER_ALIASES = {
    "google": "google_genai",
    "gemini": "google_genai",
    "google_genai": "google_genai",
    "google_vertexai": "google_vertexai",
    "grok": "groq",
    "groq": "groq",
    "openai": "openai",
}

_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM configuration taken from .env / process environment."""

    model: str
    provider: str
    api_key_env: str


def configure_tls_trust() -> None:
    """
    Make outbound HTTPS trust the corporate/OS certificate store.

    Prefer an explicit PEM via ``OSP_SOS_CA_BUNDLE``. Otherwise inject the OS
    trust store (Windows enterprise roots) so TLS inspection CAs work without
    disabling verification.
    """
    ca_bundle = os.getenv("OSP_SOS_CA_BUNDLE", "").strip()
    if ca_bundle:
        bundle_path = Path(ca_bundle).expanduser()
        if not bundle_path.is_file():
            raise MissingLLMConfiguration(
                f"OSP_SOS_CA_BUNDLE does not point to a readable PEM file: {bundle_path}"
            )
        path = str(bundle_path)
        os.environ["SSL_CERT_FILE"] = path
        os.environ["REQUESTS_CA_BUNDLE"] = path
        return

    use_system = os.getenv("OSP_SOS_USE_SYSTEM_CERTS", "1").strip().lower()
    if use_system in {"0", "false", "no", "off"}:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 - TLS helper must never block startup
        pass


def load_app_env(*, override: bool = False) -> None:
    """Load `.env` into os.environ unless OSP_SOS_SKIP_DOTENV=1."""
    if os.getenv("OSP_SOS_SKIP_DOTENV") == "1":
        configure_tls_trust()
        return
    from dotenv import load_dotenv

    load_dotenv(override=override)
    # Common typo / alternate name for Groq.
    grok = os.getenv("GROK_API_KEY")
    if grok and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = grok
    configure_tls_trust()


def normalize_provider(model_provider: str) -> str:
    raw = (model_provider or "").strip().lower().replace("-", "_")
    return _PROVIDER_ALIASES.get(raw, raw)


def api_key_env_for_provider(model_provider: str) -> str | None:
    return _PROVIDER_API_KEY_ENV.get(normalize_provider(model_provider))


def clear_llm_settings_cache() -> None:
    """Kept for API compatibility with tests; settings are resolved fresh each call."""


def get_llm_settings(
    *,
    model: str | None = None,
    model_provider: str | None = None,
) -> LLMSettings:
    """
    Resolve model + provider + API key env var from `.env`.

    Optional ``model`` / ``model_provider`` arguments only override when the
    caller explicitly passes non-empty values (CLI). Empty UI fields do not
    replace `.env`. There are no hardcoded model/provider defaults in code.
    """
    load_app_env()

    provider_raw = (model_provider or "").strip() or os.getenv("OSP_SOS_MODEL_PROVIDER", "").strip()
    model_raw = (model or "").strip() or os.getenv("OSP_SOS_MODEL", "").strip()

    if not provider_raw:
        raise MissingLLMConfiguration(
            "OSP_SOS_MODEL_PROVIDER is not set.\n\n"
            "Add it to your `.env` file, for example:\n"
            "  OSP_SOS_MODEL_PROVIDER=google_genai\n"
            "  OSP_SOS_MODEL=gemini-2.5-pro\n"
            "  GOOGLE_API_KEY=...\n"
        )
    if not model_raw:
        raise MissingLLMConfiguration(
            "OSP_SOS_MODEL is not set.\n\n"
            "Add it to your `.env` file next to OSP_SOS_MODEL_PROVIDER, for example:\n"
            "  OSP_SOS_MODEL=gemini-2.5-pro\n"
            "  OSP_SOS_MODEL_PROVIDER=google_genai\n"
        )

    provider = normalize_provider(provider_raw)
    api_key_env = api_key_env_for_provider(provider)
    if not api_key_env:
        raise MissingLLMConfiguration(
            f"Unsupported OSP_SOS_MODEL_PROVIDER={provider_raw!r}.\n"
            "Use one of: google_genai, groq, openai."
        )
    if not os.getenv(api_key_env):
        raise MissingLLMConfiguration(
            f"Provider '{provider}' requires {api_key_env} in your `.env` file.\n"
            f"Set {api_key_env}=... alongside OSP_SOS_MODEL and OSP_SOS_MODEL_PROVIDER."
        )

    _validate_model_provider_match(model_raw, provider)
    return LLMSettings(model=model_raw, provider=provider, api_key_env=api_key_env)


def describe_llm_settings() -> str:
    """Human-readable one-liner for the chat UI (never prints the API key)."""
    try:
        settings = get_llm_settings()
    except MissingLLMConfiguration as exc:
        return f"LLM: not configured — {exc}"
    key_set = "yes" if os.getenv(settings.api_key_env) else "no"
    return (
        f"LLM from `.env`: provider=`{settings.provider}` · "
        f"model=`{settings.model}` · {settings.api_key_env}=set({key_set})"
    )


def _validate_model_provider_match(model: str, provider: str) -> None:
    lower = (model or "").strip().lower()
    looks_like_groq = lower.startswith("openai/") or lower.startswith("meta-llama/")
    looks_like_gemini = lower.startswith("gemini")
    looks_like_openai_api = (
        lower.startswith("gpt-") or lower.startswith("o1") or lower.startswith("o3")
    )

    if provider == "google_genai" and (looks_like_groq or looks_like_openai_api):
        raise MissingLLMConfiguration(
            f"Model '{model}' is not a Google Gemini model, but "
            f"OSP_SOS_MODEL_PROVIDER={provider}.\n\n"
            "Fix your `.env` so provider and model match, for example:\n"
            "  OSP_SOS_MODEL_PROVIDER=google_genai\n"
            "  OSP_SOS_MODEL=gemini-2.5-pro\n"
            "  GOOGLE_API_KEY=...\n"
        )
    if provider == "groq" and looks_like_gemini:
        raise MissingLLMConfiguration(
            f"Model '{model}' looks like Gemini, but provider is '{provider}'. "
            "Set OSP_SOS_MODEL_PROVIDER=google_genai or use a Groq model id in `.env`."
        )
    if provider == "openai" and (looks_like_groq or looks_like_gemini):
        raise MissingLLMConfiguration(
            f"Model '{model}' does not match provider '{provider}'. "
            "Use an OpenAI model id in OSP_SOS_MODEL or change OSP_SOS_MODEL_PROVIDER."
        )
