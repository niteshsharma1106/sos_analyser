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

# Groq Compound systems require the ``groq/`` prefix as part of the model id.
_GROQ_MODEL_ALIASES = {
    "compound": "groq/compound",
    "compound-mini": "groq/compound-mini",
    "compound_mini": "groq/compound-mini",
}

# Optional unified key name; copied onto the provider-specific env var when unset.
_UNIFIED_API_KEY_ENV = "OSP_SOS_API_KEY"

# These keys are always taken from the project `.env` when present (source of truth).
_LLM_DOTENV_KEYS = (
    "OSP_SOS_MODEL",
    "OSP_SOS_MODEL_PROVIDER",
    "OSP_SOS_API_KEY",
    "GROQ_API_KEY",
    "GROK_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
)


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM configuration taken from `.env` / process environment."""

    model: str
    provider: str
    api_key_env: str
    api_key: str


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


def resolve_dotenv_path() -> Path | None:
    """Locate `.env`, preferring repo root over CWD when both exist."""
    custom = os.getenv("OSP_SOS_DOTENV", "").strip()
    candidates: list[Path] = []
    if custom:
        candidates.append(Path(custom).expanduser())
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / ".env")
    candidates.append(Path.cwd() / ".env")
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _strip_env_value(value: str | None) -> str:
    text = (value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def load_app_env(*, override: bool = False) -> None:
    """Load `.env` into os.environ unless OSP_SOS_SKIP_DOTENV=1.

    Non-LLM keys use python-dotenv's normal merge rules. LLM model/provider/API
    key values from the project `.env` always win over stale shell/User env vars,
    because those fields must come from `.env`.
    """
    if os.getenv("OSP_SOS_SKIP_DOTENV") == "1":
        _normalize_api_key_aliases()
        configure_tls_trust()
        return

    from dotenv import dotenv_values, load_dotenv

    dotenv_path = resolve_dotenv_path()
    if dotenv_path is not None:
        load_dotenv(dotenv_path, override=override)
        file_values = dotenv_values(dotenv_path)
        for key in _LLM_DOTENV_KEYS:
            if key not in file_values:
                continue
            raw = file_values.get(key)
            if raw is None:
                continue
            text = _strip_env_value(str(raw))
            if text:
                os.environ[key] = text
    else:
        # Fallback: python-dotenv CWD / parent search.
        load_dotenv(override=override)
    _normalize_api_key_aliases()
    configure_tls_trust()


def _normalize_api_key_aliases() -> None:
    """Accept common alternate env names and map them onto provider keys."""
    # Common typo / alternate name for Groq.
    grok = os.getenv("GROK_API_KEY", "").strip()
    if grok and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = grok

    unified = os.getenv(_UNIFIED_API_KEY_ENV, "").strip()
    if not unified:
        return
    provider = normalize_provider(os.getenv("OSP_SOS_MODEL_PROVIDER", ""))
    api_key_env = api_key_env_for_provider(provider) if provider else None
    if api_key_env and not os.getenv(api_key_env):
        os.environ[api_key_env] = unified


def normalize_model_id(model: str, provider: str) -> str:
    """Normalize known provider-specific model id aliases from `.env`."""
    raw = (model or "").strip()
    if not raw:
        return raw
    if provider == "groq":
        lower = raw.lower()
        if lower in _GROQ_MODEL_ALIASES:
            return _GROQ_MODEL_ALIASES[lower]
        # Accept accidental compound-mini without prefix.
        if lower.startswith("compound") and not lower.startswith("groq/"):
            return f"groq/{raw}"
    return raw


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
    Resolve model + provider + API key from `.env`.

    Required `.env` keys:
      - OSP_SOS_MODEL
      - OSP_SOS_MODEL_PROVIDER
      - provider API key (GOOGLE_API_KEY / GROQ_API_KEY / OPENAI_API_KEY)
        or unified OSP_SOS_API_KEY

    Optional ``model`` / ``model_provider`` arguments only override when the
    caller explicitly passes non-empty values (CLI). Empty UI fields do not
    replace `.env`. There are no hardcoded model/provider/api-key defaults.
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
    model_raw = normalize_model_id(model_raw, provider)
    api_key_env = api_key_env_for_provider(provider)
    if not api_key_env:
        raise MissingLLMConfiguration(
            f"Unsupported OSP_SOS_MODEL_PROVIDER={provider_raw!r}.\n"
            "Use one of: google_genai, groq, openai."
        )

    # Re-apply alias mapping now that provider is known (covers OSP_SOS_API_KEY).
    _normalize_api_key_aliases()
    api_key = (os.getenv(api_key_env) or "").strip()
    if not api_key:
        raise MissingLLMConfiguration(
            f"Provider '{provider}' requires {api_key_env} in your `.env` file "
            f"(or set {_UNIFIED_API_KEY_ENV}).\n"
            f"Set {api_key_env}=... alongside OSP_SOS_MODEL and OSP_SOS_MODEL_PROVIDER."
        )

    _validate_model_provider_match(model_raw, provider)
    return LLMSettings(
        model=model_raw,
        provider=provider,
        api_key_env=api_key_env,
        api_key=api_key,
    )


def init_chat_model_from_env(
    *,
    model: str | None = None,
    model_provider: str | None = None,
):
    """
    Build a LangChain chat model using only `.env` (plus optional CLI overrides).

    Always passes the API key resolved from `.env` into ``init_chat_model``.
    """
    from langchain.chat_models import init_chat_model

    settings = get_llm_settings(model=model, model_provider=model_provider)
    return init_chat_model(
        model=settings.model,
        model_provider=settings.provider,
        api_key=settings.api_key,
    )


def describe_llm_settings() -> str:
    """Human-readable one-liner for the chat UI (never prints the API key)."""
    try:
        settings = get_llm_settings()
    except MissingLLMConfiguration as exc:
        return f"LLM: not configured — {exc}"
    dotenv_path = resolve_dotenv_path()
    source = str(dotenv_path) if dotenv_path else "process environment"
    return (
        f"LLM from `.env` ({source}): provider=`{settings.provider}` · "
        f"model=`{settings.model}` · {settings.api_key_env}=set(yes)"
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
    if provider == "groq":
        lower_model = (model or "").strip().lower()
        if lower_model in {"compound", "compound-mini", "compound_mini"}:
            # Should already be normalized; keep as defensive guidance.
            raise MissingLLMConfiguration(
                f"Groq Compound model ids must include the groq/ prefix.\n"
                f"Set OSP_SOS_MODEL=groq/compound-mini (not {model!r})."
            )
    if provider == "openai" and (looks_like_groq or looks_like_gemini):
        raise MissingLLMConfiguration(
            f"Model '{model}' does not match provider '{provider}'. "
            "Use an OpenAI model id in OSP_SOS_MODEL or change OSP_SOS_MODEL_PROVIDER."
        )
