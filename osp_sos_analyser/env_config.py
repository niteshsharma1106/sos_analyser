# env_config.py — load LLM settings from `.env`, then init_chat_model.
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .llm_client import MissingLLMConfiguration

# Which env var holds the API key for a given OSP_SOS_MODEL_PROVIDER value.
_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}

_UNIFIED_API_KEY_ENV = "OSP_SOS_API_KEY"

# These keys are always taken from the project `.env` when present (source of truth).
_LLM_DOTENV_KEYS = (
    "OSP_SOS_MODEL",
    "OSP_SOS_MODEL_PROVIDER",
    "OSP_SOS_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "OLLAMA_API_KEY",
)


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM configuration taken from `.env` / process environment."""

    model: str
    provider: str
    api_key_env: str
    api_key: str


def _use_system_certs() -> bool:
    return os.getenv("OSP_SOS_USE_SYSTEM_CERTS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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

    if not _use_system_certs():
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 - TLS helper must never block startup
        pass


def build_httpx_client():
    """
    httpx client that verifies TLS with the corporate/OS trust store.

    ``httpx`` defaults to certifi and ignores Windows enterprise roots; Groq and
    OpenAI SDKs use httpx, so callers must pass this client explicitly.
    """
    import httpx

    configure_tls_trust()
    ca_bundle = (
        os.getenv("OSP_SOS_CA_BUNDLE", "").strip()
        or os.getenv("SSL_CERT_FILE", "").strip()
    )
    if ca_bundle:
        return httpx.Client(verify=ca_bundle, timeout=60.0)
    if _use_system_certs():
        try:
            import ssl

            import truststore

            return httpx.Client(
                verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                timeout=60.0,
            )
        except Exception:  # noqa: BLE001 - fall back to default verify
            pass
    return httpx.Client(timeout=60.0)


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
    key values from the project `.env` always win over stale shell/User env vars.
    """
    if os.getenv("OSP_SOS_SKIP_DOTENV") == "1":
        _apply_unified_api_key()
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
        load_dotenv(override=override)
    _apply_unified_api_key()
    configure_tls_trust()


def _apply_unified_api_key() -> None:
    """If OSP_SOS_API_KEY is set and the provider key is empty, copy it over."""
    unified = os.getenv(_UNIFIED_API_KEY_ENV, "").strip()
    if not unified:
        return
    provider = (os.getenv("OSP_SOS_MODEL_PROVIDER") or "").strip()
    api_key_env = api_key_env_for_provider(provider) if provider else None
    if api_key_env and not (os.getenv(api_key_env) or "").strip():
        os.environ[api_key_env] = unified


def api_key_env_for_provider(model_provider: str) -> str | None:
    """Return the API-key env var name for an exact provider id (no rewriting)."""
    return _PROVIDER_API_KEY_ENV.get((model_provider or "").strip())


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
        or unified OSP_SOS_API_KEY. A local Ollama server does not need a key.

    Values are used as written — no provider/model rewriting.
    Optional ``model`` / ``model_provider`` only override when non-empty (CLI).
    """
    load_app_env()

    provider = (model_provider or "").strip() or (os.getenv("OSP_SOS_MODEL_PROVIDER") or "").strip()
    model_id = (model or "").strip() or (os.getenv("OSP_SOS_MODEL") or "").strip()

    if not provider:
        raise MissingLLMConfiguration(
            "OSP_SOS_MODEL_PROVIDER is not set.\n\n"
            "Add it to your `.env` file, for example:\n"
            "  OSP_SOS_MODEL_PROVIDER=google_genai\n"
            "  OSP_SOS_MODEL=gemini-2.5-pro\n"
            "  GOOGLE_API_KEY=...\n"
        )
    if not model_id:
        raise MissingLLMConfiguration(
            "OSP_SOS_MODEL is not set.\n\n"
            "Add it to your `.env` file next to OSP_SOS_MODEL_PROVIDER, for example:\n"
            "  OSP_SOS_MODEL=gemini-2.5-pro\n"
            "  OSP_SOS_MODEL_PROVIDER=google_genai\n"
        )

    api_key_env = api_key_env_for_provider(provider)
    if not api_key_env:
        raise MissingLLMConfiguration(
            f"Unsupported OSP_SOS_MODEL_PROVIDER={provider!r}.\n"
            "Use one of: google_genai, groq, openai."
        )

    _apply_unified_api_key()
    api_key = (os.getenv(api_key_env) or "").strip()
    if not api_key and provider != "ollama":
        raise MissingLLMConfiguration(
            f"Provider '{provider}' requires {api_key_env} in your `.env` file "
            f"(or set {_UNIFIED_API_KEY_ENV}).\n"
            f"Set {api_key_env}=... alongside OSP_SOS_MODEL and OSP_SOS_MODEL_PROVIDER."
        )

    return LLMSettings(
        model=model_id,
        provider=provider,
        api_key_env=api_key_env,
        api_key=api_key,
    )


def init_chat_model_from_env(
    *,
    model: str | None = None,
    model_provider: str | None = None,
):
    """Load `.env` settings and pass them straight into ``init_chat_model``."""
    from langchain.chat_models import init_chat_model

    settings = get_llm_settings(model=model, model_provider=model_provider)
    kwargs: dict = {
        "model": settings.model,
        "model_provider": settings.provider,
    }
    # Local Ollama normally runs without authentication. Avoid passing an
    # explicit empty key because provider adapters may treat it as a credential.
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    if settings.provider in {"groq", "openai"}:
        kwargs["http_client"] = build_httpx_client()
    return init_chat_model(**kwargs)


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
