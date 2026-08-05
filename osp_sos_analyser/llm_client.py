from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def structured(self, schema: type[T], system: str, user: str) -> T:
        ...


class MissingLLMConfiguration(RuntimeError):
    """Raised when LLM-backed analysis is requested without credentials."""


class OpenAIResponsesClient:
    def __init__(self, model: str | None = None) -> None:
        # Import locally to avoid circular import with env_config.
        from .env_config import get_llm_settings

        settings = get_llm_settings(model=model, model_provider="openai")

        from openai import OpenAI

        self._client = OpenAI(api_key=settings.api_key)
        self._model = settings.model

    def structured(self, schema: type[T], system: str, user: str) -> T:
        response = self._client.responses.parse(
            model=self._model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text_format=schema,
        )
        return response.output_parsed
