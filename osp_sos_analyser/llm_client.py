from __future__ import annotations

import os
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
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise MissingLLMConfiguration(
                "LLM analysis requires OPENAI_API_KEY. Set it, or run analyze with --offline."
            )

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model or os.getenv("OSP_SOS_MODEL", "gpt-5.5")

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
