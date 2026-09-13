"""LLM client for OpenRouter and local Ollama models."""

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Tuple

import yaml
from dotenv import find_dotenv, load_dotenv
from ollama import chat as ollama_chat
from openai import AsyncOpenAI

from llms.llm_call_logger import LLMCallLogger
from prompts.prompt_response import Response
from prompts.prompt_templates import PromptManager

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_DEFAULT_MODEL = "gemma4"
OLLAMA_MAX_RETRIES = 2


@dataclass
class Config:
    provider: str = "openrouter"
    base_url: str = OPENROUTER_DEFAULT_BASE_URL
    api_key: str = ""
    model_name: str = "openai/gpt-4o-mini"


def load_config(
    config_path: str | Path = "config/config.yaml",
    *,
    model_name: str | None = None,
    provider: str | None = None,
) -> Config:
    load_dotenv(find_dotenv())

    config_path = Path(config_path)
    llm_cfg: dict = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            llm_cfg = (yaml.safe_load(f) or {}).get("llm", {}) or {}

    resolved_provider = (
        provider
        or os.getenv("LLM_PROVIDER")
        or llm_cfg.get("provider", "openrouter")
    ).lower()

    if resolved_provider == "ollama":
        return Config(
            provider="ollama",
            model_name=(
                model_name
                or os.getenv("OLLAMA_MODEL")
                or llm_cfg.get("model_name", OLLAMA_DEFAULT_MODEL)
            ),
        )

    return Config(
        provider="openrouter",
        base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            llm_cfg.get("base_url", OPENROUTER_DEFAULT_BASE_URL),
        ),
        api_key=os.getenv("OPENROUTER_API_KEY") or llm_cfg.get("api_key", ""),
        model_name=(
            model_name
            or os.getenv("OPENROUTER_MODEL")
            or llm_cfg.get("model_name", "openai/gpt-4o-mini")
        ),
    )


class LLMManager:
    """Route prompt templates to OpenRouter or Ollama backends."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        config_path: str | Path = "config/config.yaml",
        model_name: str | None = None,
        provider: str | None = None,
        call_logger: LLMCallLogger | None = None,
    ):
        self.prompt_manager = prompt_manager
        self.config = load_config(
            config_path,
            model_name=model_name,
            provider=provider,
        )
        self.call_logger = call_logger
        self.client: AsyncOpenAI | None = None

        if self._uses_ollama():
            return

        if not self.config.api_key:
            raise ValueError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY in .env."
            )

        self.client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "multi-turn-attack-defenses"),
            },
        )

    def _uses_ollama(self) -> bool:
        if self.config.provider == "ollama":
            return True
        return "gemma" in self.config.model_name.lower()

    def _ollama_model_name(self) -> str:
        name = self.config.model_name
        return name.rsplit("/", 1)[-1] if "/" in name else name

    async def analyze_conversation(
        self,
        llm_type: str,
        prev_pair: Tuple[str, str],
        current_pair: Tuple[str, str],
        *,
        call_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        template = self.prompt_manager.get_template(llm_type)
        prompt = template.format(prev_pair=prev_pair, current_pair=current_pair)
        return await self._complete(prompt, call_context=call_context)

    async def _complete(
        self,
        prompt: str,
        *,
        call_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if self._uses_ollama():
            model = self._ollama_model_name()

            async def fetch() -> tuple[str, Dict[str, Any] | None]:
                last_error: Exception | None = None
                for attempt in range(OLLAMA_MAX_RETRIES + 1):
                    try:
                        response = await asyncio.to_thread(
                            ollama_chat,
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        content = response.message.content or ""
                        if content.strip():
                            return content, None
                        last_error = ValueError("Ollama returned empty response")
                    except Exception as exc:
                        last_error = exc
                    if attempt < OLLAMA_MAX_RETRIES:
                        await asyncio.sleep(2**attempt)
                raise last_error or ValueError("Ollama returned empty response")
        else:
            model = self.config.model_name

            async def fetch() -> tuple[str, Dict[str, Any] | None]:
                assert self.client is not None
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                content = response.choices[0].message.content or ""
                usage = self._serialize_usage(getattr(response, "usage", None))
                return content, usage

        content = await self._logged_invoke(
            prompt, model, fetch, call_context=call_context
        )
        return self._build_response(content)

    async def _logged_invoke(
        self,
        prompt: str,
        model: str,
        fetch: Callable[[], Awaitable[tuple[str, Dict[str, Any] | None]]],
        *,
        call_context: Dict[str, Any] | None = None,
    ) -> str:
        started = time.perf_counter()
        content = ""
        usage: Dict[str, Any] | None = None
        error: str | None = None

        try:
            content, usage = await fetch()
            return content
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            if self.call_logger:
                self.call_logger.log_call(
                    model=model,
                    prompt=prompt,
                    response=content,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    usage=usage,
                    context=call_context,
                    error=error,
                )

    @staticmethod
    def _build_response(content: str) -> Dict[str, Any]:
        response_out = Response(content=content)
        if hasattr(response_out, "model_dump"):
            return response_out.model_dump()
        return response_out.dict()

    @staticmethod
    def _serialize_usage(usage: Any) -> Dict[str, Any] | None:
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if hasattr(usage, "dict"):
            return usage.dict()
        return dict(usage)
