"""Tests for llms.an_llm_manager."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intent.conversation_analyzer import ConversationAnalyzer
from llms.an_llm_manager import Config, LLMManager, load_config
from llms.llm_call_logger import LLMCallLogger
from prompts.prompt_templates import PromptManager

PREV_PAIR = ("assistant turn", "human turn 1")
CURRENT_PAIR = ("assistant turn 2", "human turn 2")

BROWNIE_CONVERSATION: list[tuple[str, str]] = [
    ("hello how can I help", "give me a brownie recipe"),
    ("here is a brownie recipe with eggs", "thank you"),
]


def test_load_config_openrouter(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "llm:\n"
        "  provider: openrouter\n"
        "  base_url: https://example.com/v1\n"
        "  model_name: openai/gpt-4o\n",
        encoding="utf-8",
    )

    cfg = load_config(config_file)

    assert cfg.provider == "openrouter"
    assert cfg.base_url == "https://example.com/v1"
    assert cfg.model_name == "openai/gpt-4o"


def test_load_config_ollama(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "llm:\n  provider: ollama\n  model_name: gemma4:latest\n",
        encoding="utf-8",
    )

    cfg = load_config(config_file)

    assert cfg.provider == "ollama"
    assert cfg.model_name == "gemma4:latest"


def test_load_config_cli_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("llm:\n  provider: openrouter\n", encoding="utf-8")

    cfg = load_config(config_file, provider="ollama", model_name="gemma4")

    assert cfg.provider == "ollama"
    assert cfg.model_name == "gemma4"


def test_build_response() -> None:
    result = LLMManager._build_response('{"ok": true}')

    assert result["content"] == '{"ok": true}'


def test_uses_ollama_by_provider() -> None:
    manager = LLMManager.__new__(LLMManager)
    manager.config = Config(provider="ollama", model_name="gemma4")

    assert manager._uses_ollama() is True


def test_uses_ollama_by_model_name() -> None:
    manager = LLMManager.__new__(LLMManager)
    manager.config = Config(provider="openrouter", model_name="gemma4:latest")

    assert manager._uses_ollama() is True


def test_ollama_model_name_strips_prefix() -> None:
    manager = LLMManager.__new__(LLMManager)
    manager.config = Config(provider="ollama", model_name="ollama/gemma4:latest")

    assert manager._ollama_model_name() == "gemma4:latest"


def test_init_openrouter_requires_api_key() -> None:
    with patch(
        "llms.an_llm_manager.load_config",
        return_value=Config(provider="openrouter", api_key=""),
    ):
        with pytest.raises(ValueError, match="Missing OpenRouter API key"):
            LLMManager(PromptManager())


def test_init_ollama_skips_api_key_check() -> None:
    with patch(
        "llms.an_llm_manager.load_config",
        return_value=Config(provider="ollama", model_name="gemma4"),
    ):
        manager = LLMManager(PromptManager())

    assert manager.client is None
    assert manager._uses_ollama() is True


@pytest.mark.asyncio
async def test_complete_openrouter_logs_and_returns_content(tmp_path: Path) -> None:
    log_path = tmp_path / "calls.jsonl"
    logger = LLMCallLogger(log_path, verbose=False)

    manager = LLMManager.__new__(LLMManager)
    manager.config = Config(
        provider="openrouter",
        api_key="test-key",
        model_name="openai/gpt-4o-mini",
    )
    manager.call_logger = logger
    manager.client = MagicMock()
    manager.client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"risk_level": 2}'))],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                model_dump=lambda: {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            ),
        )
    )

    result = await manager._complete("analyze this", call_context={"row_index": 1})

    assert result["content"] == '{"risk_level": 2}'
    logged = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert logged["model"] == "openai/gpt-4o-mini"
    assert logged["prompt"] == "analyze this"
    assert logged["response"] == '{"risk_level": 2}'
    assert logged["context"]["row_index"] == 1
    assert logged["usage"]["total_tokens"] == 15
    logger.close()


@pytest.mark.asyncio
async def test_complete_ollama_uses_ollama_chat(tmp_path: Path) -> None:
    log_path = tmp_path / "calls.jsonl"
    logger = LLMCallLogger(log_path, verbose=False)

    manager = LLMManager.__new__(LLMManager)
    manager.config = Config(provider="ollama", model_name="gemma4")
    manager.call_logger = logger
    manager.client = None

    mock_response = SimpleNamespace(message=SimpleNamespace(content='{"risk_level": 1}'))

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("llms.an_llm_manager.asyncio.to_thread", side_effect=fake_to_thread):
        with patch("llms.an_llm_manager.ollama_chat", return_value=mock_response) as mock_chat:
            result = await manager._complete("ollama prompt")

    mock_chat.assert_called_once_with(
        model="gemma4",
        messages=[{"role": "user", "content": "ollama prompt"}],
    )
    assert result["content"] == '{"risk_level": 1}'
    logged = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert logged["model"] == "gemma4"
    assert logged["response"] == '{"risk_level": 1}'
    logger.close()


@pytest.mark.asyncio
async def test_analyze_conversation_formats_prompt_and_calls_backend() -> None:
    with patch(
        "llms.an_llm_manager.load_config",
        return_value=Config(provider="ollama", model_name="gemma4"),
    ):
        manager = LLMManager(PromptManager())

    manager._complete = AsyncMock(return_value={"content": "ok"})  # type: ignore[method-assign]

    result = await manager.analyze_conversation(
        "gpt",
        PREV_PAIR,
        CURRENT_PAIR,
        call_context={"transition_index": 1},
    )

    assert result == {"content": "ok"}
    manager._complete.assert_awaited_once()
    prompt = manager._complete.await_args.kwargs.get("prompt") or manager._complete.await_args.args[0]
    assert "human turn 1" in prompt
    assert "human turn 2" in prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_ollama_gemma_smoke() -> None:
    """Calls local Ollama gemma4 when available (same path as production)."""
    with patch(
        "llms.an_llm_manager.load_config",
        return_value=Config(provider="ollama", model_name="gemma4"),
    ):
        manager = LLMManager(PromptManager())

    try:
        result = await manager.analyze_conversation("gpt", PREV_PAIR, CURRENT_PAIR)
    except Exception as exc:
        pytest.skip(f"Ollama not available: {exc}")

    assert result["content"]
    print(f"\nLive Ollama response preview: {result['content'][:200]}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_gemma_brownie_conversation_analysis(tmp_path: Path) -> None:
    """Full conversation analysis on a brownie recipe dialog using local Gemma."""
    log_path = tmp_path / "brownie.llm_calls.jsonl"

    with patch(
        "llms.an_llm_manager.load_config",
        return_value=Config(provider="ollama", model_name="gemma4"),
    ):
        analyzer = ConversationAnalyzer(
            llm_type="gpt",
            model_name="gemma4",
            llm_provider="ollama",
            verbose=True,
            llm_calls_log=log_path,
        )

    try:
        results = await analyzer.analyze_conversation_row(
            row_index=0,
            llm_pairs=BROWNIE_CONVERSATION,
        )
    except Exception as exc:
        pytest.skip(f"Ollama not available: {exc}")
    finally:
        analyzer.close()

    assert len(results) == 1
    record = results[0]
    assert record["row_index"] == 0
    assert record["transition_index"] == 1
    assert record["model"] == "gemma4"
    assert record["decision"] in ("allow", "warn", "block")

    log_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 1
    logged = json.loads(log_lines[0])
    assert logged["model"] == "gemma4"
    assert "brownie recipe" in logged["prompt"].lower()
    assert logged["response"]

    print(
        f"\nBrownie conversation analysis:"
        f"\n  decision={record['decision']}"
        f"\n  progressive_risk={record['progressive_risk']}"
        f"\n  interaction_risk={record['interaction_risk']}"
        f"\n  response_preview={logged['response'][:300]}"
    )
