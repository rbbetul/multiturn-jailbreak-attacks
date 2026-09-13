"""Structured JSONL logging for individual LLM API calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, TextIO


def llm_calls_log_path_for_output(output_csv: Path) -> Path:
    """Pair a results CSV with an LLM call log that shares the same stem."""
    return Path(output_csv).with_suffix(".llm_calls.jsonl")


class LLMCallLogger:
    """Append one JSON object per LLM call to a JSONL file."""

    def __init__(self, log_path: Path, *, verbose: bool = True):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._call_count = 0
        self._file: TextIO = self.log_path.open("w", encoding="utf-8")

    def log_call(
        self,
        *,
        model: str,
        prompt: str,
        response: str,
        duration_ms: float,
        usage: Dict[str, Any] | None = None,
        context: Dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self._call_count += 1
        record: Dict[str, Any] = {
            "call_index": self._call_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "duration_ms": round(duration_ms, 1),
            "context": context or {},
            "prompt": prompt,
            "response": response,
        }
        if usage:
            record["usage"] = usage
        if error:
            record["error"] = error

        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

        if self.verbose:
            ctx = context or {}
            row = ctx.get("row_index", "?")
            transition = ctx.get("transition_index", "?")
            status = f"error: {error}" if error else f"{duration_ms:.0f}ms"
            print(
                f"LLM call #{self._call_count}: "
                f"row={row} transition={transition} model={model} ({status})"
            )

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> LLMCallLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
