"""
Load multi-turn conversation datasets for TCA batch analysis.

Supported formats
-----------------

1. **nested** — one CSV row per conversation

   Required column:
   - ``edited_query``: Python-literal list of turn dicts with ``user`` and ``assistant``.

2. **pairs** — one CSV row per turn, grouped by ``row_index``

   Required: ``row_index``, ``pair_index``, ``human_message``, ``assistant_message``.

3. **harmbench** — HarmBench-style CSV with ``message_0``, ``message_1``, ...

   Pairs are built as consecutive message columns: (assistant, human) per pair.
   Typical metadata: ``tactic``, ``Source``, ``temperature``.

4. **chatgpt** — one CSV row per message, grouped by ``url``

   Required: ``url``, ``message_index``, ``role``, ``plain_text``.
   Roles ``user`` / ``llm`` are paired into turns: (assistant, human).

Turn order is preserved. The analyzer scores transitions between consecutive
turn-pairs, so each conversation needs at least 2 turns.
"""

from __future__ import annotations

import ast
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ConversationPair = Tuple[str, str]  # (assistant, human) — TCA / prompt format


class DatasetFormat(str, Enum):
    NESTED = "nested"
    PAIRS = "pairs"
    HARMBENCH = "harmbench"
    CHATGPT = "chatgpt"
    AUTO = "auto"


RESERVED_PAIR_COLUMNS = {
    "row_index",
    "pair_index",
    "human_message",
    "assistant_message",
}

CHATGPT_REQUIRED_COLUMNS = {"url", "message_index", "role", "plain_text"}
CHATGPT_RESERVED_COLUMNS = CHATGPT_REQUIRED_COLUMNS | {"turns_count"}


def _message_columns(columns) -> List[str]:
    return sorted(
        [c for c in columns if str(c).startswith("message_")],
        key=lambda x: int(str(x).split("_")[1]),
    )


def detect_format(df: pd.DataFrame) -> DatasetFormat:
    cols = set(df.columns)
    if {"row_index", "pair_index", "human_message", "assistant_message"}.issubset(cols):
        return DatasetFormat.PAIRS
    if CHATGPT_REQUIRED_COLUMNS.issubset(cols):
        return DatasetFormat.CHATGPT
    if "edited_query" in cols:
        return DatasetFormat.NESTED
    if _message_columns(df.columns):
        return DatasetFormat.HARMBENCH
    raise ValueError(
        "Could not detect dataset format. Use --format nested, pairs, chatgpt, or harmbench.\n"
        + DATA_FORMAT_HELP
    )


def _metadata_from_series(row: pd.Series, exclude: set[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for col, value in row.items():
        if col in exclude:
            continue
        if pd.isna(value):
            continue
        meta[col] = value
    return meta


def _parse_edited_query(value: Any) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return ast.literal_eval(value)
    return list(value)


def nested_row_to_llm_pairs(row: pd.Series) -> List[ConversationPair]:
    pairs: List[ConversationPair] = []
    for turn in _parse_edited_query(row.get("edited_query")):
        if not isinstance(turn, dict):
            continue
        human = "" if turn.get("user") is None else str(turn["user"])
        assistant = "" if turn.get("assistant") is None else str(turn["assistant"])
        pairs.append((assistant, human))
    return pairs


def harmbench_row_to_llm_pairs(row: pd.Series) -> List[ConversationPair]:
    messages: List[str] = []
    for col in _message_columns(row.index):
        raw = row[col]
        if pd.notna(raw) and str(raw).strip():
            try:
                messages.append(json.loads(raw).get("body", str(raw)))
            except json.JSONDecodeError:
                messages.append(str(raw))

    pairs: List[ConversationPair] = []
    for i in range(0, len(messages), 2):
        assistant = messages[i] if i < len(messages) else ""
        human = messages[i + 1] if i + 1 < len(messages) else ""
        pairs.append((str(assistant), str(human)))
    return pairs


def load_nested_conversations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        llm_pairs = nested_row_to_llm_pairs(row)
        metadata = _metadata_from_series(row, exclude={"edited_query"})
        metadata["row_index"] = row_index
        rows.append({"row_index": row_index, "llm_pairs": llm_pairs, "metadata": metadata})
    return rows


def load_pairs_conversations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    required = {"row_index", "pair_index", "human_message", "assistant_message"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pairs format missing columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    for row_index, group in df.groupby("row_index", sort=True):
        group = group.sort_values("pair_index")
        first = group.iloc[0]
        llm_pairs = [
            (str(r["assistant_message"]), str(r["human_message"]))
            for _, r in group.iterrows()
        ]
        metadata = _metadata_from_series(first, exclude=RESERVED_PAIR_COLUMNS)
        metadata["row_index"] = int(row_index)
        rows.append(
            {"row_index": int(row_index), "llm_pairs": llm_pairs, "metadata": metadata}
        )
    return rows


def _normalize_chatgpt_role(role: Any) -> str:
    value = "" if role is None or pd.isna(role) else str(role).strip().lower()
    if value in {"llm", "assistant", "model"}:
        return "assistant"
    if value in {"user", "human"}:
        return "user"
    return value


def messages_to_llm_pairs(messages: List[Dict[str, Any]]) -> List[ConversationPair]:
    """Build (assistant, human) turns from ordered ChatGPT-style message rows."""
    pairs: List[ConversationPair] = []
    i = 0
    while i < len(messages):
        role = _normalize_chatgpt_role(messages[i].get("role"))
        text = "" if messages[i].get("plain_text") is None else str(messages[i]["plain_text"])
        if role == "user":
            assistant = ""
            if (
                i + 1 < len(messages)
                and _normalize_chatgpt_role(messages[i + 1].get("role")) == "assistant"
            ):
                assistant = str(messages[i + 1].get("plain_text") or "")
                i += 2
            else:
                i += 1
            pairs.append((assistant, text))
        elif role == "assistant":
            pairs.append((text, ""))
            i += 1
        else:
            i += 1
    return pairs


def chatgpt_group_to_llm_pairs(group: pd.DataFrame) -> List[ConversationPair]:
    ordered = group.sort_values("message_index")
    messages = [
        {"role": row["role"], "plain_text": row["plain_text"]}
        for _, row in ordered.iterrows()
    ]
    return messages_to_llm_pairs(messages)


def load_chatgpt_conversations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    missing = CHATGPT_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"chatgpt format missing columns: {sorted(missing)}")

    rows: List[Dict[str, Any]] = []
    for row_index, (url, group) in enumerate(df.groupby("url", sort=False)):
        llm_pairs = chatgpt_group_to_llm_pairs(group)
        first = group.iloc[0]
        metadata = _metadata_from_series(first, exclude=CHATGPT_RESERVED_COLUMNS)
        metadata["row_index"] = row_index
        metadata["url"] = url
        metadata["num_messages"] = len(group)
        rows.append({"row_index": row_index, "llm_pairs": llm_pairs, "metadata": metadata})
    return rows


def load_harmbench_conversations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    message_cols = set(_message_columns(df.columns))
    rows: List[Dict[str, Any]] = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        llm_pairs = harmbench_row_to_llm_pairs(row)
        metadata = _metadata_from_series(row, exclude=message_cols)
        metadata["row_index"] = row_index
        rows.append({"row_index": row_index, "llm_pairs": llm_pairs, "metadata": metadata})
    return rows


def load_conversations(
    path: str | Path,
    fmt: DatasetFormat | str = DatasetFormat.AUTO,
) -> tuple[List[Dict[str, Any]], DatasetFormat, Path]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path).dropna(axis=1, how="all")
    print(f"Loaded {len(df)} CSV rows from {path}")

    if isinstance(fmt, str):
        fmt = DatasetFormat(fmt)

    resolved = detect_format(df) if fmt == DatasetFormat.AUTO else DatasetFormat(fmt)
    print(f"Using dataset format: {resolved.value}")

    loaders = {
        DatasetFormat.NESTED: load_nested_conversations,
        DatasetFormat.PAIRS: load_pairs_conversations,
        DatasetFormat.HARMBENCH: load_harmbench_conversations,
        DatasetFormat.CHATGPT: load_chatgpt_conversations,
    }
    conversations = loaders[resolved](df)

    print(f"Parsed {len(conversations)} conversations")
    return conversations, resolved, path


def filter_by_metadata(
    conversations: List[Dict[str, Any]],
    key: str,
    value: Any,
) -> List[Dict[str, Any]]:
    return [c for c in conversations if c.get("metadata", {}).get(key) == value]


def to_human_only_pairs(
    llm_pairs: List[ConversationPair],
) -> List[ConversationPair]:
    """Drop assistant text; keep (assistant, human) shape with empty assistant slots."""
    return [("", human) for _, human in llm_pairs]


def apply_human_only_mode(
    conversations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return conversations whose llm_pairs contain human messages only."""
    updated: List[Dict[str, Any]] = []
    for item in conversations:
        copy = dict(item)
        copy["llm_pairs"] = to_human_only_pairs(item["llm_pairs"])
        updated.append(copy)
    return updated


def filter_min_turns(
    conversations: List[Dict[str, Any]],
    min_turns: int = 2,
) -> List[Dict[str, Any]]:
    """Keep only conversations with at least ``min_turns`` (assistant, human) pairs."""
    eligible = [c for c in conversations if len(c.get("llm_pairs") or []) >= min_turns]
    dropped = len(conversations) - len(eligible)
    if dropped:
        print(
            f"Filtered out {dropped} conversation(s) with fewer than {min_turns} turns "
            f"({len(eligible)} eligible)."
        )
    return eligible


DATA_FORMAT_HELP = __doc__ or ""
