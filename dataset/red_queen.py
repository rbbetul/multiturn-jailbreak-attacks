"""
Build Red Queen nested-format CSVs from raw model JSON attack data.

Each JSON file has 56,000 conversations indexed 0..55999. The list index is stored
as ``source_row_id`` so results can be traced back to the full corpus.

Usage::

    python dataset/red_queen.py --model gpt4o_mini
    python dataset/red_queen.py --model llama3_70b
    python dataset/red_queen.py --model llama3_70b --sample 500 --seed 42 \\
        --output dataset/red_queen_subset500_llama.csv
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.conversation_dataset_loader import (  # noqa: E402
    DatasetFormat,
    filter_min_turns,
    load_conversations,
)

RAW_DIR = Path(__file__).resolve().parent / "raw_data" / "Red_Queen_Attack"

MODEL_SOURCES: dict[str, tuple[Path, str]] = {
    "gpt4o_mini": (RAW_DIR / "gpt_4o_mini_attack_data.json", "gpt-4o-mini"),
    "llama3_70b": (RAW_DIR / "llama3_70b_attack_data.json", "llama3-70b"),
}

DEFAULT_OUTPUTS: dict[str, Path] = {
    "gpt4o_mini": Path(__file__).resolve().parent / "red_queen_full_gpt4o_mini.csv",
    "llama3_70b": Path(__file__).resolve().parent / "red_queen_full_llama3_70b.csv",
}


def gpt_messages_to_edited_query(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style message list to nested ``edited_query`` turns."""
    pairs: list[dict[str, Any]] = []
    index = 1  # skip system prompt
    while index < len(messages):
        if messages[index]["role"] == "user":
            user = messages[index]["content"]
            assistant = None
            if (
                index + 1 < len(messages)
                and messages[index + 1]["role"] == "assistant"
            ):
                assistant = messages[index + 1]["content"]
                index += 2
            else:
                index += 1
            pairs.append({"user": user, "assistant": assistant})
        else:
            index += 1
    return pairs


def llama_query_to_edited_query(query: str) -> list[dict[str, Any]]:
    """Parse Llama3 chat-template string into nested ``edited_query`` turns."""
    pairs: list[dict[str, Any]] = []
    for part in query.split("<|eot_id|>"):
        match = re.search(
            r"header_id\|>(user|assistant)<\|[^>]*\|>\s*\n\n(.*)",
            part,
            re.DOTALL,
        )
        if not match:
            continue
        role, content = match.group(1), match.group(2).strip()
        if role == "user":
            pairs.append({"user": content, "assistant": None})
        elif role == "assistant" and pairs:
            pairs[-1]["assistant"] = content
    return pairs


def _serialize_edited_query(pairs: list[dict[str, Any]]) -> str:
    return repr(pairs)


def record_to_row(model: str, source_row_id: int, record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_row_id": source_row_id,
        "action": record["action"],
        "turn": record["turn"],
        "type": record["type"],
        "category": record["category"],
        "response_model": MODEL_SOURCES[model][1],
    }
    if model == "gpt4o_mini":
        messages = record["query"]
        if not isinstance(messages, list):
            raise ValueError(
                f"Expected message list for gpt4o_mini at index {source_row_id}"
            )
        row["edited_query"] = _serialize_edited_query(
            gpt_messages_to_edited_query(messages)
        )
    elif model == "llama3_70b":
        query = record["query"]
        if not isinstance(query, str):
            raise ValueError(
                f"Expected query string for llama3_70b at index {source_row_id}"
            )
        row["query"] = query
        row["edited_query"] = _serialize_edited_query(llama_query_to_edited_query(query))
    else:
        raise ValueError(f"Unknown model: {model}")
    return row


def load_attack_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Attack data not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def build_rows(model: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record_to_row(model, index, record) for index, record in enumerate(records)]


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    sample: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample > len(rows):
        raise SystemExit(
            f"--sample {sample} exceeds corpus size ({len(rows)})."
        )
    chosen = random.Random(seed).sample(rows, sample)
    return chosen


def verify_nested_csv(path: Path, *, min_turns: int) -> None:
    conversations, fmt, _ = load_conversations(path, DatasetFormat.NESTED)
    eligible = filter_min_turns(conversations, min_turns=min_turns)
    print(
        f"Verified {path.name}: format={fmt.value}, "
        f"{len(conversations)} conversations, "
        f"{len(eligible)} with >= {min_turns} turns."
    )


def write_csv(rows: list[dict[str, Any]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} conversations to {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Red Queen nested CSV from raw attack JSON."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SOURCES),
        required=True,
        help="Which model attack JSON to convert",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Override path to attack JSON (default: raw_data/Red_Queen_Attack/...)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: dataset/red_queen_full_<model>.csv)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N conversations after loading the full JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --sample (default: 42)",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Turns required when verifying output (default: 2)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip nested-format verification after writing",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    default_source, _ = MODEL_SOURCES[args.model]
    source = args.source or default_source
    output = args.output or DEFAULT_OUTPUTS[args.model]

    print(f"Loading {source} ...")
    records = load_attack_json(source)
    print(f"Loaded {len(records):,} attack records.")

    rows = build_rows(args.model, records)
    if args.sample is not None:
        rows = sample_rows(rows, sample=args.sample, seed=args.seed)
        print(f"Sampled {len(rows)} conversations (seed={args.seed}).")

    write_csv(rows, output)
    if not args.no_verify:
        verify_nested_csv(output, min_turns=args.min_turns)


if __name__ == "__main__":
    main()
