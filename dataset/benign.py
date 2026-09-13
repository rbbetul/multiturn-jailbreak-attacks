"""
Build a manageable benign-conversation subset from the large ChatGPT export.

The full export is one row per message (grouped by ``url``). This script streams
the file, keeps multi-turn English conversations, and writes a smaller CSV that
the TCA loader can read directly (``--format chatgpt``).

Conversations with any single message longer than ``--max-message-chars`` are
skipped so local Ollama runs stay within a reasonable context budget. Only
conversations with at least ``--min-turns`` turns (default 4) are included.

Use ``--format nested`` to write one row per conversation with Red Queen-style
``edited_query`` (for human review or nested TCA loading).

Usage::

    python dataset/benign.py --sample 100 --output dataset/benign_subset100.csv
    python dataset/benign.py --sample 20 --output dataset/benign_subset20.csv
    python dataset/benign.py --preset standard
    python dataset/benign.py --format nested --min-turns 3 --max-turns 15 \\
        --sample 100 --output dataset/benign_subset100_review.csv
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.conversation_dataset_loader import (  # noqa: E402
    DatasetFormat,
    chatgpt_group_to_llm_pairs,
    filter_min_turns,
    load_chatgpt_conversations,
    load_conversations,
)

DEFAULT_SOURCE = Path(__file__).resolve().parent / "chatgpt_results_final_language_filtered.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "benign_subset100.csv"
DEFAULT_NESTED_REVIEW_OUTPUT = (
    Path(__file__).resolve().parent / "benign_subset100_review.csv"
)
DEFAULT_MAX_MESSAGE_CHARS = 2000
CHUNK_SIZE = 100_000


def _serialize_edited_query(pairs: list[tuple[str, str]]) -> str:
    """Match Red Queen nested ``edited_query`` serialization (``repr`` of turn dicts)."""
    turns = [
        {"user": human, "assistant": assistant or None}
        for assistant, human in pairs
    ]
    return repr(turns)


def _format_conversation(pairs: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for index, (assistant, human) in enumerate(pairs, 1):
        parts.append(f"--- Turn {index} ---")
        if human:
            parts.append(f"Human: {human}")
        if assistant:
            parts.append(f"Assistant: {assistant}")
        parts.append("")
    return "\n".join(parts).strip()


def max_message_len(group: pd.DataFrame) -> int:
    """Longest assistant or human text in a conversation."""
    pairs = chatgpt_group_to_llm_pairs(group)
    if not pairs:
        return 0
    lengths = [
        len(str(text or ""))
        for assistant, human in pairs
        for text in (assistant, human)
    ]
    return max(lengths)


def _eligible_group(
    group: pd.DataFrame,
    *,
    min_turns: int,
    max_turns: int | None,
    language: str | None,
    max_message_chars: int | None,
    truncate_over_max_turns: bool = False,
) -> bool:
    if language is not None:
        langs = group["detected_language_final"].dropna().astype(str).str.strip()
        if langs.empty or not (langs == language).all():
            return False
    pairs = chatgpt_group_to_llm_pairs(group)
    if len(pairs) < min_turns:
        return False
    if (
        max_turns is not None
        and not truncate_over_max_turns
        and len(pairs) > max_turns
    ):
        return False
    if max_message_chars is not None and max_message_len(group) > max_message_chars:
        return False
    return True


def collect_eligible_from_frame(
    frame: pd.DataFrame,
    *,
    min_turns: int,
    max_turns: int | None,
    language: str | None,
    max_message_chars: int | None,
    truncate_over_max_turns: bool = False,
) -> list[pd.DataFrame]:
    eligible: list[pd.DataFrame] = []
    for _, group in frame.groupby("url", sort=False):
        if _eligible_group(
            group,
            min_turns=min_turns,
            max_turns=max_turns,
            language=language,
            max_message_chars=max_message_chars,
            truncate_over_max_turns=truncate_over_max_turns,
        ):
            eligible.append(group)
    return eligible


def stream_conversations(
    source: Path,
    *,
    min_turns: int = 4,
    max_turns: int | None = None,
    language: str | None = "English",
    max_message_chars: int | None = DEFAULT_MAX_MESSAGE_CHARS,
    max_scan: int | None = None,
    truncate_over_max_turns: bool = False,
) -> list[pd.DataFrame]:
    """Load message rows and return eligible conversation groups."""
    if not source.is_file():
        raise FileNotFoundError(f"Source not found: {source}")

    length_note = (
        f", max_message_chars={max_message_chars}"
        if max_message_chars is not None
        else ""
    )
    turns_note = (
        f", max_turns={max_turns}"
        if max_turns is not None
        else ""
    )

    if max_scan is not None:
        frame = pd.read_csv(source, nrows=max_scan)
        eligible = collect_eligible_from_frame(
            frame,
            min_turns=min_turns,
            max_turns=max_turns,
            language=language,
            max_message_chars=max_message_chars,
            truncate_over_max_turns=truncate_over_max_turns,
        )
        print(
            f"Scanned {len(frame):,} message rows; found {len(eligible):,} eligible conversations "
            f"(min_turns={min_turns}, language={language!r}{turns_note}{length_note})."
        )
        return eligible

    collected: list[pd.DataFrame] = []
    scanned = 0
    for chunk in pd.read_csv(source, chunksize=CHUNK_SIZE):
        collected.extend(
            collect_eligible_from_frame(
                chunk,
                min_turns=min_turns,
                max_turns=max_turns,
                language=language,
                max_message_chars=max_message_chars,
                truncate_over_max_turns=truncate_over_max_turns,
            )
        )
        scanned += len(chunk)
        print(f"  ... scanned {scanned:,} rows, {len(collected):,} eligible so far")

    print(
        f"Scanned {scanned:,} message rows; found {len(collected):,} eligible conversations "
        f"(min_turns={min_turns}, language={language!r}{turns_note}{length_note})."
    )
    return collected


def chatgpt_group_to_nested_row(
    group: pd.DataFrame,
    *,
    max_turns: int | None = None,
) -> dict[str, object]:
    """One nested-format row for human review (Red Queen ``edited_query`` style)."""
    pairs = chatgpt_group_to_llm_pairs(group)
    truncated = False
    if max_turns is not None and len(pairs) > max_turns:
        pairs = pairs[:max_turns]
        truncated = True
    first = group.iloc[0]
    source_row_id = first.get("Unnamed: 0")
    if pd.isna(source_row_id):
        source_row_id = ""
    return {
        "source_row_id": source_row_id,
        "url": first["url"],
        "turn": len(pairs),
        "edited_query": _serialize_edited_query(pairs),
        "conversation": _format_conversation(pairs),
        "benign": "",
        "notes": "",
        "truncated": "yes" if truncated else "no",
    }


def write_nested_review_from_eligible(
    eligible: list[pd.DataFrame],
    output: Path,
    *,
    sample: int,
    seed: int,
    min_turns: int,
    max_turns: int | None,
    max_message_chars: int | None,
) -> Path:
    if not eligible:
        raise SystemExit(
            "No eligible conversations found. Relax --language, --min-turns, "
            "--max-turns, or --max-message-chars."
        )
    if sample > len(eligible):
        raise SystemExit(
            f"--sample {sample} exceeds eligible pool ({len(eligible)}). "
            "Relax filters or scan more source rows."
        )

    rng = random.Random(seed)
    chosen = rng.sample(eligible, sample)
    rows = [
        chatgpt_group_to_nested_row(group, max_turns=max_turns)
        for group in chosen
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)

    conversations, fmt, _ = load_conversations(output, DatasetFormat.NESTED)
    verified = filter_min_turns(conversations, min_turns=min_turns)
    max_lens = [
        max(len(str(a or "")), len(str(h or "")))
        for item in conversations
        for a, h in item["llm_pairs"]
    ]
    turn_counts = [len(item["llm_pairs"]) for item in conversations]
    longest = max(max_lens) if max_lens else 0
    print(f"Wrote {len(rows)} nested review rows to {output}")
    print(f"Verified format={fmt.value}, {len(verified)} conversations with >= {min_turns} turns.")
    if turn_counts:
        print(
            f"Turns per conversation: min {min(turn_counts)}, "
            f"max {max(turn_counts)}, avg {sum(turn_counts) / len(turn_counts):.1f}."
        )
    if max_message_chars is not None:
        print(
            f"Longest single message in subset: {longest} chars "
            f"(limit {max_message_chars})."
        )
    if max_turns is not None:
        over_max = sum(1 for count in turn_counts if count > max_turns)
        if over_max:
            raise SystemExit(
                f"Verification failed: {over_max} conversation(s) exceed max_turns={max_turns}."
            )
        truncated_count = sum(1 for row in rows if row.get("truncated") == "yes")
        if truncated_count:
            print(
                f"Truncated {truncated_count} conversation(s) to the first {max_turns} turns."
            )
    return output


def write_subset_from_eligible(
    eligible: list[pd.DataFrame],
    output: Path,
    *,
    sample: int,
    seed: int,
    min_turns: int,
    max_message_chars: int | None,
) -> Path:
    if not eligible:
        raise SystemExit(
            "No eligible conversations found. Relax --language, --min-turns, "
            "or --max-message-chars."
        )
    if sample > len(eligible):
        raise SystemExit(
            f"--sample {sample} exceeds eligible pool ({len(eligible)}). "
            "Increase --max-message-chars or scan more source rows."
        )

    rng = random.Random(seed)
    chosen = rng.sample(eligible, sample)
    subset = pd.concat(chosen, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output, index=False)

    conversations = load_chatgpt_conversations(subset)
    conversations = filter_min_turns(conversations, min_turns=min_turns)
    max_lens = [
        max(len(str(a or "")), len(str(h or "")))
        for item in conversations
        for a, h in item["llm_pairs"]
    ]
    longest = max(max_lens) if max_lens else 0
    print(f"Wrote {len(chosen)} conversations ({len(subset)} message rows) to {output}")
    print(f"Verified {len(conversations)} conversations with >= {min_turns} turns.")
    if max_message_chars is not None:
        print(
            f"Longest single message in subset: {longest} chars "
            f"(limit {max_message_chars})."
        )
    return output


def build_subset(
    source: Path,
    output: Path,
    *,
    sample: int,
    seed: int,
    min_turns: int,
    max_turns: int | None,
    language: str | None,
    max_message_chars: int | None,
    max_scan: int | None,
    output_format: str = "chatgpt",
    truncate_over_max_turns: bool = False,
    eligible: list[pd.DataFrame] | None = None,
) -> Path:
    if eligible is None:
        eligible = stream_conversations(
            source,
            min_turns=min_turns,
            max_turns=max_turns,
            language=language,
            max_message_chars=max_message_chars,
            max_scan=max_scan,
            truncate_over_max_turns=truncate_over_max_turns,
        )
    if output_format == "nested":
        return write_nested_review_from_eligible(
            eligible,
            output,
            sample=sample,
            seed=seed,
            min_turns=min_turns,
            max_turns=max_turns,
            max_message_chars=max_message_chars,
        )
    return write_subset_from_eligible(
        eligible,
        output,
        sample=sample,
        seed=seed,
        min_turns=min_turns,
        max_message_chars=max_message_chars,
    )


STANDARD_PRESETS: dict[str, tuple[int, Path]] = {
    "20": (20, Path(__file__).resolve().parent / "benign_subset20.csv"),
    "100": (100, Path(__file__).resolve().parent / "benign_subset100.csv"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a benign ChatGPT subset CSV.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Max turns per conversation (exclude when sampling, or cap when truncating)",
    )
    parser.add_argument(
        "--truncate-over-max-turns",
        action="store_true",
        help=(
            "With --format nested: keep overlong conversations but cap edited_query "
            "to --max-turns (sets truncated=yes)"
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["chatgpt", "nested"],
        default="chatgpt",
        help=(
            "chatgpt = one row per message (TCA default); "
            "nested = one row per conversation with edited_query"
        ),
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Keep conversations where every message has this language (default: English; use '' for any)",
    )
    parser.add_argument(
        "--max-message-chars",
        type=int,
        default=DEFAULT_MAX_MESSAGE_CHARS,
        help=(
            "Skip conversations where any assistant or human message exceeds this length "
            f"(default: {DEFAULT_MAX_MESSAGE_CHARS}; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=None,
        help="Stop after scanning this many message rows (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--preset",
        choices=["standard"],
        default=None,
        help="Write standard project subsets (benign_subset20.csv and benign_subset100.csv) in one scan",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    language = args.language.strip() or None
    max_message_chars = args.max_message_chars or None

    if args.preset == "standard":
        if args.output_format != "chatgpt":
            raise SystemExit("--preset standard only supports --format chatgpt.")
        eligible = stream_conversations(
            args.source,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            language=language,
            max_message_chars=max_message_chars,
            max_scan=args.max_scan,
        )
        for sample, output in STANDARD_PRESETS.values():
            write_subset_from_eligible(
                eligible,
                output,
                sample=sample,
                seed=args.seed,
                min_turns=args.min_turns,
                max_message_chars=max_message_chars,
            )
        return

    build_subset(
        args.source,
        args.output,
        sample=args.sample,
        seed=args.seed,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        language=language,
        max_message_chars=max_message_chars,
        max_scan=args.max_scan,
        output_format=args.output_format,
        truncate_over_max_turns=args.truncate_over_max_turns,
    )


if __name__ == "__main__":
    main()
