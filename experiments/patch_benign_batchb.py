"""
Patch batchB benign CSVs: drop removed rows and append the replacement conversation.

Usage::

    python experiments/patch_benign_batchb.py
    python experiments/patch_benign_batchb.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.conversation_dataset_loader import DatasetFormat  # noqa: E402
from experiments.benign_batches import (  # noqa: E402
    BENIGN_BATCH_REPLACEMENT_ROW,
    BENIGN_BATCHB_REMOVED_ROWS,
    BENIGN_EXCLUDED_ROWS,
)
from intent.cli import OUTPUT_DIR, resolve_data_path, run_analysis  # noqa: E402

DATASET = "benign_subset100.csv"
BATCH_INDEX = 1

PATCH_TARGETS = [
    ("openai/gpt-4o", "gpt/gpt-4o/batchB/benign_full/results.csv", False),
    ("openai/gpt-4o", "gpt/gpt-4o/batchB/benign_human_only/results.csv", True),
    ("openai/gpt-5.5", "gpt/gpt-5.5/batchB/benign_full/results.csv", False),
    ("openai/gpt-5.5", "gpt/gpt-5.5/batchB/benign_human_only/results.csv", True),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch batchB benign CSVs with the current replacement row."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Drop removed rows only; do not run LLM analysis for the replacement row.",
    )
    parser.add_argument("--config", default="config/config.yaml")
    return parser


async def analyze_replacement_row(
    *,
    config_path: str,
    model_name: str,
    human_only: bool,
    temp_csv: Path,
) -> None:
    await run_analysis(
        config_path=config_path,
        data_path=resolve_data_path(DATASET),
        dataset_format=DatasetFormat.CHATGPT,
        llm_type="gpt",
        model_name=model_name,
        llm_provider="openrouter",
        max_rows=None,
        random_sample=1,
        batch_index=BATCH_INDEX,
        random_seed=None,
        min_turns=2,
        output_csv=temp_csv,
        llm_calls_log=None,
        interactive=False,
        human_only=human_only,
        batch_row_indices=[BENIGN_BATCH_REPLACEMENT_ROW],
        exclude_row_indices=None,
        random_sampling=False,
    )


def merge_results(results_csv: Path, temp_csv: Path | None) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    drop_rows = set(BENIGN_BATCHB_REMOVED_ROWS) | {BENIGN_BATCH_REPLACEMENT_ROW}
    removed_counts = {
        row_id: int((df["row_index"] == row_id).sum()) for row_id in sorted(drop_rows)
    }
    df = df[~df["row_index"].isin(drop_rows)].copy()

    added = 0
    if temp_csv is not None:
        replacement = pd.read_csv(temp_csv)
        replacement["batch_index"] = BATCH_INDEX
        df = pd.concat([df, replacement], ignore_index=True)
        added = len(replacement)

    merged = df.sort_values(["row_index", "transition_index"]).reset_index(drop=True)
    removed_summary = ", ".join(f"row {k}: {v}" for k, v in removed_counts.items() if v)
    print(
        f"  {results_csv.name}: removed [{removed_summary}], "
        f"added {added} row-{BENIGN_BATCH_REPLACEMENT_ROW} transitions"
    )
    return merged


async def main_async(args: argparse.Namespace) -> None:
    print(f"Replacement row: {BENIGN_BATCH_REPLACEMENT_ROW}")
    print(f"Excluded row:    {sorted(BENIGN_EXCLUDED_ROWS)}")
    print(f"Drop from CSVs:  {sorted(BENIGN_BATCHB_REMOVED_ROWS)}")

    for model_name, rel_path, human_only in PATCH_TARGETS:
        results_csv = OUTPUT_DIR / rel_path
        if not results_csv.is_file():
            print(f"SKIP missing: {results_csv}")
            continue

        mode = "human_only" if human_only else "full"
        print(f"\nPatching {model_name} ({mode}) -> {results_csv}")

        if args.dry_run:
            merged = merge_results(results_csv, temp_csv=None)
            merged.to_csv(results_csv, index=False)
            print(f"  dry-run: wrote {len(merged)} rows")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_csv = Path(tmpdir) / "replacement.csv"
            await analyze_replacement_row(
                config_path=args.config,
                model_name=model_name,
                human_only=human_only,
                temp_csv=temp_csv,
            )
            merged = merge_results(results_csv, temp_csv)
            merged.to_csv(results_csv, index=False)
            print(f"  wrote {len(merged)} rows -> {results_csv}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
