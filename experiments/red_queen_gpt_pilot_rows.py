"""
Rerun the June GPT-5.5 N=100 pilot rows with another model (paired comparison).

The pilot (red_queen_outputs/preliminary/2026-06_pilot_n100/gpt-5.5/n100_seed0/rq_full)
used the old shuffled batching on ``red_queen_subset500.csv`` (CLI --seed 1), which
``red_queen_batch.py`` can no longer reproduce. This script pins the exact 100
``row_index`` values from that run, so the same conversations can be analyzed by
Gemma (full-conversation mode, i.e. the GPT counterpart of cell A).

Usage::

    python experiments/red_queen_gpt_pilot_rows.py --provider ollama --model gemma4:latest
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.conversation_dataset_loader import DatasetFormat  # noqa: E402
from experiments.run_logging import log_path_for_output, tee_terminal_output  # noqa: E402
from llms.llm_call_logger import llm_calls_log_path_for_output  # noqa: E402
from llms.an_llm_manager import load_config  # noqa: E402
from intent.cli import OUTPUT_DIR, resolve_data_path, run_analysis  # noqa: E402

DATASET = "red_queen_subset500.csv"
DEFAULT_MIN_TURNS = 2
PILOT_BATCH_INDEX = 1  # value recorded in the GPT pilot's batch_index column

# row_index values of the 100 conversations in the GPT-5.5 pilot results.csv
PILOT_ROW_INDICES = [
    3, 11, 13, 18, 21, 29, 39, 47, 48, 55, 56, 60, 72, 77, 82, 86, 99, 103, 105,
    107, 112, 114, 115, 118, 121, 122, 134, 135, 146, 150, 156, 160, 161, 162,
    164, 178, 182, 184, 187, 189, 195, 199, 206, 207, 208, 212, 222, 229, 231,
    232, 236, 248, 249, 252, 260, 262, 264, 267, 268, 270, 274, 278, 282, 291,
    297, 301, 303, 309, 315, 317, 322, 323, 326, 328, 337, 360, 367, 370, 374,
    378, 394, 399, 408, 412, 418, 420, 422, 433, 446, 457, 458, 467, 475, 480,
    481, 484, 485, 489, 495, 496,
]


def experiment_output_path(model_name: str) -> Path:
    slug = model_name.replace("/", "_").replace(":", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"gptpilot_rows_n{len(PILOT_ROW_INDICES)}_{slug}_{stamp}.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerun the GPT-5.5 N=100 pilot rows with another model."
    )
    parser.add_argument("--data", default=DATASET)
    parser.add_argument("--min-turns", type=int, default=DEFAULT_MIN_TURNS)
    parser.add_argument("--model", default=None, help="Model id (default: from config.yaml)")
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=None)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    args = build_parser().parse_args()
    data_path = resolve_data_path(args.data)
    llm_cfg = load_config(args.config, provider=args.provider)
    model_name = args.model or llm_cfg.model_name
    llm_provider = args.provider or llm_cfg.provider
    output_csv = args.output or experiment_output_path(model_name)
    log_path = log_path_for_output(output_csv)
    llm_log_path = llm_calls_log_path_for_output(output_csv)

    with tee_terminal_output(log_path):
        print("Red Queen GPT-pilot-rows rerun")
        print(f"  Dataset:     {data_path}")
        print(f"  Rows:        {len(PILOT_ROW_INDICES)} fixed row indices")
        print(f"  Provider:    {llm_provider}")
        print(f"  Model:       {model_name}")
        print(f"  Output CSV:  {output_csv}")

        asyncio.run(
            run_analysis(
                config_path=args.config,
                data_path=data_path,
                dataset_format=DatasetFormat.NESTED,
                llm_type="gpt",
                model_name=model_name,
                llm_provider=llm_provider,
                max_rows=None,
                random_sample=len(PILOT_ROW_INDICES),
                batch_index=PILOT_BATCH_INDEX,
                random_seed=None,
                min_turns=args.min_turns,
                output_csv=output_csv,
                llm_calls_log=llm_log_path,
                interactive=False,
                batch_row_indices=PILOT_ROW_INDICES,
            )
        )


if __name__ == "__main__":
    main()
