"""
Red Queen batch experiment.

--batch selects a disjoint slice of the frozen pool in CSV order (no re-shuffle).

Usage::

    python experiments/red_queen_batch.py
    python experiments/red_queen_batch.py --batch 1 --sample 5
    python experiments/red_queen_batch.py --random --random-seed 42
    python experiments/red_queen_batch.py --provider ollama --model gemma4:latest
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
from intent.cli import (  # noqa: E402
    OUTPUT_DIR,
    resolve_data_path,
    resolve_exclude_row_indices,
    run_analysis,
)

DATASET = "red_queen_full_llama3_70b.csv"
DEFAULT_SAMPLE = 5
DEFAULT_MIN_TURNS = 2
DEFAULT_BATCH = 0
DEFAULT_RANDOM_SEED = 42


def experiment_output_path(model_name: str, sample: int, batch: int) -> Path:
    slug = model_name.replace("/", "_").replace(":", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"batch_{batch:04d}_n{sample}_{slug}_{stamp}.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Red Queen batch + TCA analysis.")
    parser.add_argument("--data", default=DATASET, help="Red Queen nested CSV")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--min-turns", type=int, default=DEFAULT_MIN_TURNS)
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Disjoint batch index in CSV pool order (default: 0)",
    )
    parser.add_argument("--model", default=None, help="Model id (default: from config.yaml)")
    parser.add_argument(
        "--provider",
        choices=["openrouter", "ollama"],
        default=None,
        help="LLM backend (default: config.yaml llm.provider)",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--random",
        action="store_true",
        help="RNG sampling instead of fixed --batch slices",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="RNG seed when using --random (default: 42)",
    )
    parser.add_argument(
        "--allow-repeat-rows",
        action="store_true",
        help="With --random: allow rows already in red_queen_outputs/*.csv",
    )
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
    output_csv = args.output or experiment_output_path(model_name, args.sample, args.batch)
    log_path = log_path_for_output(output_csv)
    llm_log_path = llm_calls_log_path_for_output(output_csv)

    exclude_rows: set[int] = set()
    if args.random and not args.allow_repeat_rows:
        exclude_rows = resolve_exclude_row_indices(
            exclude_from_outputs=True,
            skip_output=output_csv,
        )

    with tee_terminal_output(log_path):
        print("Red Queen batch experiment")
        print(f"  Dataset:     {data_path}")
        print(f"  Batch size:  {args.sample}")
        print(f"  Batch index: {args.batch}")
        print(f"  Mode:        {'random' if args.random else 'CSV-order batch'}")
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
                random_sample=args.sample,
                batch_index=args.batch,
                random_seed=args.random_seed,
                min_turns=args.min_turns,
                output_csv=output_csv,
                llm_calls_log=llm_log_path,
                interactive=False,
                exclude_row_indices=exclude_rows or None,
                random_sampling=args.random,
            )
        )


if __name__ == "__main__":
    main()
