"""CLI for batch and interactive conversation risk analysis."""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.conversation_dataset_loader import (  # noqa: E402
    DATA_FORMAT_HELP,
    DatasetFormat,
    apply_human_only_mode,
    filter_by_metadata,
    filter_min_turns,
    load_conversations,
)
from experiments.batching import select_batch_conversations  # noqa: E402
from experiments.run_logging import write_run_manifest  # noqa: E402
from intent.conversation_analyzer import ConversationAnalyzer  # noqa: E402
from llms.llm_call_logger import llm_calls_log_path_for_output  # noqa: E402
from llms.an_llm_manager import load_config  # noqa: E402

OUTPUT_DIR = ROOT / "red_queen_outputs"
DATASET_DIR = ROOT / "dataset"
ARCHIVE_DIR = ROOT / "archive"


def _output_name_stem(data_path: Path) -> str:
    stem = data_path.stem
    if stem.startswith("red_queen_"):
        return stem[len("red_queen_") :]
    return stem


def resolve_data_path(data: str) -> Path:
    candidate = Path(data)
    if candidate.is_file():
        return candidate.resolve()
    for base in (DATASET_DIR, ARCHIVE_DIR, ROOT):
        path = base / data
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Dataset not found: {data}\n"
        f"Searched: {candidate}, {DATASET_DIR / data}, {ARCHIVE_DIR / data}, {ROOT / data}"
    )


def default_output_path(data_path: Path, model_name: str) -> Path:
    slug = model_name.replace("/", "_").replace(":", "_")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{_output_name_stem(data_path)}_{slug}_risk_results.csv"


def _prompt_tactic(conversations: list) -> list:
    tactics = sorted(
        c["metadata"]["tactic"]
        for c in conversations
        if c.get("metadata", {}).get("tactic") is not None
    )
    if not tactics:
        raise SystemExit("No tactics found in dataset metadata.")

    choices = {chr(97 + i): t for i, t in enumerate(tactics)}
    print("\nAvailable tactics:")
    for key, tactic in choices.items():
        print(f"  {key}. {tactic}")

    while True:
        choice = input("\nSelect a tactic (e.g., 'a'): ").strip().lower()
        if choice in choices:
            return filter_by_metadata(conversations, "tactic", choices[choice])
        print("Invalid choice.")


def _prompt_max_rows(total: int) -> int:
    while True:
        try:
            n = int(input(f"Number of rows to analyze (1-{total}): "))
            if 1 <= n <= total:
                return n
        except ValueError:
            pass
        print(f"Enter a number between 1 and {total}.")


def collect_analyzed_row_indices(
    *,
    csv_paths: list[Path] | None = None,
    output_dir: Path | None = None,
    skip_paths: set[Path] | None = None,
) -> set[int]:
    """Collect unique ``row_index`` values from prior experiment result CSVs."""
    skip = {Path(p).resolve() for p in (skip_paths or set())}
    paths: list[Path] = []
    if csv_paths:
        paths.extend(Path(p) for p in csv_paths)
    if output_dir is not None:
        paths.extend(Path(output_dir).glob("*.csv"))

    row_indices: set[int] = set()
    seen_files: set[Path] = set()
    for path in paths:
        path = Path(path).resolve()
        if path in skip or path in seen_files or not path.is_file():
            continue
        seen_files.add(path)
        try:
            df = pd.read_csv(path, usecols=["row_index"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        row_indices.update(int(v) for v in df["row_index"].dropna().unique())

    return row_indices


def resolve_exclude_row_indices(
    *,
    exclude_rows: list[int] | None = None,
    exclude_from_outputs: bool = False,
    output_dir: Path = OUTPUT_DIR,
    skip_output: Path | None = None,
    extra_csvs: list[Path] | None = None,
) -> set[int]:
    """Build the set of row indices to omit before random sampling."""
    excluded = set(exclude_rows or [])
    if exclude_from_outputs:
        skip = {skip_output.resolve()} if skip_output else set()
        excluded |= collect_analyzed_row_indices(
            csv_paths=extra_csvs,
            output_dir=output_dir,
            skip_paths=skip,
        )
    return excluded


async def run_analysis(
    *,
    config_path: str,
    data_path: Path,
    dataset_format: DatasetFormat,
    llm_type: str,
    model_name: str | None,
    llm_provider: str | None = None,
    max_rows: int | None,
    random_sample: int | None,
    batch_index: int | None,
    random_seed: int | None,
    min_turns: int,
    output_csv: Path | None,
    llm_calls_log: Path | None,
    interactive: bool,
    human_only: bool = False,
    exclude_row_indices: set[int] | None = None,
    batch_row_indices: list[int] | None = None,
    random_sampling: bool = False,
) -> None:
    started_at = datetime.now(timezone.utc)
    conversations, resolved_format, resolved_path = load_conversations(
        data_path, dataset_format
    )
    conversations = filter_min_turns(conversations, min_turns=min_turns)
    if not conversations:
        raise SystemExit(f"No conversations with at least {min_turns} turns.")

    if interactive:
        if resolved_format != DatasetFormat.HARMBENCH:
            raise SystemExit("Interactive mode requires a HarmBench dataset with a 'tactic' column.")
        conversations = _prompt_tactic(conversations)
        max_rows = _prompt_max_rows(len(conversations))

    if exclude_row_indices:
        excluded_seen = sorted(
            c["row_index"] for c in conversations if c["row_index"] in exclude_row_indices
        )
        conversations = [
            c for c in conversations if c["row_index"] not in exclude_row_indices
        ]
        if excluded_seen:
            print(
                f"Excluded {len(excluded_seen)} already-analyzed conversation(s): "
                f"{excluded_seen}"
            )
        if not conversations:
            raise SystemExit("No eligible conversations left after exclusions.")

    if random_sample is not None:
        if random_sample > len(conversations):
            raise SystemExit(
                f"--random-sample {random_sample} exceeds eligible size ({len(conversations)})."
            )
        if batch_row_indices is not None:
            resolved_batch_index = 0 if batch_index is None else batch_index
            by_index = {c["row_index"]: c for c in conversations}
            missing = set(batch_row_indices) - set(by_index)
            if missing:
                raise SystemExit(
                    f"Batch rows not found in dataset: {sorted(missing)}"
                )
            conversations = [by_index[i] for i in batch_row_indices]
            print(
                f"Batch {resolved_batch_index} (fixed rows, n={len(batch_row_indices)}): "
                f"row indices {batch_row_indices}"
            )
        elif random_sampling:
            rng_seed = 42 if random_seed is None else random_seed
            rng = random.Random(rng_seed)
            conversations = rng.sample(conversations, random_sample)
            indices = sorted(c["row_index"] for c in conversations)
            print(f"Random sample (random_seed={rng_seed}): row indices {indices}")
        else:
            resolved_batch_index = 0 if batch_index is None else batch_index
            conversations = select_batch_conversations(
                conversations,
                batch_index=resolved_batch_index,
                batch_size=random_sample,
            )
            indices = sorted(c["row_index"] for c in conversations)
            print(
                f"Batch {resolved_batch_index} (size={random_sample}, CSV order): "
                f"row indices {indices}"
            )

    elif max_rows is not None:
        conversations = conversations[:max_rows]

    if human_only:
        conversations = apply_human_only_mode(conversations)
        llm_type = "gpt_human_only"
        print("Analysis mode: human messages only (assistant text omitted)")

    resolved_llm_log = llm_calls_log
    if resolved_llm_log is None and output_csv is not None:
        resolved_llm_log = llm_calls_log_path_for_output(output_csv)

    analyzer = ConversationAnalyzer(
        config_path=config_path,
        llm_type=llm_type,
        model_name=model_name,
        llm_provider=llm_provider,
        llm_calls_log=resolved_llm_log,
        human_only=human_only,
    )

    try:
        for item in conversations:
            metadata = dict(item.get("metadata") or {})
            metadata["dataset_file"] = resolved_path.name
            metadata["dataset_format"] = resolved_format.value
            metadata["num_turns"] = len(item["llm_pairs"])
            metadata["analysis_mode"] = "human_only" if human_only else "full"
            if random_sample is not None and not random_sampling:
                metadata["batch_index"] = 0 if batch_index is None else batch_index
            await analyzer.analyze_conversation_row(
                item["row_index"], item["llm_pairs"], metadata
            )

        if output_csv is not None:
            analyzer.save_results_csv(output_csv)
        else:
            print(f"Done. Transitions recorded: {len(analyzer.recorded_results)}")
    finally:
        analyzer.close()

    if resolved_llm_log is not None:
        print(f"LLM call log: {resolved_llm_log}")

    if output_csv is not None:
        risk_cfg = analyzer.config.risk
        config_snapshot = {
            "risk": {
                "weights": {
                    "alpha": risk_cfg.weights.alpha,
                    "beta": risk_cfg.weights.beta,
                    "gamma": risk_cfg.weights.gamma,
                },
                "pattern_weights": risk_cfg.pattern_weights.to_dict(),
                "warn_threshold": risk_cfg.warn_threshold,
                "block_threshold": risk_cfg.block_threshold,
            }
        }
        args_snapshot = {
            "config_path": config_path,
            "data_path": str(resolved_path),
            "dataset_format": resolved_format.value,
            "prompt_template": llm_type,
            "resolved_model": analyzer.active_model,
            "resolved_provider": analyzer.llm_manager.config.provider,
            "min_turns": min_turns,
            "max_rows": max_rows,
            "random_sample": random_sample,
            "batch_index": batch_index,
            "random_seed": random_seed,
            "random_sampling": random_sampling,
            "human_only": human_only,
            "exclude_row_indices": sorted(exclude_row_indices) if exclude_row_indices else None,
            "batch_row_indices": batch_row_indices,
        }
        manifest_path = write_run_manifest(
            output_csv,
            args=args_snapshot,
            config_snapshot=config_snapshot,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            num_conversations=len(conversations),
            num_transitions=len(analyzer.recorded_results),
        )
        print(f"Run manifest: {manifest_path}")


def default_llm_calls_log(data_path: Path, model_name: str) -> Path:
    slug = model_name.replace("/", "_").replace(":", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / f"{_output_name_stem(data_path)}_{slug}_{stamp}.llm_calls.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TCA conversation analysis (batch or interactive).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m intent.cli --data red_queen_full_llama3_70b.csv --format nested --max-rows 5\n"
            "  python -m intent.cli --data red_queen_full_llama3_70b.csv --random-sample 5 --batch 0\n"
            "  python -m intent.cli --data harmbench_behaviors.csv --interactive\n"
        ),
    )
    parser.add_argument("--data", help="CSV dataset (searches dataset/, archive/, cwd)")
    parser.add_argument(
        "--format", choices=[f.value for f in DatasetFormat], default=DatasetFormat.AUTO.value
    )
    parser.add_argument("--model", help="Model id (OpenRouter or Ollama name)")
    parser.add_argument(
        "--provider",
        choices=["openrouter", "ollama"],
        default=None,
        help="LLM backend (default: config.yaml llm.provider)",
    )
    parser.add_argument(
        "--prompt-template", default="gpt", choices=["gpt", "claude", "gemini", "llama"]
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--max-rows", type=int, default=None, help="Analyze first N eligible conversations")
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Skip conversations with fewer than N turns (default: 2)",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Select N conversations per batch (--batch) or via --random",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Disjoint batch index for --random-sample (CSV pool order, default: 0)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="RNG sample instead of fixed --batch slicing",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="RNG seed for --random (default: 42)",
    )
    parser.add_argument(
        "--exclude-rows",
        type=int,
        nargs="+",
        default=None,
        help="Row indices to skip (in addition to --exclude-from-outputs)",
    )
    parser.add_argument(
        "--exclude-from-outputs",
        action="store_true",
        help="Skip rows already present in red_queen_outputs/*.csv",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--llm-log",
        type=Path,
        default=None,
        help="JSONL log for each LLM call (default: <output>.llm_calls.jsonl)",
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--show-data-format", action="store_true")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    args = build_parser().parse_args()

    if args.show_data_format:
        print(DATA_FORMAT_HELP.strip())
        return
    if not args.data:
        build_parser().error("--data is required (or use --show-data-format)")

    data_path = resolve_data_path(args.data)
    llm_cfg = load_config(args.config, provider=args.provider)
    model_name = args.model or llm_cfg.model_name
    llm_provider = args.provider or llm_cfg.provider
    output_csv = None if args.interactive else (
        args.output or default_output_path(data_path, model_name)
    )

    llm_calls_log = args.llm_log
    if llm_calls_log is None and output_csv is not None:
        llm_calls_log = llm_calls_log_path_for_output(output_csv)
    elif llm_calls_log is None:
        llm_calls_log = default_llm_calls_log(data_path, model_name)

    print(f"Dataset: {data_path}\nProvider: {llm_provider}\nModel: {model_name}")
    if output_csv:
        print(f"Output CSV: {output_csv}")
    if llm_calls_log:
        print(f"LLM call log: {llm_calls_log}")

    exclude_row_indices = resolve_exclude_row_indices(
        exclude_rows=args.exclude_rows,
        exclude_from_outputs=args.exclude_from_outputs,
        skip_output=output_csv,
    )

    asyncio.run(
        run_analysis(
            config_path=args.config,
            data_path=data_path,
            dataset_format=DatasetFormat(args.format),
            llm_type=args.prompt_template,
            model_name=args.model,
            llm_provider=llm_provider,
            max_rows=args.max_rows,
            random_sample=args.random_sample,
            batch_index=args.batch,
            random_seed=args.random_seed,
            min_turns=args.min_turns,
            output_csv=output_csv,
            llm_calls_log=llm_calls_log,
            interactive=args.interactive,
            exclude_row_indices=exclude_row_indices,
            random_sampling=args.random,
        )
    )


if __name__ == "__main__":
    main()
