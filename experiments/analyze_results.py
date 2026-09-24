"""
Summarize TCA batch results: decision distribution, progressive risk by turn,
call latency, and token usage — per the metrics list in EXPERIMENT_PLAN.md.

Stdlib only (no pandas) so it runs anywhere, including a laptop without the
project's Python environment set up.

Usage::

    python experiments/analyze_results.py red_queen_outputs/batch_0000_n100_....csv
    python experiments/analyze_results.py red_queen_outputs/batch_0000_n100_A.csv \\
        red_queen_outputs/batch_0000_n100_B.csv ...   # prints a comparison table too
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def load_results(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_llm_calls(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.is_file():
        return []
    calls = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                calls.append(json.loads(line))
    return calls


def load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def decision_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    if not total:
        return {}
    counts: dict[str, int] = {}
    for row in rows:
        d = row.get("decision", "unknown")
        counts[d] = counts.get(d, 0) + 1
    return {d: 100.0 * n / total for d, n in sorted(counts.items())}


def flagged_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    flagged = sum(1 for r in rows if r.get("decision") in ("warn", "block"))
    return 100.0 * flagged / len(rows)


def progressive_risk_by_turn(rows: list[dict[str, Any]]) -> dict[int, float]:
    by_turn: dict[int, list[float]] = {}
    for row in rows:
        try:
            turn = int(row["transition_index"])
            risk = float(row["progressive_risk"])
        except (KeyError, ValueError, TypeError):
            continue
        by_turn.setdefault(turn, []).append(risk)
    return {t: statistics.mean(vs) for t, vs in sorted(by_turn.items())}


def latency_stats(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Median/range of duration_ms, excluding the first call (cold-start load)."""
    if not calls:
        return {}
    ordered = sorted(calls, key=lambda c: c.get("timestamp", ""))
    rest = ordered[1:] if len(ordered) > 1 else ordered
    durations = [c["duration_ms"] for c in rest if c.get("duration_ms") is not None]
    if not durations:
        return {}
    return {
        "n": len(durations),
        "median_ms": statistics.median(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "first_call_ms": ordered[0].get("duration_ms"),
    }


def token_usage(calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    with_usage = [c for c in calls if c.get("usage")]
    if not with_usage:
        return None
    prompt_tokens = [c["usage"].get("prompt_tokens", 0) for c in with_usage]
    completion_tokens = [c["usage"].get("completion_tokens", 0) for c in with_usage]
    return {
        "n_calls_with_usage": len(with_usage),
        "n_calls_total": len(calls),
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "mean_prompt_tokens": statistics.mean(prompt_tokens),
        "mean_completion_tokens": statistics.mean(completion_tokens),
    }


def summarize_cell(csv_path: Path) -> dict[str, Any]:
    rows = load_results(csv_path)
    calls = load_llm_calls(csv_path.with_suffix(".llm_calls.jsonl"))
    manifest = load_manifest(csv_path.with_suffix(".manifest.json"))

    conversations = sorted({row["row_index"] for row in rows if "row_index" in row})

    return {
        "path": csv_path,
        "n_transitions": len(rows),
        "n_conversations": len(conversations),
        "decision_distribution": decision_distribution(rows),
        "flagged_rate": flagged_rate(rows),
        "risk_by_turn": progressive_risk_by_turn(rows),
        "latency": latency_stats(calls),
        "tokens": token_usage(calls),
        "manifest": manifest,
    }


def print_cell(summary: dict[str, Any]) -> None:
    print(f"\n=== {summary['path'].name} ===")
    print(f"Transitions: {summary['n_transitions']}  |  Conversations: {summary['n_conversations']}")

    dist = summary["decision_distribution"]
    if dist:
        dist_str = ", ".join(f"{d} {pct:.1f}%" for d, pct in dist.items())
        print(f"Decision distribution: {dist_str}")
    print(f"Flagged rate (warn+block): {summary['flagged_rate']:.1f}%")

    by_turn = summary["risk_by_turn"]
    if by_turn:
        turns_str = ", ".join(f"t{t}={r:.2f}" for t, r in list(by_turn.items())[:10])
        print(f"Progressive risk by turn: {turns_str}")

    latency = summary["latency"]
    if latency:
        print(
            f"Latency (excl. first call, n={latency['n']}): "
            f"median={latency['median_ms']:.0f}ms  "
            f"range={latency['min_ms']:.0f}-{latency['max_ms']:.0f}ms  "
            f"(first call was {latency['first_call_ms']:.0f}ms — cold start)"
        )
    else:
        print("Latency: no llm_calls.jsonl found alongside this CSV")

    tokens = summary["tokens"]
    if tokens:
        print(
            f"Tokens (n={tokens['n_calls_with_usage']}/{tokens['n_calls_total']} calls with usage): "
            f"mean prompt={tokens['mean_prompt_tokens']:.0f}  "
            f"mean completion={tokens['mean_completion_tokens']:.0f}  "
            f"total prompt={tokens['total_prompt_tokens']}  "
            f"total completion={tokens['total_completion_tokens']}"
        )
    else:
        print("Tokens: no usage data in llm_calls.jsonl (older run, or provider without usage capture)")

    manifest = summary["manifest"]
    if manifest:
        args = manifest.get("args", {})
        print(
            f"Manifest: model={args.get('resolved_model')}  "
            f"provider={args.get('resolved_provider')}  "
            f"batch_index={args.get('batch_index')}  "
            f"human_only={args.get('human_only')}  "
            f"git_commit={manifest.get('git_commit')}"
        )


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    print("\n=== Comparison across cells ===")
    header = f"{'file':<45} {'n':>5} {'flagged%':>9} {'median_ms':>10}"
    print(header)
    for s in summaries:
        median_ms = s["latency"].get("median_ms")
        median_str = f"{median_ms:.0f}" if median_ms is not None else "n/a"
        print(f"{s['path'].name:<45} {s['n_transitions']:>5} {s['flagged_rate']:>8.1f}% {median_str:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", type=Path, help="Result CSV(s) to summarize")
    args = parser.parse_args()

    summaries = []
    for csv_path in args.csv_paths:
        if not csv_path.is_file():
            print(f"Skipping missing file: {csv_path}")
            continue
        summaries.append(summarize_cell(csv_path))

    for summary in summaries:
        print_cell(summary)

    if len(summaries) > 1:
        print_comparison(summaries)


if __name__ == "__main__":
    main()
