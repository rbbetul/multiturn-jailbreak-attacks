"""Mirror terminal output to a log file during experiment runs."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, TextIO


class _TeeStream:
    """Write to the console and a log file at the same time."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return self.streams[0].isatty() if self.streams else False


@contextmanager
def tee_terminal_output(log_path: Path) -> Iterator[Path]:
    """
    Copy stdout and stderr to ``log_path`` while still printing to the terminal.

    Usage::

        with tee_terminal_output(Path("red_queen_outputs/run.log")):
            print("also saved to the log file")
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = log_path.open("w", encoding="utf-8")
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(stdout, log_file)
    sys.stderr = _TeeStream(stderr, log_file)

    try:
        yield log_path
    finally:
        sys.stdout = stdout
        sys.stderr = stderr
        log_file.close()


def log_path_for_output(output_csv: Path) -> Path:
    """Pair a results CSV with a log file that shares the same stem."""
    return Path(output_csv).with_suffix(".log")


def manifest_path_for_output(output_csv: Path) -> Path:
    """Pair a results CSV with a run-manifest file that shares the same stem."""
    return Path(output_csv).with_suffix(".manifest.json")


def _git_commit_hash() -> str | None:
    """Best-effort commit hash of the code that produced a run; None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def write_run_manifest(
    output_csv: Path,
    *,
    args: Dict[str, Any],
    config_snapshot: Dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    num_conversations: int,
    num_transitions: int,
) -> Path:
    """Write a per-run manifest: code version, args, resolved config, timing.

    Pairs with the results CSV so a run stays reproducible even after
    config.yaml or the code itself changes later.
    """
    manifest = {
        "git_commit": _git_commit_hash(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 1),
        "num_conversations": num_conversations,
        "num_transitions": num_transitions,
        "args": args,
        "config_snapshot": config_snapshot,
    }
    path = manifest_path_for_output(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
