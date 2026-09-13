"""Mirror terminal output to a log file during experiment runs."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


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
