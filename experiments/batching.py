"""Disjoint batch selection for experiment pools."""

from __future__ import annotations

from typing import Any


def select_batch_conversations(
    conversations: list[dict[str, Any]],
    *,
    batch_index: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Return slice ``[batch_index * batch_size : (batch_index + 1) * batch_size]``.

    Conversations are taken in ascending ``row_index`` order (CSV pool order).
    No shuffling is applied.
    """
    pool = sorted(conversations, key=lambda c: c["row_index"])
    start = batch_index * batch_size
    end = start + batch_size
    if end > len(pool):
        last = (len(pool) - batch_size) // batch_size
        raise SystemExit(
            f"Batch {batch_index} needs {batch_size} conversations but only "
            f"{len(pool) - start} remain. Valid batch indices: 0..{last}."
        )
    return pool[start:end]
