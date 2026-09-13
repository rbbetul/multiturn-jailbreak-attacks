"""Fixed benign batch row indices for reproducible experiments."""

from __future__ import annotations

import random

BENIGN_EXCLUDED_ROWS = frozenset({91})
# Rows dropped from batchB result CSVs when patching (91 = suspicious probe, 15 = jailbreak-style).
BENIGN_BATCHB_REMOVED_ROWS = frozenset({91, 15})
BENIGN_REPLACEMENT_PICK_SEED = 42

# Row 31: train-wheel / mirror physics trivia (4 turns, all mundane).
BENIGN_BATCH_REPLACEMENT_ROW = 31

BENIGN_FIXED_BATCHES: dict[int, list[int]] = {
    0: [7, 8, 11, 23, 48],
    1: [1, 13, BENIGN_BATCH_REPLACEMENT_ROW, 54, 94],
}


def pick_replacement_row(eligible_row_indices: list[int], *, seed: int = BENIGN_REPLACEMENT_PICK_SEED) -> int:
    """Pick one replacement row deterministically from eligible indices."""
    return random.Random(seed).choice(eligible_row_indices)


def get_benign_batch_rows(batch_index: int) -> list[int]:
    """Return the fixed row_index list for a benign batch index."""
    if batch_index in BENIGN_FIXED_BATCHES:
        return list(BENIGN_FIXED_BATCHES[batch_index])
    raise ValueError(
        f"No fixed benign batch for index {batch_index}. "
        f"Known batches: {sorted(BENIGN_FIXED_BATCHES)}"
    )
