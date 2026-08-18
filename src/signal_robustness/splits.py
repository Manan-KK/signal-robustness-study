"""Walk-forward and purged combinatorial split construction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationSplit:
    split_id: str
    kind: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purged_rows: int
    embargoed_rows: int


def walk_forward_splits(
    observations: pd.DataFrame,
    *,
    minimum_train: int = 40,
    test_size: int = 20,
    embargo_rows: int = 2,
) -> tuple[ValidationSplit, ...]:
    """Create anchored test blocks with resolved-label purge and train-side gap."""

    frame = _split_frame(observations)
    _positive_int("minimum_train", minimum_train)
    _positive_int("test_size", test_size)
    _nonnegative_int("embargo_rows", embargo_rows)
    splits = []
    start = minimum_train + embargo_rows
    sequence = 0
    while start < len(frame):
        stop = min(start + test_size, len(frame))
        test = np.arange(start, stop, dtype=int)
        candidate_stop = max(0, start - embargo_rows)
        candidates = np.arange(candidate_stop, dtype=int)
        test_start = frame.loc[start, "decision_date"]
        keep = frame.loc[candidates, "outcome_end_date"].lt(test_start).to_numpy()
        train = candidates[keep]
        purged = int((~keep).sum())
        if len(train) >= minimum_train:
            splits.append(
                ValidationSplit(
                    split_id=f"wfa-{sequence:02d}",
                    kind="walk_forward",
                    train_indices=tuple(int(value) for value in train),
                    test_indices=tuple(int(value) for value in test),
                    purged_rows=purged,
                    embargoed_rows=min(embargo_rows, start),
                )
            )
            sequence += 1
        start = stop
    if not splits:
        raise ValueError("not enough observations for a walk-forward split")
    return tuple(splits)


def combinatorial_purged_splits(
    observations: pd.DataFrame,
    *,
    groups: int = 6,
    test_groups: int = 2,
    embargo_rows: int = 2,
) -> tuple[ValidationSplit, ...]:
    """Create all declared group combinations and remove interval leakage."""

    frame = _split_frame(observations)
    _positive_int("groups", groups)
    _positive_int("test_groups", test_groups)
    _nonnegative_int("embargo_rows", embargo_rows)
    if groups < 3 or groups > len(frame):
        raise ValueError("groups must be between three and the observation count")
    if test_groups >= groups:
        raise ValueError("test_groups must be smaller than groups")
    blocks = tuple(np.asarray(block, dtype=int) for block in np.array_split(np.arange(len(frame)), groups))
    splits = []
    for sequence, selected_groups in enumerate(combinations(range(groups), test_groups)):
        selected = tuple(blocks[index] for index in selected_groups)
        test = np.sort(np.concatenate(selected))
        candidates = np.setdiff1d(np.arange(len(frame)), test, assume_unique=True)
        intervals = tuple(
            (
                frame.loc[block[0], "decision_date"],
                frame.loc[block, "outcome_end_date"].max(),
            )
            for block in selected
        )
        overlap = np.array(
            [
                any(
                    _intervals_overlap(
                        frame.loc[index, "decision_date"],
                        frame.loc[index, "outcome_end_date"],
                        start,
                        end,
                    )
                    for start, end in intervals
                )
                for index in candidates
            ],
            dtype=bool,
        )
        remaining = candidates[~overlap]
        embargo_positions = set()
        for _, interval_end in intervals:
            after_interval = np.flatnonzero(
                frame["decision_date"].gt(interval_end).to_numpy()
            )
            if len(after_interval) == 0:
                continue
            first = int(after_interval[0])
            embargo_positions.update(
                range(first, min(len(frame), first + embargo_rows))
            )
        embargo = np.array(
            [int(index) in embargo_positions for index in remaining], dtype=bool
        )
        train = remaining[~embargo]
        splits.append(
            ValidationSplit(
                split_id=f"cpcv-{sequence:02d}",
                kind="combinatorial_purged",
                train_indices=tuple(int(value) for value in train),
                test_indices=tuple(int(value) for value in test),
                purged_rows=int(overlap.sum()),
                embargoed_rows=int(embargo.sum()),
            )
        )
    expected = math.comb(groups, test_groups)
    if len(splits) != expected:
        raise AssertionError("combinatorial split count is inconsistent")
    return tuple(splits)


def audit_splits(
    observations: pd.DataFrame,
    splits: tuple[ValidationSplit, ...],
) -> pd.DataFrame:
    """Return compact invariants without exposing observation-level data."""

    frame = _split_frame(observations)
    rows = []
    for split in splits:
        train = np.asarray(split.train_indices, dtype=int)
        test = np.asarray(split.test_indices, dtype=int)
        test_blocks = _contiguous_blocks(test)
        intervals = tuple(
            (
                frame.loc[block[0], "decision_date"],
                frame.loc[block, "outcome_end_date"].max(),
            )
            for block in test_blocks
        )
        overlaps = 0
        for index in train:
            overlaps += int(
                any(
                    _intervals_overlap(
                        frame.loc[index, "decision_date"],
                        frame.loc[index, "outcome_end_date"],
                        start,
                        end,
                    )
                    for start, end in intervals
                )
            )
        rows.append(
            {
                "split_id": split.split_id,
                "kind": split.kind,
                "train_rows": len(train),
                "test_rows": len(test),
                "purged_rows": split.purged_rows,
                "embargoed_rows": split.embargoed_rows,
                "train_test_disjoint": int(set(train).isdisjoint(test)),
                "interval_overlap_count": overlaps,
            }
        )
    return pd.DataFrame(rows)


def _contiguous_blocks(indices: np.ndarray) -> tuple[np.ndarray, ...]:
    if len(indices) == 0:
        return ()
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return tuple(np.asarray(block, dtype=int) for block in np.split(indices, boundaries))


def _intervals_overlap(
    left_start: pd.Timestamp,
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
    right_end: pd.Timestamp,
) -> bool:
    return bool(left_start <= right_end and left_end >= right_start)


def _split_frame(observations: pd.DataFrame) -> pd.DataFrame:
    missing = {"decision_date", "outcome_end_date"} - set(observations.columns)
    if missing:
        raise ValueError(f"observations are missing columns: {', '.join(sorted(missing))}")
    frame = observations.loc[:, ["decision_date", "outcome_end_date"]].copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="coerce")
    frame["outcome_end_date"] = pd.to_datetime(
        frame["outcome_end_date"], errors="coerce"
    )
    if frame.empty or frame.isna().any().any():
        raise ValueError("split dates must be nonempty and valid")
    if not frame["decision_date"].is_monotonic_increasing:
        raise ValueError("decision dates must be increasing")
    if frame["decision_date"].duplicated().any():
        raise ValueError("decision dates must be unique")
    if frame["outcome_end_date"].le(frame["decision_date"]).any():
        raise ValueError("outcome intervals must end after decisions")
    return frame.reset_index(drop=True)


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
