from __future__ import annotations

import random
import re
from typing import Optional

from datasets import Dataset, load_dataset

HF_REPO = "twangodev/devpost-hacks-judgments"
FRONTIER_MODEL = "Qwen/Qwen3.5-27B"
VERDICT_RE = re.compile(r"VERDICT:\s*(A|B|tie|invalid)", re.IGNORECASE)


def load_frontier(hackathon: Optional[str] = None) -> Dataset:
    ds = load_dataset(HF_REPO, hackathon or "all")
    return ds["train"].filter(lambda r: r["model"] == FRONTIER_MODEL)


def validate(ds: Dataset) -> None:
    required = [
        "messages", "judgment_id", "pair_id", "hackathon", "position",
        "project_a_id", "project_b_id", "verdict", "gt_a_result", "gt_b_result", "model",
    ]
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    errors = []
    for row in ds:
        msgs = row["messages"]
        if len(msgs) != 3 or [m["role"] for m in msgs] != ["system", "user", "assistant"]:
            errors.append("malformed messages")
            break
        if row["verdict"] not in {"A", "B", "tie", "invalid"}:
            errors.append(f"unexpected verdict: {row['verdict']}")
            break
    if errors:
        raise ValueError("; ".join(errors))

    no_verdict = sum(
        1 for row in ds
        if not VERDICT_RE.search(row["messages"][2]["content"])
    )
    if no_verdict:
        print(f"  warning: {no_verdict} assistant messages have no parseable VERDICT line")


def _winner_project_id(row: dict) -> str:
    return row["project_a_id"] if row["verdict"] == "A" else row["project_b_id"]


def _position_consistent_pair_ids(ds: Dataset) -> set[str]:
    by_pair: dict[str, dict[str, str]] = {}
    for row in ds:
        by_pair.setdefault(row["pair_id"], {})[row["position"]] = _winner_project_id(row)

    return {
        pair_id
        for pair_id, winners_by_position in by_pair.items()
        if winners_by_position.get("ab") is not None
        and winners_by_position.get("ab") == winners_by_position.get("ba")
    }


def split(
    ds: Dataset,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset, set, set]:
    """Split by pair_id so both position-swapped rows land in the same split."""
    trainable = ds.filter(
        lambda r: r["verdict"] in ("A", "B")
        and VERDICT_RE.search(r["messages"][2]["content"]) is not None
    )
    n_pairs_before_bias_filter = len(set(trainable["pair_id"]))
    consistent_pairs = _position_consistent_pair_ids(trainable)
    trainable = trainable.filter(lambda r: r["pair_id"] in consistent_pairs)
    n_pairs_dropped = n_pairs_before_bias_filter - len(consistent_pairs)
    if n_pairs_dropped:
        print(f"  dropped {n_pairs_dropped} position-inconsistent pairs")

    unique_pairs = list(consistent_pairs)
    rng = random.Random(seed)
    rng.shuffle(unique_pairs)

    n_test = int(len(unique_pairs) * test_size)
    test_pairs = set(unique_pairs[:n_test])
    train_pairs = set(unique_pairs[n_test:])

    # rename verdict → answer so training/eval code has a stable field name
    trainable = trainable.map(lambda r: {"answer": r["verdict"]})

    train_ds = trainable.filter(lambda r: r["pair_id"] in train_pairs)
    test_ds = trainable.filter(lambda r: r["pair_id"] in test_pairs)

    return train_ds, test_ds, train_pairs, test_pairs
