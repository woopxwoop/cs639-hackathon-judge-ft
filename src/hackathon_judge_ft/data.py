from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

HF_REPO = "twangodev/devpost-hacks-judgments"
FRONTIER_MODEL = "Qwen/Qwen3.5-27B"


def load_frontier_df(hackathon: Optional[str] = None) -> pd.DataFrame:
    if hackathon:
        ds = load_dataset(HF_REPO, hackathon)
    else:
        ds = load_dataset(HF_REPO)
    df = ds["train"].to_pandas()
    return df[df["model"] == FRONTIER_MODEL].reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    required = [
        "messages", "judgment_id", "pair_id", "hackathon", "position",
        "project_a_id", "project_b_id", "verdict", "gt_a_result", "gt_b_result", "model",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    def check_msgs(msgs):
        return (
            len(msgs) == 3
            and [m.get("role") for m in msgs] == ["system", "user", "assistant"]
        )

    bad = df[~df["messages"].apply(check_msgs)]
    if len(bad):
        raise ValueError(f"{len(bad)} rows have malformed messages")

    bad_verdicts = df[~df["verdict"].isin({"A", "B", "tie", "invalid"})]
    if len(bad_verdicts):
        raise ValueError(f"Unexpected verdict values: {bad_verdicts['verdict'].unique()}")

    def has_verdict(msgs):
        content = msgs[2]["content"] if len(msgs) > 2 else ""
        return bool(re.search(r"VERDICT:\s*(A|B|tie|invalid)", content, re.IGNORECASE))

    no_verdict = df[~df["messages"].apply(has_verdict)]
    if len(no_verdict):
        print(f"  warning: {len(no_verdict)} assistant messages have no parseable VERDICT line")


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset, set, set]:
    """Split by pair_id so both position-swapped rows for a pair land in the same split."""
    trainable = df[df["verdict"].isin(["A", "B"])]
    unique_pairs = trainable["pair_id"].unique().tolist()
    train_pairs, test_pairs = train_test_split(unique_pairs, test_size=test_size, random_state=seed)
    train_pairs, test_pairs = set(train_pairs), set(test_pairs)

    def to_records(subset_pairs):
        return [
            {
                "messages": row["messages"],
                "judgment_id": row["judgment_id"],
                "pair_id": row["pair_id"],
                "hackathon": row["hackathon"],
                "position": row["position"],
                "project_a_id": row["project_a_id"],
                "project_b_id": row["project_b_id"],
                "answer": row["verdict"],
                "gt_a_result": row["gt_a_result"],
                "gt_b_result": row["gt_b_result"],
            }
            for _, row in trainable[trainable["pair_id"].isin(subset_pairs)].iterrows()
        ]

    return (
        Dataset.from_list(to_records(train_pairs)),
        Dataset.from_list(to_records(test_pairs)),
        train_pairs,
        test_pairs,
    )
