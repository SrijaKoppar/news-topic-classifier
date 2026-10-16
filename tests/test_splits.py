"""Leakage tests. They use small synthetic data, so no real files are needed."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.config import LABELS
from src.data import assert_no_leakage, make_split


def synthetic_df(n=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = np.array([5, 4, 3, 3, 3, 2, 2, 1, 1], dtype=float)
    return pd.DataFrame({
        "id": [f"r{i}" for i in range(n)],
        "headline": [f"headline {i}" for i in range(n)],
        "short_description": [f"description {i}" for i in range(n)],
        "category": rng.choice(LABELS, size=n, p=w / w.sum()),
        "split_a": np.where(rng.random(n) < 0.2, "test", "train"),
        "split_b": np.where(rng.random(n) < 0.3, "test", "train"),
    })


@pytest.mark.parametrize("col", ["split_a", "split_b"])
def test_no_leakage(col):
    df = synthetic_df()
    sd = make_split(df, col)
    assert_no_leakage(df, sd)
    assert set(sd.test["id"]) == set(df.loc[df[col] == "test", "id"])


def test_reproducible_and_seed_sensitive():
    df = synthetic_df()
    a = make_split(df, "split_a", seed=1)
    b = make_split(df, "split_a", seed=1)
    c = make_split(df, "split_a", seed=2)
    assert list(a.val["id"]) == list(b.val["id"])
    assert set(a.val["id"]) != set(c.val["id"])


def test_validation_is_stratified():
    sd = make_split(synthetic_df(), "split_a")
    tr = sd.train["category"].value_counts(normalize=True)
    va = sd.val["category"].value_counts(normalize=True).reindex(tr.index).fillna(0)
    assert (tr - va).abs().max() < 0.02


def test_splits_are_independent():
    df = synthetic_df()
    scrambled = df.copy()
    scrambled["split_b"] = scrambled["split_b"].sample(frac=1, random_state=1).values
    a1 = make_split(df, "split_a")
    a2 = make_split(scrambled, "split_a")
    assert list(a1.val["id"]) == list(a2.val["id"])


def test_rejects_bad_split_values():
    df = synthetic_df()
    df.loc[0, "split_a"] = "validation"
    with pytest.raises(ValueError):
        make_split(df, "split_a")


def test_rejects_duplicate_ids():
    df = synthetic_df()
    df.loc[1, "id"] = df.loc[0, "id"]
    with pytest.raises(ValueError):
        make_split(df, "split_a")


def test_checker_catches_a_planted_leak():
    df = synthetic_df()
    sd = make_split(df, "split_a")
    leaky = dataclasses.replace(sd, train=pd.concat([sd.train, sd.test.head(1)]))
    with pytest.raises(AssertionError):
        assert_no_leakage(df, leaky)