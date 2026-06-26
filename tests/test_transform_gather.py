"""Regression tests for the integer-code gather transform path (KI-031).

These lock the behavior the gather rewrite must preserve exactly:
- a unit whose stats have *different* category orders (mean = first-appearance, count =
  count-descending) must still gather each column's values into the right category after the
  canonical-index alignment;
- combination (tuple-key) units gather and fall back correctly for unseen joint keys;
- a tiny-n category is baked to the global statistic *at fit* (so it is a KNOWN code at transform
  and never trips handle_unknown), while a truly unseen category still follows handle_unknown.
"""

import numpy as np
import pandas as pd
import pytest

from catstat import TargetEncoder


def _name_idx(enc, substring):
    names = list(enc.get_feature_names_out())
    return next(i for i, n in enumerate(names) if substring in n)


def test_mixed_order_stats_gather_to_correct_categories():
    # 'a' appears first but is rare; 'c' appears later but is the most frequent, so value_counts
    # order (count-desc: c, b, a) differs from groupby appearance order (a, b, c). The canonical
    # alignment must keep each stat matched to the right category.
    g = ["a", "b", "c", "b", "c", "c", "b", "c", "b", "c"]
    y = np.array([10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    enc = TargetEncoder(
        cols=["g"],
        stats=("mean", "count"),
        target_type="continuous",
        smooth=0.0,
        handle_unknown="value",
    ).fit(pd.DataFrame({"g": g}), y)

    out = np.asarray(enc.transform(pd.DataFrame({"g": ["a", "b", "c"]})))
    mean_col, count_col = _name_idx(enc, "mean"), _name_idx(enc, "count")

    s = pd.Series(y).groupby(pd.Series(g)).mean()
    counts = pd.Series(g).value_counts()
    for i, cat in enumerate(["a", "b", "c"]):
        assert out[i, mean_col] == pytest.approx(s[cat])
        assert out[i, count_col] == pytest.approx(counts[cat])


def test_combination_gather_known_and_unknown_joint_key():
    a = ["x", "x", "y", "y", "z", "z"]
    b = ["p", "q", "p", "q", "p", "q"]
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    enc = TargetEncoder(
        cols=["a", "b"],
        multi_feature_mode="combination",
        stats=("mean",),
        target_type="continuous",
        smooth=0.0,
        handle_unknown="value",
    ).fit(pd.DataFrame({"a": a, "b": b}), y)

    # (x, p) is seen exactly once -> its own mean; (x, UNSEEN) is an unseen joint key -> global.
    out = np.asarray(enc.transform(pd.DataFrame({"a": ["x", "x"], "b": ["p", "UNSEEN"]})))
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(enc.target_mean_)


def test_tiny_n_is_baked_global_unseen_follows_handle_unknown():
    # median respects min_samples_category: 'rare' (n=1 < 5) is baked to the global median *at fit*.
    g = ["common"] * 10 + ["rare"]
    y = np.arange(11, dtype=float)
    enc = TargetEncoder(
        cols=["g"],
        stats=("median",),
        target_type="continuous",
        min_samples_category=5,
        handle_unknown="error",
    ).fit(pd.DataFrame({"g": g}), y)

    # 'rare' is a KNOWN code holding the baked global median -> gather returns it, no raise.
    out = np.asarray(enc.transform(pd.DataFrame({"g": ["rare"]})))
    assert out[0, 0] == pytest.approx(np.median(y))
    # a genuinely unseen category still trips handle_unknown="error".
    with pytest.raises(ValueError, match="unknown categories"):
        enc.transform(pd.DataFrame({"g": ["NOPE"]}))
