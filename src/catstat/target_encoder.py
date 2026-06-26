"""``TargetEncoder`` -- the supervised, cross-fitted, generalized target encoder."""

from __future__ import annotations

from ._base import _BaseStatEncoder
from ._stats import resolve_stats


class TargetEncoder(_BaseStatEncoder):
    """Leakage-safe target encoding, generalized over a set of statistics.

    ``fit_transform`` is leakage-safe for the training set; ``fit().transform()`` learns full-data
    encodings and is the path for *new* data. ``stats`` accepts built-ins
    (``mean``/``count``/``frequency``/``var``/``std``/``median``/``min``/``max``/``skew``) and
    custom ``(name, callable)`` aggregations. ``scheme`` selects how the *mean* is cross-fitted on
    the training set: ``"kfold"`` (default, out-of-fold), ``"loo"`` (leave-one-out), or
    ``"ordered"`` (CatBoost-style ordered target statistics). ``loo``/``ordered`` apply to the mean
    only (use with ``stats=["mean"]``, optionally plus count/frequency).

    Parameters mirror ``docs/proposals/target-encoder-library-design.md`` §3.
    """

    def __init__(
        self,
        cols="auto",
        stats=("mean",),
        target_type="auto",
        smooth="auto",
        cv=5,
        scheme="kfold",
        shuffle=True,
        random_state=None,
        handle_unknown="value",
        handle_missing="value",
        multi_feature_mode="independent",
        min_samples_category=1,
        backend="auto",
        output="auto",
    ):
        self.cols = cols
        self.stats = stats
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.scheme = scheme
        self.shuffle = shuffle
        self.random_state = random_state
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.multi_feature_mode = multi_feature_mode
        self.min_samples_category = min_samples_category
        self.backend = backend
        self.output = output

    def _is_supervised(self) -> bool:
        return True

    def _resolve_stat_specs(self):
        return resolve_stats(self.stats)
