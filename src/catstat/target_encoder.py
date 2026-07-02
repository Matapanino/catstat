"""``TargetEncoder`` -- the supervised, cross-fitted, generalized target encoder."""

from __future__ import annotations

from ._base import _BaseStatEncoder
from ._stats import resolve_stats


class TargetEncoder(_BaseStatEncoder):
    """Leakage-safe target encoding, generalized over a set of statistics.

    ``fit_transform`` is leakage-safe for the training set; ``fit().transform()`` learns full-data
    encodings and is the path for *new* data. ``stats`` accepts built-ins
    (``mean``/``count``/``frequency``/``var``/``std``/``median``/``min``/``max``/``skew``/``kurt``/
    ``woe``, the last binary-only: ``logit(smoothed p) - logit(prior)``, unknown -> 0.0) and
    custom ``(name, callable)`` aggregations. ``scheme`` selects how the *mean* is cross-fitted on
    the training set: ``"kfold"`` (default, out-of-fold), ``"loo"`` (leave-one-out), or
    ``"ordered"`` (CatBoost-style ordered target statistics). ``loo``/``ordered`` apply to the mean
    only (use with ``stats=["mean"]``, optionally plus count/frequency).

    ``numeric`` opts numeric columns into encoding (default ``"ignore"`` keeps today's behavior:
    ``cols="auto"`` skips numerics). ``"auto"`` routes each numeric column by cardinality -- at most
    ``cardinality_threshold`` distinct values are encoded **directly** (each value a category),
    otherwise the column is **binned** into ``n_bins`` (``binning="quantile"`` equal-frequency, or
    ``"uniform"`` equal-width) and the bins are target-encoded; ``"direct"``/``"bin"`` force one
    strategy. ``binning`` also accepts an explicit **edge array** (``[0, 18, 65, 120]`` -> 3 bins,
    applied to every binned column) or a per-column ``{col: strategy-or-edges}`` **dict** (e.g.
    ``{"age": [0, 18, 65, 120], "income": "quantile"}``); explicit edges set the bin count, so
    ``n_bins`` is ignored for that column, and ``binning`` only controls *how* a column is binned --
    *whether* it is binned stays with ``numeric`` + ``cardinality_threshold``. Bin edges come from
    feature values only (computed strategies) or from the user (explicit edges) -- never ``y`` -- so
    the per-bin encoding stays out-of-fold. ``min_bin_size`` (an int count, or a float fraction of
    ``n``) merges adjacent sparse bins of the *computed* ``quantile``/``uniform`` strategies so each
    surviving bin holds enough rows for a stable encoding; explicit edge arrays are left exact.
    ``cardinality_threshold`` takes an int (absolute unique count) or a float in (0, 1] (unique/n
    ratio). Inspect the fitted ``numeric_strategy_`` / ``bin_edges_`` attrs.

    ``interactions`` adds one joint target-encoded column per group: e.g.
    ``interactions=[["a", "b"]]`` encodes the ``(a, b)`` pair as one category (named ``"a+b"``) on
    top of the independent ``cols``. It generalizes ``multi_feature_mode="combination"`` (which
    encodes a single joint column over *all* cols and nothing independent). Joint keys run on both
    backends via int64 mixed-radix codes.

    ``smooth`` selects the mean/probability smoothing: a float ``m`` (m-estimate), ``"auto"``
    (empirical-Bayes, the default), or ``"sigmoid"`` / ``("sigmoid", k, f)`` -- the
    category_encoders blend ``w = 1/(1 + exp(-(n - k)/f))`` toward the prior (bare string uses
    their defaults ``k=20, f=10``; singletons take the prior outright, matching their override).
    ``"sigmoid"`` is kfold-only (no loo/ordered analogue).

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
        numeric="ignore",
        cardinality_threshold=10,
        n_bins=10,
        binning="quantile",
        min_bin_size=None,
        interactions=None,
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
        self.numeric = numeric
        self.cardinality_threshold = cardinality_threshold
        self.n_bins = n_bins
        self.binning = binning
        self.min_bin_size = min_bin_size
        self.interactions = interactions

    def _is_supervised(self) -> bool:
        return True

    def _resolve_stat_specs(self):
        return resolve_stats(self.stats)
