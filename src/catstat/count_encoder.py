"""``CountEncoder`` -- unsupervised category-prevalence encoding (no target, no cross-fit)."""

from __future__ import annotations

from ._base import _BaseStatEncoder
from ._stats import resolve_stats


class CountEncoder(_BaseStatEncoder):
    """Encode each category by its training count (or frequency if ``normalize=True``).

    Unsupervised: no ``y`` is used, so there is no target leakage and ``fit_transform`` equals
    ``fit().transform()``. Unseen categories map to 0 (count) / 0.0 (frequency).

    ``numeric`` opts numeric columns into encoding (default ``"ignore"`` keeps today's behavior:
    ``cols="auto"`` skips numerics). ``"auto"`` routes each numeric column by cardinality -- at most
    ``cardinality_threshold`` distinct values are counted **directly** (each value a category),
    otherwise the column is **binned** into ``n_bins`` (``binning="quantile"`` equal-frequency, or
    ``"uniform"`` equal-width) and each row takes its **bin's count** -- a histogram (a normalized
    histogram when ``normalize=True``). ``"direct"``/``"bin"`` force one strategy. Bin edges come
    from feature values only (there is no ``y``), so they are a plain function of the training
    column. ``cardinality_threshold`` accepts an int (absolute unique count) or a float in (0, 1]
    (unique/n ratio). Inspect the fitted ``numeric_cols_`` / ``numeric_strategy_`` / ``bin_edges_``
    attrs.
    """

    def __init__(
        self,
        cols="auto",
        normalize=False,
        handle_unknown="value",
        handle_missing="value",
        backend="auto",
        output="auto",
        numeric="ignore",
        cardinality_threshold=10,
        n_bins=10,
        binning="quantile",
    ):
        self.cols = cols
        self.normalize = normalize
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.backend = backend
        self.output = output
        self.numeric = numeric
        self.cardinality_threshold = cardinality_threshold
        self.n_bins = n_bins
        self.binning = binning

    def _is_supervised(self) -> bool:
        return False

    def _resolve_stat_specs(self):
        return resolve_stats(["frequency" if self.normalize else "count"])
