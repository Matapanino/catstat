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
    histogram when ``normalize=True``). ``"direct"``/``"bin"`` force one strategy. ``binning`` also
    accepts an explicit **edge array** (``[0, 18, 65, 120]`` -> 3 bins, applied to every binned
    column) or a per-column ``{col: strategy-or-edges}`` **dict**; explicit edges set the bin count
    (``n_bins`` is then ignored for that column), and ``binning`` controls only *how* a column is
    binned (*whether* stays with ``numeric`` + ``cardinality_threshold``). Bin edges come from
    feature values only (there is no ``y``), so they are a plain function of the training column (or
    of the user's explicit edges). ``min_bin_size`` (an int count, or a float fraction of ``n``)
    merges adjacent sparse bins of the *computed* ``quantile``/``uniform`` strategies; explicit edge
    arrays are left exact. ``cardinality_threshold`` accepts an int (absolute unique count) or a
    float in (0, 1] (unique/n ratio). Inspect the fitted ``numeric_cols_`` / ``numeric_strategy_`` /
    ``bin_edges_`` attrs.
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
        min_bin_size=None,
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
        self.min_bin_size = min_bin_size

    def _is_supervised(self) -> bool:
        return False

    def _resolve_stat_specs(self):
        return resolve_stats(["frequency" if self.normalize else "count"])
