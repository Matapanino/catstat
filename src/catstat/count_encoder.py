"""``CountEncoder`` -- unsupervised category-prevalence encoding (no target, no cross-fit)."""

from __future__ import annotations

from ._base import _BaseStatEncoder
from ._stats import resolve_stats


class CountEncoder(_BaseStatEncoder):
    """Encode each category by its training count (or frequency if ``normalize=True``).

    Unsupervised: no ``y`` is used, so there is no target leakage and ``fit_transform`` equals
    ``fit().transform()``. Unseen categories map to 0 (count) / 0.0 (frequency).
    """

    def __init__(
        self,
        cols="auto",
        normalize=False,
        handle_unknown="value",
        handle_missing="value",
        backend="auto",
        output="auto",
    ):
        self.cols = cols
        self.normalize = normalize
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.backend = backend
        self.output = output

    def _is_supervised(self) -> bool:
        return False

    def _resolve_stat_specs(self):
        return resolve_stats(["frequency" if self.normalize else "count"])
