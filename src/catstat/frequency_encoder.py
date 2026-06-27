"""``FrequencyEncoder`` -- ``CountEncoder(normalize=True)`` as a named class."""

from __future__ import annotations

from .count_encoder import CountEncoder


class FrequencyEncoder(CountEncoder):
    """Encode each category by its training frequency (count / n). Unseen categories map to 0.0.

    ``numeric`` (and ``cardinality_threshold`` / ``n_bins`` / ``binning``) opt numeric columns into
    binning exactly as on :class:`CountEncoder`; a binned numeric column then takes each row's
    **bin frequency** -- a normalized histogram. See :class:`CountEncoder` for the full description.
    """

    def __init__(
        self,
        cols="auto",
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
        super().__init__(
            cols=cols,
            normalize=True,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            backend=backend,
            output=output,
            numeric=numeric,
            cardinality_threshold=cardinality_threshold,
            n_bins=n_bins,
            binning=binning,
            min_bin_size=min_bin_size,
        )
