"""``FrequencyEncoder`` -- ``CountEncoder(normalize=True)`` as a named class."""

from __future__ import annotations

from .count_encoder import CountEncoder


class FrequencyEncoder(CountEncoder):
    """Encode each category by its training frequency (count / n). Unseen categories map to 0.0."""

    def __init__(
        self,
        cols="auto",
        handle_unknown="value",
        handle_missing="value",
        backend="auto",
        output="auto",
    ):
        super().__init__(
            cols=cols,
            normalize=True,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            backend=backend,
            output=output,
        )
