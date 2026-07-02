"""Statistic registry.

M0: ``mean`` (supervised, cross-fitted), ``count``/``frequency`` (unsupervised). Phase 2 added the
dispersion/order stats ``var``/``std``/``median``/``min``/``max``. Phase 3 adds ``skew`` and
**custom-callable aggregations** (which subsume quantiles, IQR, etc.); the stats arc adds ``kurt``
and reworks ``skew``/``kurt`` to power-sum moments shared by both backends.

Non-mean target statistics are cross-fitted, continuous-target only, with **no principled
smoothing** (the smoothing honesty rule): order/shape stats never blend; a category below
``min_samples_category`` (or where the statistic is undefined) falls back to the global statistic.
Custom callables are CPU-only and must be order-independent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class StatSpec:
    """Metadata that drives how a statistic is computed, smoothed, named, and fallen back."""

    name: str
    smoothing: str  # "mean" (principled), "none", or "dispersion_optin" (heuristic, default off)
    class_expanded: bool  # multiclass: emit one column per class?
    target_dependent: bool  # uses y -> must be cross-fitted in fit_transform
    name_infix: str  # output column infix, e.g. "te_mean", "count", "freq", "te_std"
    gpu_supported: bool = True  # cudf groupby supports this agg
    continuous_only: bool = False  # requires a continuous (regression) target
    func: Callable | None = None  # custom aggregation callable: f(values: ndarray) -> scalar


_REGISTRY: dict[str, StatSpec] = {
    "mean": StatSpec("mean", "mean", True, True, "te_mean"),
    "count": StatSpec("count", "none", False, False, "count"),
    "frequency": StatSpec("frequency", "none", False, False, "freq"),
    "var": StatSpec("var", "none", False, True, "te_var", continuous_only=True),
    "std": StatSpec("std", "none", False, True, "te_std", continuous_only=True),
    "median": StatSpec("median", "none", False, True, "te_median", continuous_only=True),
    "min": StatSpec("min", "none", False, True, "te_min", continuous_only=True),
    "max": StatSpec("max", "none", False, True, "te_max", continuous_only=True),
    # skew/kurt are reconstructed from per-category power sums (category_moments), which both
    # backends provide -- neither pandas' .skew() nor a cuDF equivalent is needed.
    "skew": StatSpec("skew", "none", False, True, "te_skew", continuous_only=True),
    "kurt": StatSpec("kurt", "none", False, True, "te_kurt", continuous_only=True),
}


def _custom_spec(name: str, fn: Callable) -> StatSpec:
    # Custom aggregations are CPU-only, cross-fitted, continuous-target, no smoothing.
    return StatSpec(
        name=name,
        smoothing="none",
        class_expanded=False,
        target_dependent=True,
        name_infix=name,
        gpu_supported=False,
        continuous_only=True,
        func=fn,
    )


def resolve_stats(stats) -> list[StatSpec]:
    """Normalize ``stats`` to a list of :class:`StatSpec`.

    Accepts: a string; a list whose items are built-in stat names (str) or ``(name, callable)``
    custom aggregations; or a dict ``{name: callable}`` of custom aggregations.
    """
    if isinstance(stats, str):
        items = [stats]
    elif isinstance(stats, dict):
        items = list(stats.items())
    else:
        items = list(stats)
    if not items:
        raise ValueError("stats must name at least one statistic.")

    specs: list[StatSpec] = []
    for it in items:
        if isinstance(it, str):
            if it in _REGISTRY:
                specs.append(_REGISTRY[it])
            elif it == "quantile":
                raise ValueError(
                    "stat='quantile' needs a quantile level; pass it as a custom aggregation, "
                    "e.g. stats=[('q90', lambda v: np.quantile(v, 0.9))]."
                )
            else:
                raise ValueError(f"Unknown stat {it!r}. Known: {sorted(_REGISTRY)}.")
        elif (
            isinstance(it, tuple)
            and len(it) == 2
            and isinstance(it[0], str)
            and callable(it[1])
        ):
            specs.append(_custom_spec(it[0], it[1]))
        else:
            raise ValueError(
                f"Invalid stats entry {it!r}: use a stat name (str) or a (name, callable) pair."
            )
    return specs
