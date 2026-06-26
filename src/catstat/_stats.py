"""Statistic registry.

Each statistic is described by a :class:`StatSpec`. M0 shipped ``mean`` (supervised, cross-fitted),
``count`` and ``frequency`` (unsupervised). Phase 2 adds the dispersion/order statistics
``var``/``std``/``median``/``min``/``max`` -- target-dependent (so cross-fitted), continuous-target
only, with **no principled smoothing** (the smoothing honesty rule): order stats never blend, and a
category below ``min_samples_category`` (or where the statistic is undefined) falls back to the
global statistic. ``quantile``/``skew``/custom callables remain Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatSpec:
    """Metadata that drives how a statistic is computed, smoothed, named, and fallen back."""

    name: str
    smoothing: str  # "mean" (principled), "none", or "dispersion_optin" (heuristic, default off)
    class_expanded: bool  # multiclass: emit one column per class?
    target_dependent: bool  # uses y -> must be cross-fitted in fit_transform
    name_infix: str  # output column infix, e.g. "te_mean", "count", "freq", "te_std"
    gpu_supported: bool = True  # cudf groupby supports all of these aggs
    continuous_only: bool = False  # requires a continuous (regression) target


_REGISTRY: dict[str, StatSpec] = {
    "mean": StatSpec("mean", "mean", True, True, "te_mean"),
    "count": StatSpec("count", "none", False, False, "count"),
    "frequency": StatSpec("frequency", "none", False, False, "freq"),
    "var": StatSpec("var", "none", False, True, "te_var", continuous_only=True),
    "std": StatSpec("std", "none", False, True, "te_std", continuous_only=True),
    "median": StatSpec("median", "none", False, True, "te_median", continuous_only=True),
    "min": StatSpec("min", "none", False, True, "te_min", continuous_only=True),
    "max": StatSpec("max", "none", False, True, "te_max", continuous_only=True),
}

# Designed but not implemented yet (see docs/roadmap.md Phase 3).
_DEFERRED = {"quantile", "skew"}


def resolve_stats(stats) -> list[StatSpec]:
    """Normalize the ``stats`` argument to a list of :class:`StatSpec`.

    Accepts a string or an iterable of strings. (Custom callables are Phase 3.)
    """
    if isinstance(stats, str):
        names = [stats]
    else:
        names = list(stats)
    if not names:
        raise ValueError("stats must name at least one statistic.")

    specs = []
    for n in names:
        if not isinstance(n, str):
            raise NotImplementedError(
                "Custom callable aggregations are planned for Phase 3; pass stat names for now."
            )
        if n in _REGISTRY:
            specs.append(_REGISTRY[n])
        elif n in _DEFERRED:
            raise NotImplementedError(
                f"stat={n!r} is designed but not implemented yet (Phase 3). See docs/roadmap.md."
            )
        else:
            raise ValueError(f"Unknown stat {n!r}. Known: {sorted(_REGISTRY)}.")
    return specs
