"""catstat -- unified CPU/GPU statistical categorical encoding.

Leakage-safe target encoding generalized to arbitrary statistics
(``mean``/``count``/``frequency``/``var``/``std``/``median``/``min``/``max``/``skew``/``kurt``/
``woe``/custom callables), with one sklearn-compatible API and three principled mean smoothers
(fixed m-estimate; ``"auto"`` empirical-Bayes -- exactly scikit-learn's formula; ``"sigmoid"``,
category_encoders' blend).

Runs on CPU (pandas/numpy) and on GPU (RAPIDS cuDF/CuPy), CPU/GPU-parity-validated. Pass a cuDF
DataFrame and the whole encode stays **device-resident** (factorize, cross-fitting, gather, cuDF
output) -- ~2.6-13x faster than CPU on a T4 at 100k-10M rows. For pandas-origin data the
host<->device copies eat the win, so ``backend="auto"`` resolves to CPU there (KI-020); cuDF
input routes to the GPU automatically. See docs/roadmap.md and docs/known_issues.md.
"""

from __future__ import annotations

from .count_encoder import CountEncoder
from .frequency_encoder import FrequencyEncoder
from .target_encoder import TargetEncoder

__all__ = ["TargetEncoder", "CountEncoder", "FrequencyEncoder", "__version__"]
__version__ = "0.5.1"
