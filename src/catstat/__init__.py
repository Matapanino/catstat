"""catstat -- unified CPU/GPU statistical categorical encoding.

Leakage-safe target encoding generalized to arbitrary statistics, with one sklearn-compatible API.
Runs on CPU (pandas/numpy) today; the GPU path (cuDF/CuPy) is parity-validated but auto-selection
stays on CPU until it is faster (see docs/roadmap.md and docs/known_issues.md, KI-020).
"""

from __future__ import annotations

from .count_encoder import CountEncoder
from .frequency_encoder import FrequencyEncoder
from .target_encoder import TargetEncoder

__all__ = ["TargetEncoder", "CountEncoder", "FrequencyEncoder", "__version__"]
__version__ = "0.3.0"
