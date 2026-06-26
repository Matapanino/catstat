"""catstat -- unified CPU/GPU statistical categorical encoding.

Leakage-safe target encoding generalized to arbitrary statistics, with one sklearn-compatible API.
M0 is CPU-only (pandas/numpy); see docs/roadmap.md.
"""

from __future__ import annotations

from .count_encoder import CountEncoder
from .frequency_encoder import FrequencyEncoder
from .target_encoder import TargetEncoder

__all__ = ["TargetEncoder", "CountEncoder", "FrequencyEncoder", "__version__"]
__version__ = "0.0.1"
