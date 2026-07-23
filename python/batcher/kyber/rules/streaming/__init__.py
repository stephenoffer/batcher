"""Kyber rule families for streaming (unbounded-input) plans.

Rules that are specific to unbounded execution: pushdown through the watermark-bounded
streaming operators, and rewrites gated on whether an input is a stream at all
(`kyber.streaming` supplies that analysis).

Importing this package runs each module's ``@rule`` decorators, registering every rule
into ``kyber.registry.DEFAULT_REGISTRY``. Re-export/registration only — no logic here.
"""

from __future__ import annotations

from batcher.kyber.rules.streaming import blocking as _blocking  # noqa: F401
from batcher.kyber.rules.streaming import state as _state  # noqa: F401
from batcher.kyber.rules.streaming import watermark as _watermark  # noqa: F401
from batcher.kyber.rules.streaming import windows as _windows  # noqa: F401

__all__: list[str] = []
