"""The `Dataset` builder package.

`Dataset` is the lazy, immutable, fluent entry point (`frame.py`); its longer
transformation bodies live as free functions in `_build.py` so the class stays a
thin builder. The public import path ``batcher.api.dataset.Dataset`` is preserved.
"""

from __future__ import annotations

from batcher.api.dataset.frame import Dataset, GroupBy

__all__ = ["Dataset", "GroupBy"]
