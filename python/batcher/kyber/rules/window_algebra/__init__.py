"""Window rule families: a window function that is a cheaper window function in disguise.

`extra/window_rules` and `extra/window_extra` already prune dead window outputs, dedupe
identical specs, drop constant partition and order keys, and collapse adjacent windows.
This package covers the orthogonal case, the one those rules cannot see: a spec whose
*arguments* make it equal to a different, specialized function.

The rewrite matters twice over. The specialized form is the cheaper kernel — picking the
first row of a frame does not need the counter that finding its n-th row does — and, more
importantly, it is the *canonical* spelling, so `dedupe_window_functions` can then merge
it with a sibling that was already written that way. Two specs that compute the same
column but spell it differently otherwise survive to the data plane as two windows.
"""

from __future__ import annotations

from batcher.kyber.rules.window_algebra import positional as _positional  # noqa: F401

__all__: list[str] = []
