"""Memoization primitives shared across the layer-3 subsystems.

`identity` holds `IdentityMemo`, the identity-keyed, key-pinning memo the optimizer, cost
model and plan utilities use for pure functions of immutable objects.

Layer 0: `kyber`, `carbonite`, `core` and `governance` may not import one another, so a
memo shape they all need lives here rather than being hand-written at each site.
"""

from __future__ import annotations

from batcher._internal.memo.identity import MISSING, IdentityMemo, Missing

__all__ = ["MISSING", "IdentityMemo", "Missing"]
