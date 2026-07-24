"""`GroupBy.__getattr__`'s answer: an actionable error for a grouped API Batcher lacks.

A pandas `GroupBy` invites per-group Python (`apply`, `transform`, `get_group`, looping);
Batcher's `GroupBy` only aggregates, and everything else is a window or a filter. So when
a migrant reaches for `gb.transform(...)` the traceback names the window that replaces it.
Message-only: nothing here changes what a query computes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import absent_error
from batcher.api.dataset.compat.guidance._groupby_table import GROUPBY_UNSUPPORTED

if TYPE_CHECKING:
    from batcher.api.groupby import GroupBy

__all__ = ["groupby_attribute_error"]


def groupby_attribute_error(gb: GroupBy, name: str) -> AttributeError:
    """Build the `AttributeError` for a failed `GroupBy` attribute lookup.

    Args:
        gb: The GroupBy the attribute was looked up on.
        name: The attribute name that was not found.

    Returns:
        An `AttributeError` that explains the absence and names the window or aggregate
        to use instead.
    """
    members = [n for n in dir(type(gb)) if not n.startswith("_")]
    return absent_error("GroupBy", name, GROUPBY_UNSUPPORTED, members)
