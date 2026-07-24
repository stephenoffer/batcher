"""`Dataset.__getattr__`'s answer: an actionable error for a name Batcher does not have.

Three answers, most specific first: a known-absent ecosystem API (the big table in
`_dataset_table`), a column accessed as an attribute (the pandas ``df.amount`` habit,
which Batcher declines because a column named ``filter`` would shadow a method), and
finally a near-miss method name. Everything here is message-only: no branch changes what
a query computes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import absent_error
from batcher.api.dataset.compat.guidance._dataset_table import DATASET_UNSUPPORTED

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

__all__ = ["attribute_error_for"]


def _method_names(ds: Dataset) -> list[str]:
    """The public method and property names a user could have meant, for did-you-mean."""
    return [n for n in dir(type(ds)) if not n.startswith("_")]


def attribute_error_for(ds: Dataset, name: str) -> AttributeError:
    """Build the `AttributeError` for a failed `Dataset` attribute lookup.

    Args:
        ds: The dataset the attribute was looked up on.
        name: The attribute name that was not found.

    Returns:
        An `AttributeError` whose message explains the absence and names the
        Batcher spelling to use instead.
    """
    if name in DATASET_UNSUPPORTED:
        return AttributeError(f"Dataset has no attribute {name!r}. {DATASET_UNSUPPORTED[name]}")

    # A column name: the pandas attribute-access habit. Batcher does not add it
    # because a column named e.g. "filter" would shadow a method, so point at the
    # two spellings that can never be ambiguous.
    try:
        columns = ds.columns
    except Exception:  # pragma: no cover - a malformed plan must not mask the real error
        columns = []
    if name in columns:
        return AttributeError(
            f"Dataset has no attribute {name!r}, but it is a column. Batcher does not "
            "expose columns as attributes (a column could shadow a method). Use "
            f"ds[{name!r}] for the expression, or bt.col({name!r}) to build one."
        )

    return absent_error("Dataset", name, DATASET_UNSUPPORTED, [*_method_names(ds), *columns])
