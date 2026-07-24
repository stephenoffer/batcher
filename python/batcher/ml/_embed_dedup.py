"""In-batch embedding deduplication — encode each distinct text once, gather back.

Web and log corpora repeat text heavily (boilerplate, repeated queries, near-empty rows),
and embedding is the expensive stage. Running every duplicate through the model wastes the
GPU on a forward pass whose answer is already known. `embed_unique` collapses a batch's
texts to the distinct set, encodes those, and scatters the vectors back to every row that
shared a text — the per-text output of an inference encoder is deterministic, so the result
is identical to encoding each row, only cheaper.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def embed_unique(texts: list, encode: Callable[[list], Any]) -> Any:
    """Encode only the distinct `texts`, then gather to one vector per original row.

    `encode` maps a list of texts to a 2-D ``(n, dim)`` array. When every text is already
    distinct the batch is encoded directly — the gather would only add a copy for no saving.
    """
    if not texts:
        return encode(texts)
    uniques, inverse = _unique(texts)
    if len(uniques) == len(texts):
        return encode(texts)
    import numpy as np

    return np.asarray(encode(uniques))[np.asarray(inverse, dtype=np.intp)]


def _unique(texts: list) -> tuple[list, list[int]]:
    """First-seen-order distinct texts and, per row, the index of its distinct text."""
    seen: dict[Any, int] = {}
    order: list = []
    inverse: list[int] = []
    for text in texts:
        index = seen.get(text)
        if index is None:
            index = len(order)
            seen[text] = index
            order.append(text)
        inverse.append(index)
    return order, inverse
