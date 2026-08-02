"""Arrow ↔ framework conversion — NumPy, PyTorch, pandas, polars, JAX.

A neutral layer, below every subsystem and above `io`: converting an Arrow batch into the
object a user's function expects is needed by the executor (`core.udf`), by the distributed
map path, by the public API's export verbs, and by the `ml` training-loop bridge. It used to
live in `ml`, which put a *front-end* package underneath the executor that runs it — one of
the upward edges `just lint-layers` now forbids.

Nothing here knows about plans, operators, or the engine. It converts arrays.
"""

from __future__ import annotations

from batcher.interop.arrays import arrays_to_torch, to_numpy_batches
from batcher.interop.formats import FORMATS, result_to_arrowable, to_format

__all__ = [
    "FORMATS",
    "arrays_to_torch",
    "result_to_arrowable",
    "to_format",
    "to_numpy_batches",
]
