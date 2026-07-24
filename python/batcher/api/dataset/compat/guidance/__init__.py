"""Migration-error guidance: the traceback is the documentation.

A migrant does not read the API reference first. They type what they already know from
pandas, Polars, or PySpark, and read the traceback. So when an attribute lookup fails on
a `Dataset` or a `GroupBy`, the error that comes back names why the API is absent and what
to type in Batcher instead. Each object routes its failed lookups through its own builder
here; the shared rendering is `batcher._internal.errors.absent_error`, and the redirect
tables live in the `_*_table` modules.

`Expr`'s equivalent guidance lives beside `Expr` itself in `plan/expr_ir` — `plan` is a
neutral layer that must not import `api`, so its table cannot live in this package.
"""

from __future__ import annotations

from batcher.api.dataset.compat.guidance.dataset import attribute_error_for
from batcher.api.dataset.compat.guidance.groupby import groupby_attribute_error

__all__ = [
    "attribute_error_for",
    "groupby_attribute_error",
]
