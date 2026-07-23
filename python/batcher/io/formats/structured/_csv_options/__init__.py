"""CSV keyword vocabulary — the pandas/Polars spellings, folded to pyarrow options.

`csv.py` owns reading and writing; this package owns *translation*: the accepted keyword
tables (`spec`), the coercion of a `dtype` entry or a pandas `header=` into something
typed (`dtypes`), and the resolved option objects that build pyarrow's
`ReadOptions`/`ParseOptions`/`ConvertOptions`/`WriteOptions` (`build`).
"""

from __future__ import annotations

from batcher.io.formats.structured._csv_options.build import (
    CSVReadOptions,
    CSVWriteOptions,
    resolve_read_options,
    resolve_write_options,
)
from batcher.io.formats.structured._csv_options.spec import READ_SPEC, WRITE_SPEC

__all__ = [
    "READ_SPEC",
    "WRITE_SPEC",
    "CSVReadOptions",
    "CSVWriteOptions",
    "resolve_read_options",
    "resolve_write_options",
]
