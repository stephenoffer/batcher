"""Turning pyarrow's CSV read failures into errors that say what to do about them.

CSV is the format that fails most often and explains itself least, and its two common
failures have **opposite fixes** — so telling them apart is the whole value here:

- *inference was shown too little*: a column is integral for a million rows and then
  holds ``"N/A"``. The file is fine; the declared type is wrong. Fix: declare the type.
- *the bytes are not text*: the file is not valid UTF-8. Declaring a type does nothing.
  Fix: re-encode it, skip it, or ask for the raw bytes deliberately.

Offering the first fix for the second failure costs a whole debugging session, which is
why these are separate errors rather than one message mentioning both.

Split out of `csv.py` because it is diagnosis, not parsing: the reader decides *what to
read*, this decides *what to say when that fails*.
"""

from __future__ import annotations

import contextlib

import pyarrow as pa

from batcher._internal.errors import FormatError, SchemaError

__all__ = ["invalid_utf8_error", "mismatch_reported"]


def invalid_utf8_error(path: str, detail: str) -> FormatError:
    """The error for a CSV holding bytes that are not valid UTF-8.

    This is the one CSV failure pyarrow does not report as a failure at all: asked to
    infer, it quietly types the column `binary` and hands back what looks like a
    successful read. A column's type then depends on whether a stray byte happened to
    land in the inference block, and nothing downstream can tell that apart from a column
    of genuine bytes. DuckDB and Polars both reject such a file.

    Args:
        path: The file whose bytes could not be decoded.
        detail: What pyarrow reported, or which columns were affected.

    Returns:
        The error to raise.
    """
    return FormatError(
        f"CSV file {path!r} contains bytes that are not valid UTF-8: {detail}. "
        "Re-encode the file as UTF-8, pass on_error='skip' to drop it and read the rest, "
        "or declare the column as binary to keep the raw bytes — "
        'bt.read.csv(path, schema=pa.schema([("col", pa.binary()), ...])).'
    )


@contextlib.contextmanager
def mismatch_reported(path: str):
    """Report a conversion failure as either a type mismatch or undecodable bytes.

    A CSV column is typed from the first block, so a value further down that does not fit
    is not a corrupt file — it is inference having been shown too little. The raw error
    names the offending value but not the inferred type, nor that the type is declarable,
    which is the whole of the fix.

    Undecodable bytes arrive by the same route (inference saw a clean first block, the bad
    byte is further down) but are not that problem, so they get their own diagnosis.

    Args:
        path: The file being read, for the error message.

    Yields:
        Nothing; the block runs inside the reporting context.
    """
    try:
        yield
    except pa.ArrowInvalid as exc:
        if "invalid utf8" in str(exc).lower():
            raise invalid_utf8_error(path, str(exc)) from exc
        raise SchemaError(
            f"CSV value does not fit the inferred column type in {path!r}: {exc}. "
            "The schema is inferred from the file's first block, so a value further down "
            "may not fit it. Declare the type instead — "
            'bt.read.csv(path, schema=pa.schema([("col", pa.string()), ...])).'
        ) from exc
