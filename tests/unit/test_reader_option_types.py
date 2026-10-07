"""The static reader-option types and docstrings agree with the runtime vocabulary (AP-421, AP-010).

The runtime check is each format's `OptionSpec`: an unknown keyword already fails there with
a suggestion. The `TypedDict`s give an editor and a type checker the same vocabulary, and
the reader docstrings list it for `help()`. Both are copies, so both are pinned here to the
spec they copy: an option added to a spec and not to its type, or not to its docstring,
fails this file.
"""

from __future__ import annotations

import inspect

import pytest

from batcher.api.io_namespace.reader import Reader
from batcher.io.base._options import (
    BASE_SOURCE_ALIASES,
    BASE_SOURCE_OPTIONS,
    READ_CALL_OPTIONS,
)
from batcher.io.detect import _PARTITIONED_OPTIONS
from batcher.io.formats.semistructured.json import _JSON_READ_OPTIONS, JSONReadOptions
from batcher.io.formats.structured._csv_options.spec import READ_SPEC, CsvReadOptions
from batcher.io.formats.structured.parquet.partitions import ParquetReadOptions

#: Every keyword `bt.read.parquet` accepts at runtime: what `FileSource` takes, its base
#: aliases, the option that routes a Hive tree to the partition-aware reader, and what the
#: generic read call consumes. Parquet has no `OptionSpec` of its own, so this is it.
_PARQUET_ACCEPTED = {
    *BASE_SOURCE_OPTIONS,
    *BASE_SOURCE_ALIASES,
    *(_PARTITIONED_OPTIONS - {"schema_mode"}),
}

_CASES = {
    "csv": (CsvReadOptions, set(READ_SPEC.accepted)),
    "json": (JSONReadOptions, set(_JSON_READ_OPTIONS.accepted)),
    "parquet": (ParquetReadOptions, _PARQUET_ACCEPTED),
}


def _keys(typed: type) -> set[str]:
    return set(typed.__optional_keys__) | set(typed.__required_keys__)


@pytest.mark.parametrize("fmt", sorted(_CASES))
def test_the_typed_options_are_exactly_the_runtime_vocabulary(fmt):
    typed, accepted = _CASES[fmt]
    # Aliases are included on purpose: a migrating script spells options the way pandas
    # and Polars do, and a type checker that rejected those would reject working code.
    assert _keys(typed) == accepted | set(READ_CALL_OPTIONS)


@pytest.mark.parametrize("fmt", sorted(_CASES))
def test_the_reader_signature_names_its_typed_options(fmt):
    typed, _ = _CASES[fmt]
    annotation = inspect.signature(getattr(Reader, fmt)).parameters["opts"].annotation
    assert annotation == f"Unpack[{typed.__name__}]"


@pytest.mark.parametrize(
    ("fmt", "canonical"),
    [
        ("csv", READ_SPEC._canonical),
        ("json", _JSON_READ_OPTIONS._canonical),
    ],
)
def test_the_docstring_lists_every_canonical_option(fmt, canonical):
    doc = getattr(Reader, fmt).__doc__ or ""
    own = [name for name in canonical if name not in BASE_SOURCE_OPTIONS]
    assert own  # the control: the spec has options of its own to list
    missing = [name for name in own if f"``{name}``" not in doc]
    assert not missing, f"bt.read.{fmt}'s docstring does not mention {missing}"


def test_the_csv_writer_docstring_lists_every_canonical_option():
    from batcher.api.io_namespace.writer import Writer
    from batcher.io.formats.structured._csv_options.spec import WRITE_SPEC

    doc = Writer.csv.__doc__ or ""
    missing = [name for name in WRITE_SPEC._canonical if f"``{name}``" not in doc]
    assert not missing, f"ds.write.csv's docstring does not mention {missing}"
