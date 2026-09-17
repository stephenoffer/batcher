"""A CSV Batcher wrote must read back as the table Batcher wrote, nulls included.

The writer already distinguished the two empty strings a text column can hold: it emits
``""`` for the empty string and a bare, unquoted field for NULL. The reader did not.
`pyarrow.csv` defaults `strings_can_be_null` to False, so **every NULL in a text column came
back as `""`** -- silently, on files Batcher had produced itself, and with no error anywhere.
Numeric columns were unaffected, which is what kept it invisible: the same round trip is
exact for every other type.

It had also been *recorded* as a property of CSV, in the fidelity matrix, as "CSV cannot
distinguish an empty string from a null". CSV distinguishes them by quoting, DuckDB reads
this file back exactly, and Batcher's own writer relies on the distinction -- so the entry
described the reader and named the format, which is how a defect becomes a documented
limitation and stops being looked at.

The literal string ``"NA"`` is in the fixture on purpose. Turning `strings_can_be_null` on
without `quoted_strings_can_be_null` off would read every quoted null *token* as NULL too,
trading one silent corruption for another.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.integration


@pytest.fixture
def nulls() -> pa.Table:
    return pa.table(
        {
            "s": pa.array(["x", "", None, "NA", "null", " ", "q"]),
            "n": pa.array([1, 2, None, 4, 5, 6, None], pa.int64()),
            "f": pa.array([1.5, None, 3.5, None, 5.5, 6.5, 7.5], pa.float64()),
            "b": pa.array([True, False, None, True, False, None, True], pa.bool_()),
        }
    )


def test_a_csv_batcher_wrote_reads_back_unchanged(tmp_path, nulls):
    path = str(tmp_path / "rt.csv")
    bt.from_arrow(nulls).write.csv(path)
    assert bt.read.csv(path).collect().to_pydict() == nulls.to_pydict()


def test_an_empty_string_and_a_null_stay_different(tmp_path, nulls):
    """The whole point: collapsing them is exactly what the reader used to do."""
    path = str(tmp_path / "rt.csv")
    bt.from_arrow(nulls).write.csv(path)
    back = bt.read.csv(path).to_pydict()["s"]
    assert back[1] == "", "the empty string became something else"
    assert back[2] is None, "the null came back as a value"


def test_a_quoted_null_token_is_still_a_string(tmp_path):
    """Quoting is what separates data from a sentinel, and the writer quotes every string.

    Without this, a text column holding the literal ``NA`` -- a country code, a product
    grade, an initialism -- would round-trip to NULL.
    """
    tbl = pa.table({"s": pa.array(["NA", "NULL", "NaN", "N/A", "nan", "null"])})
    path = str(tmp_path / "tokens.csv")
    bt.from_arrow(tbl).write.csv(path)
    assert bt.read.csv(path).collect().to_pydict() == tbl.to_pydict()


def test_a_bare_null_token_written_by_something_else_is_still_a_null(tmp_path):
    """The other half: an unquoted sentinel in a foreign file keeps reading as NULL.

    pandas' `na_values` default treats these as missing, and a hand-written or exported CSV
    is where they come from. Only the *quoted* form is protected above.
    """
    path = tmp_path / "foreign.csv"
    path.write_text("s,n\nx,1\nNA,2\n,3\n")
    assert bt.read.csv(str(path)).collect().to_pydict() == {
        "s": ["x", None, None],
        "n": [1, 2, 3],
    }
