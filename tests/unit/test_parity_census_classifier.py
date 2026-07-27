"""The parity census's own classification, pinned.

`tools/parity/duckdb_census.py` produces the numbers in
`docs/internals/competitor_parity_census.md`. Nothing else checks it, and two bugs in it
each moved the headline figure by more than any real change to the engine has:

* counting DuckDB's 135 `icu_collate_*` entries — one per locale, one capability — as 135
  missing functions, which reported 79% parity as 54%;
* deciding "gap" from any failure rather than from an error that proves the function is
  unreachable, which reported eight working functions as missing and then, once that was
  fixed with a case-sensitive marker compared against lowercased text, laundered eleven
  real gaps into "unprobed".

Both were invisible: the census still ran, still printed a number, and the number was
wrong in whichever direction the bug leaned. So the classification is tested directly.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_duckdb_census",
    pathlib.Path(__file__).resolve().parents[2] / "tools" / "parity" / "duckdb_census.py",
)


@pytest.fixture(scope="module")
def census():
    """The census module, loaded by path — `tools/` is not an installed package."""
    module = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(module)
    return module


# --- what counts as out of scope --------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["icu_collate_de", "icu_collate_zh_cn", "duckdb_functions", "pragma_version", "current_schema"],
)
def test_locale_and_catalog_entries_are_out_of_scope(census, name):
    assert census.out_of_scope(name) is not None


@pytest.mark.parametrize("name", ["upper", "list_concat", "regexp_extract", "icu_sort_key", "map"])
def test_a_real_data_function_is_in_scope(census, name):
    """`icu_sort_key` is deliberately here: it is one function, not one per locale, so the
    prefix rule must not swallow it along with `icu_collate_*`."""
    assert census.out_of_scope(name) is None


# --- what proves a function is absent ---------------------------------------


@pytest.mark.parametrize(
    "detail",
    [
        "NotImplementedError: unknown function 'list_cat': it is not a supported SQL function",
        "NotImplementedError: unsupported SQL expression: Decode",
        "NotImplementedError: unsupported aggregate: bitstring_agg",
    ],
)
def test_an_unreachable_function_is_proven_absent(census, detail):
    assert census._proves_absent(detail)


@pytest.mark.parametrize(
    "detail",
    [
        "PlanError: could not parse SQL (dialect 'duckdb'): Required keyword: 'this' missing",
        "NotImplementedError: datetime format 'abc' is not supported",
        "ValueError: array() requires at least one element",
    ],
)
def test_a_harness_side_failure_is_not_proof_of_absence(census, detail):
    """Each of these is the census's own synthesized call being wrong, not Batcher missing
    a function: all three name functions that work when called properly."""
    assert not census._proves_absent(detail)


def test_the_markers_are_lowercase(census):
    """The comparison lowercases the message, so a marker with any capital never matches.

    This is the bug that hid eleven real gaps, and it is silent — the census keeps running
    and keeps printing a number.
    """
    assert all(marker == marker.lower() for marker in census._ABSENT)
