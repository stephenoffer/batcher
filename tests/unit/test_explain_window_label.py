"""`explain()` labelled every partitioned window "global".

A window's IR carries `partition_keys`, but `observe.dag.describe` rendered windows with the
aggregate describer, which reads `group_keys`. The read always missed, so a correctly
partitioned `rank().over("g")` printed as `[global · rank]` and read as a whole-table rank.
Display only, like the union label beside it, and caught the same way: by someone reading the
label and doubting the plan.

Both directions are asserted, since a check on one alone passes against a constant label.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _window_line(ds) -> str:
    lines = [
        line
        for line in ds.explain().splitlines()
        if line.strip().lstrip("└├│─ ").startswith("window")
    ]
    assert len(lines) == 1, f"expected exactly one window row:\n{ds.explain()}"
    return lines[0]


@pytest.fixture
def ds():
    return bt.from_pydict({"g": [1, 1, 2], "v": [3, 1, 2]})


def test_a_partitioned_window_names_its_partition(ds):
    line = _window_line(ds.with_columns(r=bt.col("v").rank().over("g")))
    assert "over g" in line and "global" not in line, line


def test_an_unpartitioned_window_is_labelled_global(ds):
    line = _window_line(ds.with_columns(r=bt.col("v").cum_sum().over(order_by="v")))
    assert "global" in line, line
