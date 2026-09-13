"""`explain()` labelled every union backwards, and nothing noticed for want of one assertion.

`observe.dag.describe` read a key named `all` off the union's IR node. The node carries
`distinct` (`Union.to_ir`), so the read returned `None` on every plan ever rendered and the
label was the negation of the truth: a `UNION ALL` printed as `[distinct]` and a `UNION` as
`[all]`. It is a display-only defect, which is exactly why it survived — the plan was right,
the rows were right, and only the sentence describing them was wrong.

Both directions are asserted here, because an assertion on one of them passes just as happily
against the inverted mapping.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture
def pair():
    return bt.from_pydict({"x": [1, 2, 3]}), bt.from_pydict({"x": [2]})


def _union_line(ds) -> str:
    lines = [line for line in ds.explain().splitlines() if line.strip().startswith("union")]
    assert len(lines) == 1, f"expected exactly one union row:\n{ds.explain()}"
    return lines[0]


def test_a_union_all_is_labelled_all(pair):
    left, right = pair
    assert "[all]" in _union_line(left.union(right))


def test_a_distinct_union_is_labelled_distinct(pair):
    left, right = pair
    assert "[distinct]" in _union_line(left.union(right, distinct=True))


def test_the_label_follows_the_ir_field_the_node_actually_carries(pair):
    """The cause, pinned: the describer must read `distinct`, which is what `to_ir` emits."""
    left, right = pair
    ir = left.union(right)._plan.to_ir()
    assert "distinct" in ir and "all" not in ir, ir
