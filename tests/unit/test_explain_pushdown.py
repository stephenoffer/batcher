"""`explain()` says which filter reached the source, and `expr_text` can render it.

A scan printed identically whether the plan had pushed a filter into it or was reading the
whole relation and filtering above. Those two plans differ by the entire table, and nothing
in the output told them apart — so the one question pushdown raises ("did it actually
happen?") had no answer short of reading the optimizer. Spark prints `PushedFilters:` and
DuckDB prints `Filters:` for exactly this reason.

The label is what the plan *offered*. Each backend then translates the subset it can
express, and the engine keeps its own `Filter` regardless, so an in-memory source that
ignores the offer entirely still shows it here — that is the honest report, since no source
can say what it did until it runs.
"""

from __future__ import annotations

import json

import pytest

import batcher as bt
from batcher.observe.dag.describe import expr_text

pytestmark = pytest.mark.unit


def test_explain_names_the_filter_pushed_to_the_source():
    text = bt.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]}).filter(bt.col("a") > 1).explain()
    scan_line = next(line for line in text.splitlines() if "scan" in line)
    assert "pushed[a > 1]" in scan_line


def test_explain_says_nothing_when_no_filter_is_pushed():
    text = bt.from_pydict({"a": [1, 2, 3]}).explain()
    assert "pushed" not in text


def test_a_filter_behind_a_breaker_is_not_reported_as_pushed():
    # A filter above an aggregate constrains the aggregated rows, not the scanned ones,
    # so it never reaches the source and must not be labelled as though it had.
    text = (
        bt.from_pydict({"a": [1, 1, 2]})
        .group_by("a")
        .agg(n=bt.col("a").count())
        .filter(bt.col("n") > 1)
        .explain()
    )
    assert "pushed" not in text


def test_the_pushed_filter_is_in_the_json_document():
    doc = json.loads(bt.from_pydict({"a": [1, 2]}).filter(bt.col("a") > 1).explain(format="json"))
    scan = next(op for op in doc["ops"] if op["kind"] == "scan")
    assert scan["pushed"] == "a > 1"


def test_a_set_membership_filter_renders_its_members():
    # The optimizer brackets the set with the bounds it derives for zone-map pruning, so
    # the label carries those too. What matters is that the members and the column survive
    # rendering — at the node-subtitle depth they were elided to `... IN (...)`.
    text = bt.from_pydict({"c": ["US", "MX"]}).filter(bt.col("c").is_in(["US", "CA"])).explain()
    assert "c IN (US, CA)" in text


def test_negation_renders_its_operand():
    # The `not` branch read `expr["expr"]`, but the IR field is `input` on every unary
    # node, so every negated predicate rendered as a bare `NOT` with nothing after it.
    assert expr_text((~(bt.col("a") > 1)).to_ir()) == "NOT (a > 1)"


def test_expr_text_renders_the_predicate_vocabulary():
    assert expr_text(bt.col("a").is_null().to_ir()) == "a IS NULL"
    assert expr_text(bt.col("a").is_not_null().to_ir()) == "a IS NOT NULL"
    assert expr_text(bt.col("a").is_in([1, 2]).to_ir()) == "a IN (1, 2)"
    assert expr_text(bt.col("s").str.starts_with("x").to_ir()) == "starts_with(s, 'x')"
    assert expr_text(bt.col("a").cast("float64").to_ir()) == "a::float64"


def test_a_long_in_list_is_summarized_rather_than_printed():
    # A pushed set is routinely hundreds of keys long; a node subtitle that prints all of
    # them stops being a subtitle.
    rendered = expr_text(bt.col("a").is_in(list(range(50))).to_ir())
    assert rendered == "a IN (0, 1, 2, 3, … 46 more)"


def test_explain_names_the_row_cap_pushed_to_the_source():
    text = bt.from_pydict({"a": [1, 2, 3, 4, 5]}).limit(2).explain()
    scan_line = next(line for line in text.splitlines() if "scan" in line)
    assert "pushed[max 2 rows]" in scan_line


def test_a_cap_blocked_by_a_filter_is_not_reported():
    # `limit(2)` after a filter means the first two *passing* rows, so the source cannot
    # stop at two — and explain must not claim it was told to.
    text = bt.from_pydict({"a": [1, 2, 3, 4, 5]}).filter(bt.col("a") > 1).limit(2).explain()
    assert "max" not in text


def test_a_filter_and_a_cap_are_reported_together():
    text = bt.from_pydict({"a": [1, 2, 3, 4, 5]}).limit(2).filter(bt.col("a") > 1).explain()
    scan_line = next(line for line in text.splitlines() if "scan" in line)
    assert "max 2 rows" in scan_line


def test_a_pushed_top_n_names_the_order_it_is_taken_in():
    # "max 2 rows" under a sort reads like an unsound prefix; naming the ordering is the
    # difference between that and a top-N, and the ordering is what makes the cap sound.
    text = bt.from_pydict({"k": [3, 1, 2]}).sort("k", descending=True).limit(2).explain()
    scan_line = next(line for line in text.splitlines() if "scan" in line)
    assert "pushed[top 2 by k desc]" in scan_line


def test_a_non_default_null_placement_is_shown_because_it_changes_which_rows_come_back():
    text = bt.from_pydict({"k": [3, 1, 2]}).sort("k", nulls_first=True).limit(2).explain()
    assert "top 2 by k nulls first" in text


def test_an_unordered_cap_is_still_reported_as_a_plain_prefix():
    text = bt.from_pydict({"k": [3, 1, 2]}).limit(2).explain()
    assert "pushed[max 2 rows]" in text
