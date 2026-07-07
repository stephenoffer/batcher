"""Property: adaptive-tuning knobs change *how* a query runs, never *what* it returns.

Batcher's performance levers — morsel size, spill on/off, shuffle partition count,
adaptive morsel sizing — are documented as strictly result-invariant (see the field
docstrings in ``config/config.py``): a morsel only batches rows, spilling only moves
state to disk, partition count only reshapes the shuffle, and adaptive sizing only
shrinks a morsel under memory pressure. The whole adaptive-tuning story rests on this
invariant, so it deserves a property test rather than trust.

Hypothesis generates a random grouped table and runs the same aggregate/distinct
query at two settings of each knob, asserting identical results (order-independent,
float-tolerant). A setting that changes the answer is a real correctness bug in that
lever — a divergence to surface, not to weaken away.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col, count
from batcher.config import Config, ExecutionConfig, config_context

pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = [pytest.mark.property, pytest.mark.integration]

_vals = st.integers(min_value=-40, max_value=40)
_nullable = st.one_of(st.none(), _vals)
_SCHEMA = pa.schema([("g", pa.int64()), ("h", pa.int64()), ("v", pa.int64())])


def _coerce(v: object) -> object:
    if isinstance(v, bool):
        return v
    try:
        return round(float(v), 9)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return v


def _rowset(table: pa.Table) -> list[tuple]:
    cols = table.column_names
    rows = [tuple(_coerce(r[c]) for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((x is None, str(x)) for x in t))


@st.composite
def _grouped(draw: st.DrawFn) -> pa.Table:
    """A random two-key grouped table with a nullable value column."""
    n = draw(st.integers(min_value=0, max_value=90))
    g = draw(st.lists(st.integers(min_value=0, max_value=5), min_size=n, max_size=n))
    h = draw(st.lists(st.integers(min_value=0, max_value=3), min_size=n, max_size=n))
    v = draw(st.lists(_nullable, min_size=n, max_size=n))
    return pa.table({"g": g, "h": h, "v": v}, schema=_SCHEMA)


def _load(table: pa.Table) -> bt.Dataset:
    return bt.from_arrow(table.to_batches() or table)


def _agg_result(table: pa.Table, **collect_kw: object) -> list[tuple]:
    """A multi-key, multi-aggregate group-by — the mergeable-state stress query."""
    ds = (
        _load(table)
        .group_by("g", "h")
        .agg(
            s=col("v").sum(),
            n=count(),
            c=col("v").count(),
            lo=col("v").min(),
            hi=col("v").max(),
            a=col("v").mean(),
            sd=col("v").std(),
            md=col("v").median(),
            nd=col("v").n_unique(),
        )
    )
    return _rowset(ds.collect(**collect_kw))  # type: ignore[arg-type]


def _distinct_result(table: pa.Table, **collect_kw: object) -> list[tuple]:
    return _rowset(_load(table).select("g", "v").distinct().collect(**collect_kw))  # type: ignore[arg-type]


def _with_execution(**overrides: object) -> Config:
    return Config().replace(execution=ExecutionConfig(**overrides))  # type: ignore[arg-type]


_PROP = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@_PROP
@given(_grouped())
def test_morsel_rows_invariance(table: pa.Table) -> None:
    """A tiny morsel (1 row) and a large one produce the identical aggregate."""
    with config_context(_with_execution(morsel_rows=1)):
        tiny = _agg_result(table)
    with config_context(_with_execution(morsel_rows=65_536)):
        big = _agg_result(table)
    assert tiny == big, f"morsel_rows changed the result:\n1-row : {tiny}\n64k   : {big}"


@_PROP
@given(_grouped())
def test_spill_invariance(table: pa.Table) -> None:
    """Forcing out-of-core spill yields the same result as the in-memory path."""
    spilled = _agg_result(table, spill=True)
    in_memory = _agg_result(table)
    assert spilled == in_memory, f"spill changed the result:\nspill: {spilled}\nmem  : {in_memory}"


@_PROP
@given(_grouped())
def test_num_partitions_invariance(table: pa.Table) -> None:
    """Different shuffle-partition counts produce the identical result."""
    one = _agg_result(table, num_partitions=1)
    many = _agg_result(table, num_partitions=7)
    assert one == many, f"num_partitions changed the result:\n1 : {one}\n7 : {many}"


@_PROP
@given(_grouped())
def test_adaptive_morsel_sizing_invariance(table: pa.Table) -> None:
    """`adaptive_morsel_sizing` on vs off is result-invariant."""
    with config_context(_with_execution(adaptive_morsel_sizing=True)):
        on = _agg_result(table)
    with config_context(_with_execution(adaptive_morsel_sizing=False)):
        off = _agg_result(table)
    assert on == off, f"adaptive_morsel_sizing changed the result:\non : {on}\noff: {off}"


@_PROP
@given(_grouped())
def test_distinct_tuning_invariance(table: pa.Table) -> None:
    """`distinct` is invariant to morsel size, spill, and partition count together."""
    with config_context(_with_execution(morsel_rows=1)):
        tiny = _distinct_result(table, spill=True, num_partitions=5)
    baseline = _distinct_result(table)
    assert tiny == baseline, f"tuning changed distinct:\ntuned: {tiny}\nbase : {baseline}"
