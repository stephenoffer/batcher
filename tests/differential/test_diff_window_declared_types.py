"""Every window function's *declared* output type is the one the engine actually produces.

The window twin of `test_diff_aggregate_declared_types`, and it exists for the same reason:
`Window.available_schema` is what `Dataset.schema` answers from, what an empty result is typed
from, and what the predicates deciding whether a plan can spill or distribute read a key's
type out of. A function it cannot type does not merely report less — it makes the plan's
schema `None` from that node upward, and everything downstream that asked a type question
gets "not certain".

Six of the 33 members of `WINDOW_FUNCS` had no classification: `bit_and`, `bit_or`,
`bit_xor`, `bool_and`, `bool_or` and `count_distinct`. A seventh was outright wrong — a
windowed `sum` over a `null`-typed column declared `null` where the engine returns `int64`,
because the fold has no null-typed slot to accumulate into and materializes the column as
Int64. Nothing failed: `collect()` took its types from the engine, and only
`collect(spill=True)` on a relation that matched no rows exposed the gap, by typing its empty
result from the declaration and returning five `null` columns where `collect()` returned
five typed ones.

As with the aggregate file, this does not restate the rules. It asks the engine, across every
function crossed with every input type it accepts, and requires the declaration to agree.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.logical.window import WINDOW_FUNCS

pytestmark = pytest.mark.differential

#: One column per input type family a window function might be handed. `z` (all-null) earns
#: its place twice over: it is where the folds diverge from the value functions (a fold
#: materializes Int64, a value function keeps `null`), and it is the case the declaration got
#: wrong rather than merely unknown.
_COLUMNS: dict[str, pa.Array] = {
    "i32": pa.array([1, 2, 3, 4], pa.int32()),
    "i64": pa.array([1, 2, 3, 4], pa.int64()),
    "f32": pa.array([1.0, 2.0, 3.0, 4.0], pa.float32()),
    "f64": pa.array([1.0, 2.0, 3.0, 4.0], pa.float64()),
    "s": pa.array(["a", "b", "a", "c"]),
    "ts": pa.array([1, 2, 3, 4], pa.timestamp("us")),
    "bo": pa.array([True, False, True, True], pa.bool_()),
    "dec": pa.array([1.0, 2.0, 3.0, 4.0], pa.float64()).cast(pa.decimal128(10, 2)),
    "z": pa.array([None] * 4, pa.null()),
}

#: Window functions the control plane is allowed not to type, each with the reason. **Empty,
#: and meant to stay so** — an entry costs the whole plan its schema from that node upward.
_UNDECLARED: dict[str, str] = {}


@pytest.fixture(scope="module")
def rows() -> pa.Table:
    return pa.table({**_COLUMNS, "o": pa.array([1, 2, 3, 4], pa.int64())})


def _pairs(rows: pa.Table, func: str) -> list[tuple[str, pa.DataType, pa.DataType | None]]:
    """Every `(input, engine type, declared type)` this window function accepts.

    `None` stands for the no-input form, which is how the ranking functions are spelled;
    trying it alongside the columns rather than listing which functions take no argument is
    what keeps a newly added ranking function visible here.
    """
    out = []
    for name in [*_COLUMNS, None]:
        for build in _BUILDERS:
            try:
                dataset = build(rows, func, name)
                actual = dataset.collect().schema.field("w").type
            except Exception:
                continue  # this spelling is refused; try the next
            schema = dataset._plan.available_schema()
            out.append((str(name), actual, schema.arrow.field("w").type if schema else None))
            break
    return out


def _by_window(rows: pa.Table, func: str, name: str | None):
    """The ordinary spelling: `window(functions={...})` with a bare name or a `(func, col)`."""
    spec = func if name is None else (func, bt.col(name))
    return bt.from_arrow(rows).window(order_by=["o"], functions={"w": spec})


def _by_expression(rows: pa.Table, func: str, name: str | None):
    """The expression spelling, for a function `window()` cannot express.

    The EWM series need a smoothing factor, which the `(func, column)` tuple has no slot for,
    so `window()` refuses them outright — they are reachable only as
    `col(x).ewm_mean(alpha=).over(order_by=...)`. Without this arm all three would be silently
    skipped, which the vacuity assertion in the test turns into a failure rather than a quiet
    gap.
    """
    if name is None:
        raise TypeError("the expression spelling always takes a column")
    series = getattr(bt.col(name), func)(alpha=0.5).over(order_by=["o"])
    return bt.from_arrow(rows).with_columns(w=series)


#: Tried in order; the first that builds and runs is the one the pair is measured through.
_BUILDERS = (_by_window, _by_expression)


@pytest.mark.parametrize("func", sorted(WINDOW_FUNCS))
def test_the_declared_type_is_the_engine_type(rows, func):
    accepted = _pairs(rows, func)
    assert accepted, f"{func} accepted no input at all — the fixture cannot see it"
    for name, actual, declared in accepted:
        if declared is None:
            assert func in _UNDECLARED, (
                f"{func} over {name} declares no output type (the engine returns {actual}); "
                "classify it in `plan.logical.window._window_func_type` — an unknown type "
                "makes the whole plan's schema `None` from this node upward"
            )
            continue
        assert declared == actual, (
            f"{func} over {name}: declared {declared}, engine returns {actual}"
        )


def test_the_fixture_reaches_both_kinds_of_function(rows):
    """Guard against a vacuous sweep.

    Every assertion above is over "the pairs the engine accepts", so a fixture that stopped
    producing usable columns would leave each parametrization with nothing to compare and the
    file would pass while checking nothing. `row_number` takes no input and `sum` takes a
    numeric one, so between them they exercise both halves of `_pairs`.
    """
    assert [n for n, _a, _d in _pairs(rows, "row_number")] == ["None"]
    numeric = {n for n, _a, _d in _pairs(rows, "sum")}
    assert {"i32", "i64", "f64"} <= numeric
    assert "s" not in numeric
