"""No operator's result may depend on the morsel size it was executed at.

`execution.morsel_rows` is the width of the unit the Rust engine schedules, shipped across
the FFI boundary as `EngineConfig.morsel_rows`. Changing it changes how many parallel units
a query is folded in, and nothing else: the same `partial -> combine -> finalize` primitives
run, just more or fewer times, merging in a different order. So the result must be identical,
and an operator whose state leaks across a morsel boundary -- or whose `combine` is not
actually associative -- returns the right answer at the default width and a wrong one at
another.

The knob is demonstrably live rather than assumed. The same 20,000-row grouped aggregate,
metered through `execute_local_metered`:

    morsel_rows=64     aggregate rows_in=20000  threads=61
    morsel_rows=16384  aggregate rows_in=20000  threads=2

Sixty-one parallel folds against two, for one query. That is what these cases vary.

**Not to be confused with `iter_batches(batch_size=...)`, which is a different thing and a
much weaker test.** That parameter re-chunks the *output* after the operator has finished:
a `distinct` yielding four rows returns them as 4x1 or 3+1 or 1x4, and asserting those agree
tests the rebatcher, not the operator. A draft of this file varied that instead and passed
216 cases with a boundary bug deliberately injected into the streaming distinct driver.

Compared with `assert_tables_equal`, whose docstring explains why a hand-rolled
`to_pydict() ==` cannot be used here: `nan != nan`, and `INPUTS["base"]` carries a NaN.
"""

from __future__ import annotations

import pytest
from test_diff_operator_matrix import INPUTS, UNORDERED_OPS

import batcher as bt
from _harness import assert_tables_equal
from batcher.config import ExecutionConfig, active_config, set_config

pytestmark = pytest.mark.differential

#: Widths to execute at. `1` puts every row in its own morsel, which is the hardest case for
#: any operator carrying state between them; `7` divides none of these shapes evenly, so the
#: final morsel is always short; `16384` is the default and is the oracle.
_MORSELS = (1, 7)

#: `multibatch` is the shape that matters most here -- each key value repeats 2,400 times, so
#: a group genuinely spans many morsels at these widths and has to be combined across them.
_SHAPES = ("base", "multibatch")


@pytest.fixture
def at_morsel_size():
    """Run a callable at a given `morsel_rows`, restoring the config afterwards.

    Process-global state, so it is restored in a fixture teardown rather than inline: an
    exception mid-test would otherwise leave every later test in the session running at a
    width it did not ask for, and the failure would surface far from its cause.
    """
    original = active_config()

    def run(width: int, build, table):
        set_config(original.replace(execution=ExecutionConfig(morsel_rows=width)))
        return build(bt.from_arrow(table)).collect()

    try:
        yield run
    finally:
        set_config(original)


@pytest.mark.parametrize("morsel_rows", _MORSELS)
@pytest.mark.parametrize("op", sorted(UNORDERED_OPS))
@pytest.mark.parametrize("shape", _SHAPES)
def test_an_operator_gives_the_same_answer_at_every_morsel_size(
    op, shape, morsel_rows, at_morsel_size
):
    build, _sql = UNORDERED_OPS[op]
    table = INPUTS[shape]
    oracle = build(bt.from_arrow(table)).collect()  # at the ambient default
    got = at_morsel_size(morsel_rows, build, table)
    assert_tables_equal(got, oracle)


def test_the_morsel_width_actually_reaches_the_engine(at_morsel_size):
    """Guard against a vacuous suite.

    Every case above asserts two runs agree. If `morsel_rows` were clamped, ignored, or
    dropped before the FFI boundary, they would agree because nothing differed, and the file
    would pass while testing nothing. So pin that the width changes how the engine schedules:
    a narrow morsel must fold the same query in materially more parallel units than the
    default does.
    """
    from batcher import core, kyber
    from batcher.io.source import read_source

    ds = bt.from_arrow(INPUTS["multibatch"]).group_by("g").agg(s=bt.col("v").sum())
    original = active_config()

    def threads_at(width: int) -> int:
        set_config(original.replace(execution=ExecutionConfig(morsel_rows=width)))
        physical, _logical, _d = kyber.optimize_full(ds._plan, None, ds._sources, None)
        batches = [
            read_source(src, physical.source_projections.get(i), None, None, None)
            for i, src in enumerate(ds._sources)
        ]
        _out, ops, _usage = core.execute_local_metered(physical, batches, feedback=None)
        return max((o.get("threads") or 0) for o in ops) if ops else 0

    try:
        narrow, default = threads_at(64), threads_at(16384)
    finally:
        set_config(original)

    assert narrow > default, (
        f"morsel_rows made no difference to scheduling ({narrow} vs {default} units), so the "
        "invariance asserted above is not being exercised"
    )
