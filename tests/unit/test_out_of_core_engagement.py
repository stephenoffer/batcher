"""Which out-of-core path each operator actually *engages* — not merely which answer it gives.

Every differential suite in this repo asks whether a mode returns the right rows. None of them
can see the failure this file exists for: a mode that returns the right rows by quietly
**declining** its bounded-memory path and materializing the whole relation instead.

That is not hypothetical, and it is invisible from the outside. `sort(col("a") + col("b"))`
answered `collect()`, `collect(spill=True)` and `iter_batches()` with identical, correct rows,
and the last two ran the in-memory kernel — under the very memory envelope the spill exists
for — because the shuffle key was an expression and only the *distributed* dispatcher knew how
to materialize it into a column. The rewrite that fixed it (`plan.logical.hoist_sort_key`) is
now shared by all three paths; nothing in a correctness suite would have caught its absence,
and nothing would catch it coming back.

So this asserts the *mechanism*, by spying on the entry point each bounded-memory path must go
through. The declaration below is the contract: for a shape, either it names the generator that
must run, or it says `None` with the reason the operator legitimately materializes. A new
operator that materializes silently fails here until someone writes down which of the two it is.

**What this spy can and cannot see, because it cost someone twenty minutes.** Every entry
point below is in the *Python control plane*. An operator whose out-of-core path lives in the
Rust data plane spills perfectly well and still shows up here as though it materialized. That
is not a gap in the engine and must not be recorded as one: `distinct(subset=[...])` declines
the Python spill path on purpose — its surviving row carries columns the key does not
determine, so the group-by-every-column equivalence that a whole-row `DISTINCT` rides is
simply false for it — and `bc_interp::distinct_on_spill` reduces it out of core under the same
envelope. A probe watching only these names reports it as unbounded, twice over, since
`iter_batches` reaches it through `stream_distinct_on` rather than `stream_distinct`.

So read `None` below as "takes no *Python* bounded-memory path", not as "holds the relation in
memory". Only `window_global_lag` means the stronger thing, and its comment says so. Before
adding a `None` entry, check for a Rust path: `crates/bc-interp/src/*_spill.rs`, and
`ExecMetrics.spilled` from `core.execute_local_metered`, which is the one instrument that sees
across the FFI boundary.

The third mode — `collect(distributed=True)` — asks the same question of the dispatcher and
lives in `tests/integration/test_distributed_engagement.py`, because answering it needs a real
cluster. This file is deliberately Ray-free so it runs at unit speed in every CI job.

Deliberately a `unit` test and not a benchmark: peak RSS is the quantity of interest but it is
far too noisy on a shared machine to gate on, whereas "did this function get called" is exact.
"""

from __future__ import annotations

import pytest

from _engagement_shapes import _BUILDERS, EXPECTED_DISTRIBUTED, _build

pytestmark = pytest.mark.unit

#: Every bounded-memory entry point, as `(attribute, the modules that bind the name)`.
#:
#: **Both** the defining module and the re-exporting package are patched, and that is not
#: belt-and-braces. A monkeypatch follows the *name*, not the function: patching only the
#: package leaves `execute_spilling_sort`'s own module-global call to `stream_spilling_sort`
#: bound to the original, and patching only the module leaves the package attribute — which
#: is what the streaming dispatcher's function-local `from ... import` resolves — bound to it.
#: Each spelling alone makes half these shapes look as though they never spilled. (`.claude/
#: rules/concurrent-agents.md` records the same trap from the other side: a patch that
#: silently stops applying while the test keeps passing.)
_ENTRY_POINTS = {
    "sort": (
        "stream_spilling_sort",
        ("batcher.dist.spill_breakers.sort", "batcher.dist.spill_breakers"),
    ),
    "window": (
        "stream_spilling_window",
        ("batcher.dist.spill_breakers.window", "batcher.dist.spill_breakers"),
    ),
    "join": (
        "stream_spilling_join",
        ("batcher.dist.spill_breakers.join", "batcher.dist.spill_breakers"),
    ),
    "global_window": (
        "stream_spilling_global_window",
        ("batcher.dist.global_window.stream", "batcher.dist.global_window"),
    ),
    "aggregate": (
        "execute_spilling_aggregate",
        ("batcher.dist.spill.aggregate", "batcher.dist.spill"),
    ),
    # `iter_batches()` folds an aggregate into one running state per group rather than
    # grace-partitioning it to disk. A different mechanism from the spilling collect, and the
    # right one for a stream: the state is Theta(groups) and no input batch is ever retained.
    # One module each: the streaming dispatcher imports these *inside* the function, so the
    # package attribute is what it resolves at call time and there is no second binding.
    "running_fold": ("stream_aggregate", ("batcher.core.streaming",)),
    "running_dedup": ("stream_distinct", ("batcher.core.streaming",)),
}

#: shape -> the mechanism `collect(spill=True)` must engage, or `None` with the reason it may
#: legitimately materialize.
#:
#: The three `computed_key` entries are the ones this file was written for: each answered
#: correctly while materializing, for as long as the hoist was private to the distributed
#: dispatcher.
_EXPECTED: dict[str, str | None] = {
    "sort_plain": "sort",
    "sort_string_key": "sort",
    "sort_computed_key": "sort",
    "window_partitioned": "window",
    "window_computed_key": "window",
    "window_global_ordered": "global_window",
    "window_global_fold": "global_window",
    # A global window over `lag` reads rows its own ordered bucket does not hold, so no offset
    # recovers it — `global_window.offsets` declines it by design and the materializing kernel
    # runs. Correct, and the one shape here that is *meant* to hold the relation in memory.
    "window_global_lag": None,
    "aggregate": "aggregate",
    "distinct": "aggregate",  # a whole-row dedup IS the group-by-every-column aggregate
    "join": "join",
    # An ASOF join with `by` keys hashes on `by` and rides the equi-join's grace pipeline.
    "asof_join_by": "join",
    # A keyless ASOF has no group to hash on. The cluster reaches it by range-partitioning on
    # `on` and lending each bucket its boundary row, which is a different decomposition and
    # not this one — so it declines and the in-memory kernel runs.
    "asof_join_keyless": None,
}

#: Where `iter_batches()` differs. A stream folds an aggregate into one running state per group
#: instead of grace-partitioning it to disk — bounded by the group count rather than by a
#: bucket, and strictly better for a stream, since no input batch is ever retained. Shapes
#: absent from this table use their `_EXPECTED` mechanism unchanged.
_EXPECTED_STREAMING: dict[str, str | None] = {
    "aggregate": "running_fold",
    "distinct": "running_dedup",
}


@pytest.fixture
def spy(monkeypatch):
    """Record which bounded-memory entry points fire, leaving their behaviour unchanged."""
    import importlib

    fired: list[str] = []

    def install(label, attr, module_paths):
        # One wrapper per name, installed under every module that binds it, so whichever
        # spelling a caller resolves lands on the same recorder exactly once.
        original = getattr(importlib.import_module(module_paths[0]), attr)

        def wrapper(*args, **kwargs):
            fired.append(label)
            return original(*args, **kwargs)

        for path in module_paths:
            module = importlib.import_module(path)
            # `hasattr`, not an identity check against `original`: `dist.global_window` binds
            # its names through a PEP 562 `__getattr__`, so reading the package attribute
            # after the defining module is patched returns the *wrapper*. What this catches is
            # the failure that matters — a moved or renamed entry point leaving the spy
            # watching a name nothing calls, which would report every shape as materializing.
            assert hasattr(module, attr), f"{path} no longer binds {attr!r}"
            monkeypatch.setattr(module, attr, wrapper)

    for label, (attr, module_paths) in _ENTRY_POINTS.items():
        install(label, attr, module_paths)
    return fired


def test_the_spy_itself_records_a_call(spy):
    """Guard against a vacuous suite.

    Every assertion below is of the form "mechanism X fired". If the spy silently stopped
    patching — an entry point renamed, a caller switched to a spelling nothing wraps — those
    would all fail loudly, but the `None` (declared-to-materialize) cases would start passing
    for the wrong reason and quietly stop testing anything. A plain sort is the shape with the
    longest-standing spill path, so its absence means the harness, not the engine.
    """
    _build("sort_plain").collect(spill=True, num_partitions=4)
    assert "sort" in spy


@pytest.mark.parametrize("shape", sorted(_EXPECTED))
def test_collect_spilling_engages_the_declared_mechanism(spy, shape):
    ds = _build(shape)
    ds.collect(spill=True, num_partitions=4)
    expected = _EXPECTED[shape]
    if expected is None:
        assert not spy, f"{shape} is declared to materialize but engaged {sorted(set(spy))}"
    else:
        assert expected in spy, (
            f"{shape} materialized instead of spilling through {expected!r} "
            f"(engaged: {sorted(set(spy)) or 'nothing'})"
        )


@pytest.mark.parametrize("shape", sorted(_EXPECTED))
def test_iter_batches_engages_the_declared_mechanism(spy, shape):
    """`iter_batches()` is the entry point whose whole promise is bounded memory, so a silent
    materialization there is the worst version of this defect."""
    ds = _build(shape)
    list(ds.iter_batches())
    expected = _EXPECTED_STREAMING.get(shape, _EXPECTED[shape])
    if expected is None:
        assert not spy, f"{shape} is declared to materialize but engaged {sorted(set(spy))}"
    else:
        assert expected in spy, (
            f"{shape} materialized instead of streaming through {expected!r} "
            f"(engaged: {sorted(set(spy)) or 'nothing'})"
        )


def test_every_declared_shape_is_built_and_every_built_shape_is_declared():
    """The builder and the declaration must name the same shapes.

    Derived from `_build` itself rather than from a third hand-written list: the list was the
    third place a shape name had to be added, and it is the one nothing else would catch.
    """
    built = set(_BUILDERS)
    assert built == set(_EXPECTED), "the spill table and the builders disagree"
    assert set(_EXPECTED_STREAMING) <= built, "a streaming override names no builder"
    # The distributed table too, though its tests live in `tests/integration/` and need a
    # cluster. The *guard* must not: split across two suites, this is the seam that would
    # rot, and an operator added for the spill question and never asked the distributed one
    # would simply go unchecked with nothing failing — in the CI job that has no Ray, which
    # is most of them.
    assert built == set(EXPECTED_DISTRIBUTED), "the distributed table and the builders disagree"
    for name in sorted(built):
        assert _build(name) is not None
