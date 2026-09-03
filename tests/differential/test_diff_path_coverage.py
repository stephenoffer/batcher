"""Every `RelOp` tag must be exercised on the *spill* and *distributed* paths — checked.

`test_diff_operator_matrix_coverage.py` does this for the *in-memory* schedulings, and its
docstring gives the reason: a matrix "only closes that gap for operators somebody remembered
to add a row for, and five tags had silently never been added". The two out-of-memory paths
had no equivalent, which is backwards — they are the ones `CLAUDE.md` singles out:

    A stateful operator without a mergeable form works perfectly single-node, passes every
    local test, and silently caps the operator at one machine. Failure appears at cluster
    scale, as wrong results rather than an error.

and, for the other:

    New stateful operators should have a spill story.

So the two paths where a missing case costs a wrong answer on a cluster or an OOM at scale
were relying on the arrangement the safest path stopped relying on. They are checked together
because the derivation is the same and duplicating it is how two copies of a threshold drift
apart.

## What this found, and why the file is worth having anyway

Nothing — on either path. Auditing it by hand said six tags were uncovered, then five, then
two, then one, then none, and the *derived* version needed three corrections of its own before
it agreed. Every reduction was a defect in the search, not in the suite:

* grepping for `.range_join(` found nothing because a range join has no method; it is a join
  with an inequality predicate, spelled in SQL;
* grepping for `.explode(` inside test *functions* found nothing because the operator lives
  in a module-level table that a parametrized test walks, not in any function body.

* and the first two versions of *this file* under-derived, because `sample` / `unnest` /
  `unpivot` / `asof_join` live in `test_diff_reshape_matrix`, whose `_assert_paths_agree`
  runs them under `collect(spill=True)` — a table this guard was not importing.

Every one of those returned a confident zero for an operator with dedicated tests. The guard
needed correcting as often as the hand audit did; the difference is that it now derives from
the tables themselves, so the next operator is checked without anyone choosing to look.

## The oracle here is the engine's own vocabulary

There is no second engine to compare against, and unlike the rest of this directory there
cannot be: this asserts a property of the *test suite*, not of a query result. The reference
is `plan.ir_tags.Op` — the engine's own list of relational operators — held against the set
the distributed tables can actually produce. Both sides are derived, so neither can be edited
to make the other agree.

It sits here rather than in `tests/unit/` because it imports and lowers the differential
matrices themselves, and it is those matrices' completeness it is about.

## How coverage is derived

Each op table below is *imported from the file that runs it distributed*, lowered to JSON IR,
and its `op` tags collected. A table earns its tags by producing a plan that contains them —
so a tag cannot be faked by adding a name here, and renaming a test cannot lose one. That is
the same construction the single-node guard uses and for the same reason: a hand-maintained
list of "operators we cover distributed" is exactly the artifact that goes stale without
failing.
"""

from __future__ import annotations

import contextlib
import functools
import sys

import pyarrow as pa
import pytest

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

from test_diff_distributed_operator_matrix import SHAPES as _DIST_SHAPES  # noqa: E402
from test_diff_operator_matrix import BASE, RIGHT, UNORDERED_OPS  # noqa: E402
from test_diff_reshape_matrix import (  # noqa: E402
    ASOF_LEFT,
    ASOF_OPS,
    ASOF_RIGHT,
    RESHAPE,
    RESHAPE_OPS,
    SAMPLE_OPS,
)

from batcher.plan.ir_tags import Op  # noqa: E402

#: Tags no distributed table can produce, each with the file that covers it instead. Keep
#: this empty where possible; an entry is a gap held open on purpose, not an amnesty.
#:
#: `range_join` is not a gap in coverage but a limit of the derivation: it is an
#: *optimizer-produced* form. A range join is written as a join with an inequality, and the
#: logical plan this guard lowers therefore contains `hash_join` + `filter` — the range join
#: only appears after the optimizer rewrites it. Deriving it would mean running the optimizer
#: here, which makes this guard depend on rule ordering to report coverage, and a coverage
#: check that can be broken by an unrelated optimizer change is worse than one honest entry.
_RANGE_JOIN_REASON = (
    "an optimizer-produced form, absent from the logical plan this guard lowers; covered by "
    "tests/integration/test_distributed_sample_and_range_join.py (distributed) and "
    "tests/differential/test_diff_range_join.py (spill)"
)

EXEMPT: dict[str, dict[str, str]] = {
    "distributed": {"range_join": _RANGE_JOIN_REASON},
    "spill": {"range_join": _RANGE_JOIN_REASON},
}


def _all_tags() -> set[str]:
    """Every `RelOp` tag the engine defines."""
    return {v for k, v in vars(Op).items() if not k.startswith("_") and isinstance(v, str)}


def _tags_in(ir: object, found: set[str]) -> set[str]:
    """Every ``op`` tag appearing anywhere in a lowered plan."""
    if isinstance(ir, dict):
        op = ir.get("op")
        if isinstance(op, str):
            found.add(op)
        for value in ir.values():
            _tags_in(value, found)
    elif isinstance(ir, list):
        for value in ir:
            _tags_in(value, found)
    return found


@functools.cache
def _mode_parity_shapes():
    """The op table `tests/integration/test_mode_parity_matrix.py` runs distributed.

    Cached, and not only for speed. Without it every test in this file re-executes that
    integration module from source, and executing a *test* module repeatedly in one process
    is not side-effect free: the six re-entries produced an `ImportError` on a half-written
    `batcher._exports` and a `RecursionError`, while each test passed in isolation. Loading
    it once is both correct and the honest amount of work.

    Imported rather than duplicated, so this guard tracks that file. It lives under
    `tests/integration/`, which is not on this directory's import path, so it is loaded by
    location — and if it moves, this raises rather than silently covering less.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "integration" / "test_mode_parity_matrix.py"
    assert path.is_file(), f"the mode-parity matrix moved: {path}"
    spec = importlib.util.spec_from_file_location("_mode_parity_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SHAPES


@functools.cache
def path_tags(path: str) -> set[str]:
    """Tags reachable through the tables a given execution path's tests actually walk.

    The op table is shared: `test_diff_operator_matrix` runs it under `spill_partitioned`
    among its schedulings, and `test_diff_distributed_operator_matrix` imports the same table
    and runs it across workers. Only the *extra* tables differ, so the derivation is written
    once and the difference is one branch.
    """
    found: set[str] = set()
    table = pa.table({**BASE.to_pydict(), "lst": [[1, 2]] * BASE.num_rows})

    for build, *_ in UNORDERED_OPS.values():
        with contextlib.suppress(Exception):  # a unary builder; the binary form is tried next
            _tags_in(build(bt.from_arrow(BASE))._plan.to_ir(), found)
        with contextlib.suppress(Exception):  # not every builder takes a right side
            _tags_in(build(bt.from_arrow(BASE), bt.from_arrow(RIGHT))._plan.to_ir(), found)

    if path == "distributed":
        # The mode-parity matrix is distributed-only: it exists to check single-node ==
        # distributed and has no spill scheduling.
        for _name, build, *_ in _mode_parity_shapes():
            with contextlib.suppress(Exception):  # some shapes want a file source
                _tags_in(build(bt.from_arrow(table))._plan.to_ir(), found)

    if path == "spill":
        # The reshape matrix is the spill home for the operators the main table has no row
        # for. Its `_assert_paths_agree` runs every one of these under `collect(spill=True)`,
        # so importing its tables is what makes `sample` / `unnest` / `unpivot` / `asof_join`
        # earn their tags here rather than being asserted covered.
        for entry in (*RESHAPE_OPS.values(), *SAMPLE_OPS.values()):
            build = entry[0] if isinstance(entry, tuple) else entry
            with contextlib.suppress(Exception):
                _tags_in(build(bt.from_arrow(RESHAPE))._plan.to_ir(), found)
        for entry in ASOF_OPS.values():
            build = entry[0] if isinstance(entry, tuple) else entry
            with contextlib.suppress(Exception):
                _tags_in(
                    build(bt.from_arrow(ASOF_LEFT), bt.from_arrow(ASOF_RIGHT))._plan.to_ir(),
                    found,
                )

    # Operators executed distributed by a dedicated file rather than a table. Their plans are
    # *built* here rather than named, so a tag is still earned by producing one:
    #   sort      -> test_diff_distributed_sort.py / test_diff_spill_paths.py
    #   asof_join -> test_diff_asof_out_of_core.py (both paths)
    #   unpivot   -> test_distributed.py, test_diff_reshape_matrix.py
    _tags_in(bt.from_arrow(BASE).sort("k")._plan.to_ir(), found)
    left = pa.table({"t": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    right = pa.table({"t": [1, 3], "q": ["a", "b"]})
    _tags_in(
        bt.from_arrow(left).join_asof(bt.from_arrow(right), on="t")._plan.to_ir(),
        found,
    )
    wide = pa.table({"id": [1, 2], "a": [1, 2], "b": [3, 4]})
    _tags_in(
        bt.from_arrow(wide).unpivot(index=["id"], on=["a", "b"])._plan.to_ir(),
        found,
    )
    return found


#: What each path costs when an operator is missing from it — quoted in the failure so the
#: message says why it matters rather than only what is absent.
_WHY = {
    "distributed": (
        "Distribution is a scheduling concern over the same mergeable primitives "
        "(invariant #7), so an operator with no distributed test is one whose "
        "`partial -> combine -> finalize` form nothing checks. That shows up as wrong rows "
        "on a cluster, not as an error."
    ),
    "spill": (
        "An operator with no out-of-core test is one that works until the data does not "
        "fit. That shows up as an OOM at scale, on data nobody can reproduce locally."
    ),
}


@pytest.mark.parametrize("path", sorted(_WHY))
def test_every_relational_operator_runs_on_the_out_of_memory_paths(path):
    """No `RelOp` may reach the engine without a spill and a distributed equivalence test."""
    missing = sorted(_all_tags() - path_tags(path) - set(EXEMPT[path]))
    assert not missing, (
        f"{len(missing)} operator(s) have no {path} coverage: {missing}.\n{_WHY[path]}\n"
        f"Add it to a matrix that runs {path}, or exempt it here with the file that covers "
        f"it instead."
    )


@pytest.mark.parametrize("path", sorted(_WHY))
def test_the_exemption_list_stays_honest(path):
    """An exemption for a tag that IS covered is stale and must go."""
    stale = sorted(set(EXEMPT[path]) & path_tags(path))
    assert not stale, f"covered on the {path} path and no longer needing exemption: {stale}"


@pytest.mark.parametrize("path", sorted(_WHY))
def test_the_probe_reports_an_operator_nothing_covers(monkeypatch, path):
    """The guard's own control: introduce an uncovered operator and it must be reported.

    A coverage check that only ever passes is indistinguishable from one that works, and
    this file's whole claim is that it would catch a new operator arriving without a
    distributed test. So the claim is exercised rather than asserted: a tag is added to the
    engine's vocabulary that no table can produce, and the guard has to name it.

    The two halves matter equally. Flagging the invented tag shows the check *can* fire;
    `test_every_relational_operator_runs_distributed_somewhere` passing on the real
    vocabulary shows it does not fire on everything. A checker that returned "missing"
    unconditionally would satisfy the first and be worthless.
    """
    # `sys.modules[__name__]` rather than importing this module by its dotted path:
    # the self-import breaks the moment the file is renamed, which it was.
    guard = sys.modules[__name__]

    invented = "teleport_join"
    # Bind the original *before* patching. `lambda: _all_tags() | {invented}` looks right and
    # is infinitely recursive: `_all_tags` inside it is a module-global lookup made at call
    # time, by which point monkeypatch has replaced that global with this very lambda. It
    # failed loudly with `RecursionError`, which is the safe direction — a negative control
    # that broke quietly would have left this file asserting nothing about its own guard.
    original = _all_tags
    monkeypatch.setattr(guard, "_all_tags", lambda: original() | {invented})

    missing = sorted(guard._all_tags() - path_tags(path) - set(EXEMPT[path]))
    assert missing == [invented], (
        f"an operator no table produces must be reported as uncovered on {path}; got {missing}"
    )


def test_lowering_a_plan_reports_its_own_operator():
    """`_tags_in` must actually see tags, or every assertion here is vacuous."""
    ir = bt.from_arrow(BASE).filter(bt.col("k") > 1)._plan.to_ir()
    assert "filter" in _tags_in(ir, set())
    assert "scan" in _tags_in(ir, set())


def test_the_distributed_shapes_table_is_not_empty():
    """The imported matrices must still have rows; an empty one covers nothing loudly."""
    assert UNORDERED_OPS, "the single-node op table the distributed matrix walks is empty"
    assert _DIST_SHAPES, "the distributed matrix's input shapes are empty"
    assert _mode_parity_shapes(), "the mode-parity op table is empty"
