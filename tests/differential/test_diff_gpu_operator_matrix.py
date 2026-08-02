"""Every operator x every edge-case input, through the **device translator**.

`test_diff_operator_matrix.py` crosses every operator with every scheduling of the CPU engine
on a 15-row input loaded with nulls, `-0.0`/NaN, duplicates and an empty case. This file runs
that same table through `core/gpu_plan/` — the cuDF translator — and checks it against the
engine and against DuckDB.

It exists because the device tier is the one tier that cannot share the Rust `Expr`: cuDF has
no maintained Rust binding, so the translation is a second statement of the engine's semantics
and can drift from it (`.claude/rules/device-tier.md`). Until this file, the tier's only
coverage was `tests/unit/test_gpu_plan.py`, whose inputs are ordinary — and the divergences
that matter are not on ordinary data. Writing it found four, all on exactly the columns this
table exists to carry:

* a group on a float key containing `-0.0` and `0.0` returned the group labelled `0.0` where
  the engine labels it `-0.0` — same group, different name, which a sharded fan-out turns back
  into two groups;
* `SUM`/`MAX` over an **empty** partition came back `double` where the engine returns `int64`;
* `PARTITION BY g ORDER BY k, g` raised a bare `ValueError` out of pandas' multi-key sort, so
  an ordinary window silently ran on the CPU as an unclassified failure;
* so did any window ordered by a float column holding both zeros.

The operator table is *imported* rather than restated, so an operator added there is covered
here too and the two cannot drift apart.

**Backends.** pandas always — it is the head-runnable stand-in CI can run. cuDF as well when
it is importable *and* a device is visible, which is the only configuration that can catch a
device-only divergence; the two the package has already shipped were both of that kind. On a
CPU-only machine the cuDF parameter is skipped, and `just lint-skips` counts it.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.differential

pytest.importorskip("batcher._native", reason="native engine not built")
pytest.importorskip("pandas", reason="the head-runnable device backend needs pandas")

from test_diff_operator_matrix import INPUTS, UNORDERED_OPS  # noqa: E402

import batcher as bt  # noqa: E402
from _harness import assert_same  # noqa: E402
from batcher.api.terminal.gpu_backend.verify import compare_results  # noqa: E402
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops  # noqa: E402
from batcher.core.gpu_plan.execute import run_chain  # noqa: E402

#: Operators `eligibility` matches as a plan *shape* rather than as a chain step, so they are
#: reached through `gpu_join_spec`/`gpu_union_spec` and not through `gpu_plan_ops`. Listed so a
#: newly-declining operator shows up as a failure here instead of quietly joining them.
STRUCTURAL = frozenset(
    {"scan", "union", "join_inner", "join_left", "join_outer", "join_semi", "join_anti"}
)


def _backends() -> list[str]:
    """The dataframe libraries to translate against on this machine."""
    names = ["pandas"]
    try:
        import cudf  # noqa: F401

        from batcher.api.terminal.gpu_backend.fanout import _cluster_gpu_count

        if _cluster_gpu_count() >= 1:
            names.append("cudf")
    except Exception:
        pass
    return names


BACKENDS = _backends()


@pytest.fixture(params=BACKENDS)
def be(request) -> DfBackend:
    return DfBackend(pytest.importorskip(request.param))


def _translated(op: str, shape: str, be: DfBackend):
    """`(device_table, engine_table)` for one cell, or `None` when the shape is not a chain."""
    table = INPUTS[shape]
    build, _sql = UNORDERED_OPS[op]
    dataset = build(bt.from_arrow(table))
    spec = gpu_plan_ops(dataset._plan)
    if spec is None:
        return None
    return be.to_arrow(run_chain(table, spec[1], be)), dataset.collect()


@pytest.mark.parametrize("op", sorted(set(UNORDERED_OPS) - STRUCTURAL))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_the_device_translation_matches_the_engine(op, shape, be):
    """Same rows *and* same column types as the CPU engine, on every edge-case input.

    Compared with `compare_results`, the same function `gpu_shadow_verify` uses at runtime, so
    a divergence found here is reported the way a production one would be — schema first,
    because both device bugs this package has shipped were column-type bugs whose values were
    correct.
    """
    pair = _translated(op, shape, be)
    if pair is None:
        pytest.skip(f"{op} is not a translatable chain shape")
    device, engine = pair

    difference = compare_results(device, engine)

    assert difference is None, f"{op}[{shape}] on {be.lib.__name__}: {difference}"


@pytest.mark.parametrize(
    "op", sorted(o for o, (_b, sql) in UNORDERED_OPS.items() if sql and o not in STRUCTURAL)
)
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_the_device_translation_matches_duckdb(duck, op, shape, be):
    """...and the external oracle agrees too, so both engines are not wrong together."""
    pair = _translated(op, shape, be)
    if pair is None:
        pytest.skip(f"{op} is not a translatable chain shape")
    device, _engine = pair
    duck.register("t", INPUTS[shape])

    assert_same(device, duck.sql(UNORDERED_OPS[op][1]))


def test_the_structural_shapes_are_the_only_ones_the_chain_matcher_declines():
    """Nothing may join `STRUCTURAL` silently — a new decline is coverage quietly lost.

    A shape that stops translating costs no correctness, only speed, so nothing else in the
    suite would notice the accelerated path narrowing operator by operator.
    """
    declined = {
        op
        for op in UNORDERED_OPS
        if gpu_plan_ops(UNORDERED_OPS[op][0](bt.from_arrow(INPUTS["base"]))._plan) is None
    }
    assert declined == STRUCTURAL, (
        f"chain-matcher declines changed: newly declined {sorted(declined - STRUCTURAL)}, "
        f"newly translated {sorted(STRUCTURAL - declined)}"
    )
