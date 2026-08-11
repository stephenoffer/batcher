"""A deep `Expr` must fail cleanly or succeed — never take the interpreter down.

`Expr` is a recursive enum on both sides of the FFI, and every pass over one — the control
plane's `walk`, the engine's `eval`, its analyses, and the compiler-generated `Drop` —
descends it with one stack frame per level. On rayon's default 2 MiB worker stack that
ceiling was about **eighty nested nodes**, and going past it was a `SIGSEGV` that killed the
process rather than an error a caller could catch.

Eighty is not a hypothetical limit. Three ordinary things exceeded it, and all three are
pinned below:

* ``col(x).is_in(values)`` desugared to one `OR` level per value, so a hundred values crashed
  a projection. ``TfidfVectorizer(stop_words="english")`` passes 318.
* ``IsotonicCalibrator`` sums one threshold indicator per bucket, and ``n_bins`` defaults to
  100.
* A plain hundred-term arithmetic chain, which a generated query reaches easily.

Every case here runs in a **subprocess**. A regression is a segfault, and a segfault inside
the test worker is a lost run with no result rather than a failure — running it out of process
is what turns it back into an assertion.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.unit


def _run(body: str) -> subprocess.CompletedProcess[str]:
    """Execute `body` in a fresh interpreter and return the finished process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _assert_not_a_crash(done: subprocess.CompletedProcess[str]) -> None:
    """The process may succeed or raise, but must not die on a signal.

    A negative `returncode` is a signal death (`-11` is SIGSEGV); `139` is the shell's
    spelling of the same thing. Anything else — including a clean `RecursionError` — is a
    failure the caller can see and handle, which is all this file asks for.
    """
    assert done.returncode >= 0, f"died on signal {-done.returncode}\n{done.stderr[-2000:]}"
    assert done.returncode != 139, f"segfault\n{done.stderr[-2000:]}"


@pytest.mark.parametrize("n", [100, 318])
def test_is_in_over_many_values_survives_a_projection(n: int) -> None:
    """`is_in` in a *projection*, which no optimizer rule folds on its way to the engine.

    Under a `Filter` this always worked, because `or_equalities_to_in_list` folded the chain
    into an `InList` before it was serialized. A projection got no such rewrite, so the
    n-deep chain reached the engine intact — which is why the crash showed up in a vectorizer
    and not in any of the `IN`-in-a-`WHERE` tests.
    """
    done = _run(f"""
        import batcher as bt
        values = [f"w{{i}}" for i in range({n})]
        out = bt.from_pydict({{"t": ["cat", "w5"]}}).select(hit=bt.col("t").is_in(values))
        assert out.to_pydict()["hit"] == [False, True]
        print("ok")
    """)
    _assert_not_a_crash(done)
    assert done.returncode == 0, done.stderr[-2000:]


def test_is_in_lowers_to_a_set_not_a_chain_of_equalities() -> None:
    """The fold itself: a literal set becomes one `InList` node, not `n` nested `OR`s.

    This is what keeps the IR shallow at the source, and it is also why the projection above
    is *fast* — the engine probes one hash set per row instead of running `n` full-column
    comparisons.
    """
    from batcher.plan.expr_ir import Binary, InList
    from batcher.plan.expr_ir.constructors import col

    assert isinstance(col("x").is_in([1, 2, 3]), InList)
    assert isinstance(col("x").is_in(["a", "b"]), InList)
    # A set the engine cannot build one typed hash set from stays a disjunction, which is
    # correct rather than merely conservative — see `_in_list_foldable`.
    assert isinstance(col("x").is_in([1, "a"]), Binary)  # mixed types
    assert isinstance(col("x").is_in([True]), Binary)  # bool
    assert isinstance(col("x").is_in([float("nan"), 1.0]), Binary)  # NaN never equals itself
    assert isinstance(col("x").is_in([col("y"), 1]), Binary)  # an expression member


def test_isotonic_calibrator_fits_at_its_default_bin_count() -> None:
    """`IsotonicCalibrator` with enough distinct scores to use all 100 default bins.

    Needs ~700 rows: `_cut_points` drops repeated quantiles, so a smaller split ties its way
    down to a chain short enough to survive and the crash hides.
    """
    done = _run("""
        import numpy as np, batcher as bt
        from batcher.ml import IsotonicCalibrator
        rng = np.random.default_rng(7)
        s = rng.random(800)
        y = (rng.random(800) < s).astype(int)
        ds = bt.from_pydict({"p": s.tolist(), "y": y.tolist()})
        fitted = IsotonicCalibrator("p", "y").fit(ds)
        assert fitted.values_ == sorted(fitted.values_), "isotonic output must be monotone"
        out = fitted.transform(ds).to_pydict()["calibrated"]
        assert all(0.0 <= v <= 1.0 for v in out)
        print("ok")
    """)
    _assert_not_a_crash(done)
    assert done.returncode == 0, done.stderr[-2000:]


def test_a_long_arithmetic_chain_does_not_take_the_process_down() -> None:
    """300 chained additions — well past the old ~80 ceiling."""
    done = _run("""
        import batcher as bt
        from batcher.plan.expr_ir.constructors import col, lit
        e = lit(0)
        for i in range(300):
            e = e + (col("x") > lit(float(i))).cast("int64")
        assert bt.from_pydict({"x": [10.5]}).select(r=e).to_pydict()["r"] == [11]
        print("ok")
    """)
    _assert_not_a_crash(done)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_deepest_plan_the_guard_admits_can_actually_be_evaluated() -> None:
    """`MAX_PLAN_DEPTH` and `WORKER_STACK_BYTES` are a pair, and this is the seam.

    `bc_ir::MAX_PLAN_DEPTH` (512) rejects a plan whose IR nests deeper, so deserialization
    cannot overflow. But it was calibrated against *parsing*, at ~3.2 KiB of stack per level,
    and evaluation costs ~20 KiB — so a 2 MiB worker aborted around 104 levels while the
    guard was still admitting everything under 512. Every depth in that window was a signal
    death rather than an error.

    509 levels is the deepest this construction fits under the guard. If someone raises
    `MAX_PLAN_DEPTH`, or lowers the worker stack, this is the test that notices — and it
    notices as a failure rather than as a lost run, because it is out of process.

    The recursion limit is raised deliberately. There is a *third* bound in front of the two
    being tested: the control plane's own `walk`/`to_ir` are recursive and give up around 350
    levels at Python's default of 1,000, which is a clean `RecursionError` and is what an
    ordinary caller meets first. That bound would mask this one, so it is lifted here to put
    the engine's capacity under test rather than Python's.
    """
    done = _run("""
        import sys
        sys.setrecursionlimit(200000)
        import batcher as bt
        from batcher.plan.expr_ir.constructors import col, lit
        e = lit(0)
        for i in range(505):          # 509 IR levels — just under MAX_PLAN_DEPTH
            e = e + (col("x") > lit(float(i))).cast("int64")
        assert bt.from_pydict({"x": [10.5]}).select(r=e).to_pydict()["r"] == [11]
        print("ok")
    """)
    _assert_not_a_crash(done)
    assert done.returncode == 0, done.stderr[-2000:]


def test_a_plan_past_the_guard_raises_a_typed_error() -> None:
    """Past `MAX_PLAN_DEPTH` the answer is `PlanTooDeepError`, naming the depth and the fix.

    Needs the caller's own recursion limit raised, because Python's `to_ir()` is recursive
    and gives up first at the default 1000 — which is itself part of why the window above
    stayed hidden for so long.
    """
    done = _run("""
        import sys
        sys.setrecursionlimit(200000)
        import batcher as bt
        from batcher._internal.native import engine  # noqa: F401
        from batcher.plan.expr_ir.constructors import col, lit
        e = lit(0)
        for i in range(800):
            e = e + (col("x") > lit(float(i))).cast("int64")
        try:
            bt.from_pydict({"x": [1.0]}).select(r=e).to_pydict()
        except Exception as exc:
            assert type(exc).__name__ == "PlanTooDeepError", type(exc).__name__
            assert "levels deep" in str(exc), str(exc)
            print("raised")
        else:
            raise AssertionError("a plan past MAX_PLAN_DEPTH was accepted")
    """)
    _assert_not_a_crash(done)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "raised" in done.stdout


def test_a_chain_past_the_ceiling_raises_instead_of_crashing() -> None:
    """Beyond what the stack can hold the answer is an exception, not a dead process.

    The ceiling is raised, not removed, and this is the half that matters for a service: a
    caller can catch a `RecursionError` and reject the query. It cannot catch a `SIGSEGV`.
    """
    done = _run("""
        import batcher as bt
        from batcher.plan.expr_ir.constructors import col, lit
        e = lit(0)
        for i in range(5000):
            e = e + (col("x") > lit(float(i))).cast("int64")
        try:
            bt.from_pydict({"x": [1.0]}).select(r=e).to_pydict()
        except RecursionError:
            print("raised")
    """)
    _assert_not_a_crash(done)
