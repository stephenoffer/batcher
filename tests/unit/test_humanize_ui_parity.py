"""The Python and JavaScript number formatters must agree, value for value.

Batcher renders the same measurement in two places: the terminal (`_internal.humanize`)
and the web dashboard (`observe/assets/ui.js`). Several modules asserted in their own
docstrings that the two matched "exactly", and that "a differential test pins the two
together". No such test existed, and the two did not match — the browser rendered a 9
microsecond operator as ``9µs`` while the terminal rendered it as ``0.0ms``, and the
profile's byte formatter divided by 1024 while labelling the result ``KB``.

This is that test. It executes the real `ui.js` in QuickJS and the real Python functions
over one shared vector table, so a rule changed on one side and not the other fails here
rather than being noticed by a user comparing two windows.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from batcher._internal.humanize import byte_size, count, duration_ms, duration_s, percent

pytestmark = pytest.mark.unit

_ASSETS = Path(__file__).resolve().parents[2] / "python" / "batcher" / "observe" / "assets"

#: Chosen to land on every branch and every boundary of both implementations, including
#: the ones that had actually drifted: sub-millisecond durations, the 1024 boundary, and a
#: share small enough to round to zero.
_MS = [0, 0.0004, 0.009, 0.5, 1, 4.2, 9.99, 10, 850, 999.4, 1000, 1500, 59_999, 60_000, 63_000]
_BYTES = [0, 1, 512, 1023, 1024, 1536, 1048576, 1073741824, 1099511627776, 1125899906842624]
_COUNTS = [0, 1, 0.5, 999, 1000, 1500, 999_999, 2_400_000, 1_500_000_000, 3.4e12, -1500]
_FRACTIONS = [0, 0.0003, 0.005, 0.01, 0.62, 1.0, 1.5]
_SECONDS = [0, 0.4, 1, 42, 59.6, 60, 90, 3599, 3600, 3725, 86_400]


@pytest.fixture(scope="module")
def js():
    """A QuickJS context with `ui.js` loaded, or a skip when QuickJS is unavailable."""
    quickjs = pytest.importorskip("quickjs")
    ctx = quickjs.Context()
    # `ui.js` touches `localStorage` at its top level; the dashboard's own test module
    # stubs a browser, and this needs only the storage shim to get through module load.
    ctx.eval(
        "var localStorage={_d:{},getItem(k){return this._d[k]||null;},"
        "setItem(k,v){this._d[k]=String(v);},removeItem(k){delete this._d[k];}};"
        "var window={addEventListener(){},location:{hash:''}};"
        "var document={addEventListener(){},getElementById(){return null;},"
        "querySelectorAll(){return [];},createElement(){return {style:{},classList:"
        "{add(){},remove(){}},appendChild(){},setAttribute(){}};},body:{appendChild(){}}};"
    )
    ctx.eval((_ASSETS / "ui.js").read_text())
    return ctx


def _js_call(ctx, fn: str, value: float) -> str:
    """Call `UI.<fn>` with a numeric literal that survives the JS parser exactly."""
    return ctx.eval(f"String(UI.{fn}({value!r}))")


@pytest.mark.parametrize("ms", _MS)
def test_duration_ms_matches_ui_ms(js, ms):
    """`duration_ms` is the Python half of ``UI.ms``; a divergence here is user-visible."""
    assert duration_ms(ms) == _js_call(js, "ms", ms)


@pytest.mark.parametrize("n", _BYTES)
def test_byte_size_matches_ui_bytes(js, n):
    """`byte_size` is the Python half of ``UI.bytes`` — binary divisor, binary label."""
    assert byte_size(n) == _js_call(js, "bytes", n)


@pytest.mark.parametrize("n", _COUNTS)
def test_count_matches_ui_count(js, n):
    assert count(n) == _js_call(js, "count", n)


@pytest.mark.parametrize("fraction", _FRACTIONS)
def test_percent_matches_ui_pct(js, fraction):
    assert percent(fraction) == _js_call(js, "pct", fraction)


@pytest.mark.parametrize("secs", _SECONDS)
def test_duration_s_matches_ui_duration(js, secs):
    """A *span* is `UI.duration`, not `UI.ms` — the two shapes answer different questions."""
    assert duration_s(secs) == _js_call(js, "duration", secs)


def test_unknown_marker_is_the_same_em_dash_everywhere(js):
    """`None`/NaN must render as the dashboard's em dash, never as `0` or `nan`."""
    assert duration_ms(None) == duration_ms(float("nan")) == js.eval("UI.ms(null)")
    assert byte_size(None) == js.eval("UI.bytes(null)")
    assert count(float("nan")) == js.eval("UI.count(NaN)")
    assert percent(float("nan")) == js.eval("UI.pct(NaN)")
    assert duration_s(-1) == js.eval("UI.duration(-1)")


def test_binary_and_decimal_byte_units_are_labelled_for_the_divisor_they_used():
    """The bug this module exists for: a 1024 divisor may not wear a ``KB`` label.

    `plan.profile.human_bytes` printed ``1.0KB`` for 1024 bytes — off by 2.4% at KiB and
    by 10% at TiB, in the one output a person reads to decide whether a spill was large.
    """
    assert byte_size(1024) == "1.0 KiB"
    assert byte_size(1000, binary=False) == "1.0 kB"
    assert byte_size(1024, binary=False) == "1.0 kB"
    # A terabyte is where the mislabelling became a materially wrong number.
    assert byte_size(2**40) == "1.0 TiB"
    assert byte_size(10**12, binary=False) == "1.0 TB"


def test_sub_millisecond_work_is_reported_rather_than_rounded_away():
    """The specific regression: the fastest operators in a plan read as unmeasured."""
    assert duration_ms(0.009) == "9µs"
    assert duration_ms(0.0004) == "1µs"  # floors to at least one, never to "0µs"
    assert duration_ms(0) == "0"  # a real zero is distinguishable from unmeasured
    assert duration_ms(None) == "—"


def test_signed_ratio_names_the_direction_the_estimate_missed():
    """A bare ``0.9x`` leaves the reader to work out which way the optimizer was wrong."""
    from batcher._internal.humanize import signed_ratio

    assert signed_ratio(1000, 300) == "3.3x under"
    assert signed_ratio(300, 1000) == "3.3x over"
    assert signed_ratio(100, 100) == "exact"
    assert signed_ratio(100, 0) == "—"
    assert signed_ratio(100, math.nan) == "—"
