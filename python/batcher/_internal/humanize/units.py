"""The engine's one set of unit formatters — durations, byte sizes, counts, shares.

Every surface Batcher renders a number on has to answer the same four questions: how
long, how big, how many, what share. Before this module each surface answered them
itself, and the copies had already drifted in ways a reader would notice:

* `plan.profile` printed ``1.0KB`` for 1024 bytes — a *decimal* label on a *binary*
  divisor, so every byte size in ``explain(analyze=True)`` was mislabelled by 2.4% per
  power, reaching 10% at TiB.
* The console's duration formatter claimed in its own docstring to match the web UI's
  ``UI.ms`` "exactly", with "a differential test pinning the two together". No such test
  existed, and the two disagreed on everything below a millisecond: the browser rendered
  ``9µs`` where the terminal rendered ``0.0ms``.
* `observe.console` and `observe.inference.measures` carried byte-identical copies of
  `count` and `percent`.

So the rule here is parity, not preference: these functions are the Python half of the
contract whose JavaScript half is ``UI.ms`` / ``UI.bytes`` / ``UI.pct`` in
`observe/assets/ui.js`, and `tests/unit/test_humanize_ui_parity.py` executes both
implementations over a shared vector table and fails when they diverge. If you change a
rendering rule here, change it there in the same commit.

Layer 0 (`_internal`) because the callers span every layer — `plan` renders a profile,
`observe` renders a progress bar, `io` renders a manifest, `api` renders an error — and
layer 0 is the only place all four can import from.
"""

from __future__ import annotations

import math

__all__ = [
    "byte_size",
    "count",
    "duration_ms",
    "duration_s",
    "ordinal",
    "percent",
    "plural",
    "rate",
    "signed_ratio",
]

#: What every formatter renders for "not measured". An em dash rather than ``0``, because
#: ``0 bytes`` is a *reading* and the absence of one is not; conflating them is how an
#: unmeasured path comes to look like a measured-and-empty one. Matches ``ui.js``.
UNKNOWN = "—"

_BINARY_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
_DECIMAL_UNITS = ("B", "kB", "MB", "GB", "TB", "PB")
_COUNT_STEPS = ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K"))


def duration_ms(ms: float | None) -> str:
    """A duration given in milliseconds, in the unit a person would say it in.

    ``9µs``, ``4.2ms``, ``820ms``, ``4.20s``, ``1m03s``. Sub-millisecond values keep
    microsecond resolution rather than collapsing to ``0.0ms``, which is the difference
    between "this operator was fast" and "this operator was not measured".

    Args:
        ms: Milliseconds, or `None`/NaN when nothing was measured.

    Returns:
        The rendered duration, or the unknown marker.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import duration_ms
            >>> duration_ms(0.009), duration_ms(4.2), duration_ms(820)
            ('9\\xb5s', '4.2ms', '820ms')
            >>> duration_ms(63000)
            '1m03s'
    """
    if ms is None or (isinstance(ms, float) and math.isnan(ms)):
        return UNKNOWN
    if ms == 0:
        return "0"
    if ms < 0:
        return f"-{duration_ms(-ms)}"
    if ms < 1:
        return f"{max(1, round(ms * 1000))}µs"
    if ms < 1000:
        return f"{ms:.1f}ms" if ms < 10 else f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1000:.2f}s"
    if ms < 3_600_000:
        minutes, seconds = divmod(ms / 1000, 60)
        return f"{int(minutes)}m{seconds:02.0f}s"
    hours, remainder = divmod(ms / 1000, 3600)
    minutes = remainder // 60
    return f"{int(hours)}h{int(minutes):02d}m"


def duration_s(seconds: float | None) -> str:
    """A coarse clock span in seconds — an elapsed time or an ETA, not a measurement.

    Deliberately a different shape from `duration_ms`, and matching ``UI.duration`` rather
    than ``UI.ms``, because the two answer different questions. A *measurement* wants
    resolution down to the microsecond; a *span* a person is waiting out wants ``2m 30s``,
    where a third significant figure would only flicker.

    A negative or non-finite span renders as unknown rather than as ``-3s left``: a clock
    skew or a not-yet-started run produces both, and a reader will try to interpret either.

    Args:
        seconds: The span, or `None` when nothing was measured.

    Returns:
        The rendered span, or the unknown marker.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import duration_s
            >>> duration_s(0.4), duration_s(42), duration_s(90), duration_s(3725)
            ('<1s', '42s', '1m 30s', '1h 2m')
    """
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return UNKNOWN
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{round(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {round(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def byte_size(n: float | None, *, binary: bool = True) -> str:
    """A byte size with its unit, binary (``KiB``) by default.

    Binary is the default because it is what the engine actually measures — buffer pool
    envelopes, morsel sizes, and spill volumes are all powers of two — and because
    labelling a 1024-divisor result ``KB`` is simply wrong. Pass ``binary=False`` for the
    decimal form a storage vendor or a cloud bill uses.

    Args:
        n: Bytes, or `None`/`0` when nothing was measured.
        binary: Use 1024-based ``KiB``/``MiB`` units; `False` selects 1000-based ``kB``.

    Returns:
        The rendered size, or the unknown marker for `None` and `0`.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import byte_size
            >>> byte_size(1536), byte_size(1536, binary=False)
            ('1.5 KiB', '1.5 kB')
            >>> byte_size(0)
            '\\u2014'
    """
    if not n:
        return UNKNOWN
    negative = n < 0
    size = float(abs(n))
    step = 1024.0 if binary else 1000.0
    units = _BINARY_UNITS if binary else _DECIMAL_UNITS
    for unit in units:
        if size < step or unit == units[-1]:
            body = f"{size:g} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            return f"-{body}" if negative else body
        size /= step
    return str(n)  # pragma: no cover - the loop above always returns


def count(n: float | None, *, digits: int = 1) -> str:
    """A compact SI-style count: ``842``, ``1.2K``, ``3.4M``, ``5.6B``, ``7.8T``.

    Args:
        n: The count, or `None` when nothing was measured.
        digits: Decimal places on the abbreviated forms.

    Returns:
        The rendered count, or the unknown marker.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import count
            >>> count(842), count(1234), count(3_400_000)
            ('842', '1.2K', '3.4M')
    """
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return UNKNOWN
    for limit, suffix in _COUNT_STEPS:
        if abs(n) >= limit:
            return f"{n / limit:.{digits}f}{suffix}"
    # `math.floor(n + 0.5)`, not `f"{n:.0f}"`: Python rounds halves to even and JavaScript's
    # `Math.round` rounds them up, so `count(0.5)` would read `0` here and `1` in the
    # browser. The parity table in `tests/unit/test_humanize_ui_parity.py` includes that
    # exact value, which is how the difference was found rather than shipped.
    return f"{math.floor(n + 0.5):.0f}"


def percent(fraction: float | None, *, digits: int = 0) -> str:
    """A share of one, rendered as a percentage.

    A share that is small but present reads ``<1%``, never ``0%`` — the second says
    "nothing here", which is a different and wrong claim. A share above 100% is left
    as-is: operator time summed across threads legitimately exceeds wall-clock, so it is
    a real reading and clamping it would hide the most interesting case.

    Args:
        fraction: The share, where ``1.0`` is 100%. `None`/NaN render as unknown.
        digits: Decimal places.

    Returns:
        The rendered percentage, or the unknown marker.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import percent
            >>> percent(0.62), percent(0.0003), percent(0.0)
            ('62%', '<1%', '0%')
    """
    if fraction is None or (isinstance(fraction, float) and math.isnan(fraction)):
        return UNKNOWN
    if not fraction:
        return "0%"
    value = fraction * 100
    if 0 < value < 1:
        return "<1%"
    return f"{value:.{digits}f}%"


def rate(n: float | None, unit: str = "rows") -> str:
    """A throughput, as ``1.2M rows/s``.

    Args:
        n: The per-second figure, or `None` when nothing was measured.
        unit: The noun being counted.

    Returns:
        The rendered rate, or the unknown marker.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import rate
            >>> rate(1_240_000)
            '1.2M rows/s'
    """
    return UNKNOWN if n is None else f"{count(n)} {unit}/s"


def plural(n: int, singular: str, many: str | None = None) -> str:
    """``1 file`` / ``3 files`` — the count and its correctly inflected noun.

    Args:
        n: The count.
        singular: The singular noun.
        many: The plural noun; defaults to `singular` + ``"s"``.

    Returns:
        The count and noun, space-separated.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import plural
            >>> plural(1, "file"), plural(3, "file"), plural(2, "entry", "entries")
            ('1 file', '3 files', '2 entries')
    """
    noun = singular if abs(n) == 1 else (many if many is not None else f"{singular}s")
    return f"{n:,} {noun}"


def ordinal(n: int) -> str:
    """``1st``, ``2nd``, ``3rd``, ``11th`` — for naming an attempt or a round.

    Args:
        n: The position.

    Returns:
        The ordinal string.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import ordinal
            >>> ordinal(1), ordinal(11), ordinal(22)
            ('1st', '11th', '22nd')
    """
    if 10 <= abs(n) % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(abs(n) % 10, "th")
    return f"{n}{suffix}"


def signed_ratio(actual: float, estimate: float) -> str:
    """How far an estimate missed, as ``3.4x over`` / ``2.1x under`` / ``exact``.

    The optimizer's estimate error is the single most diagnostic number in an
    ``explain(analyze=True)``, and the bare ratio it used to print (``(0.9x)``) left the
    reader to work out which direction 0.9 was. Naming the direction is the whole value.

    Args:
        actual: The measured row count.
        estimate: The estimated row count.

    Returns:
        The rendered miss, or the unknown marker when the estimate is absent.

    Examples:
        .. doctest::

            >>> from batcher._internal.humanize import signed_ratio
            >>> signed_ratio(1000, 300), signed_ratio(300, 1000), signed_ratio(100, 100)
            ('3.3x under', '3.3x over', 'exact')
            >>> signed_ratio(200, 1e-12)   # an estimate below one row compares against one
            '200.0x under'
            >>> signed_ratio(200, 1_000_000)   # a real miss is reported in full
            '5000.0x over'
    """
    if not estimate or math.isnan(estimate) or math.isnan(actual):
        return UNKNOWN
    # Both sides are floored at one row before the division. A selectivity estimate
    # underflows toward zero down a chain of independent predicates -- each one multiplies
    # the last -- so a six-filter plan reaches a denominator like 9.7e-13 and printed
    # ``205891132094649.4x under``: fifteen significant digits of an artifact, in a column
    # whose width every operator on the plan then pays for.
    #
    # The magnitude was never the problem, the denominator was. A row count is a count, so
    # an estimate below one row means "the optimizer expects nothing" -- which is what the
    # `est≈0` printed beside it already says -- and the honest comparison is against one
    # row. That makes the example above read ``200.0x under``, and leaves a *real* miss
    # untouched: an estimate of 1,000,000 against 200 actual rows is still ``5000.0x over``,
    # which is exactly the signal the diagnosis section exists to raise. A cap on the
    # rendered figure would have suppressed that one too.
    scale = max(estimate, 1.0)
    measured = max(actual, 1.0)
    ratio = measured / scale if measured > scale else scale / measured
    # A ratio that would render as "1.0x" is reported as exact rather than as a miss. The
    # alternative reads as a contradiction: a row showing `est≈1  actual=1` beside
    # `1.0x over` invites the reader to hunt for a rounding they cannot see, and a 4%
    # estimate error is not a fact anyone acts on.
    if ratio < 1.05:
        return "exact"
    return f"{ratio:.1f}x under" if measured > scale else f"{ratio:.1f}x over"
