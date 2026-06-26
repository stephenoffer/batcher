"""A tiny benchmarking + correctness-checking framework.

Each query is expressed once per engine as a zero-argument callable that returns
a ``pyarrow.Table``. ``compare`` runs all engines, verifies they produce the same
result (as a sorted multiset of rows, tolerant of float rounding), and records
best-of-N wall-clock timings. ``print_table`` renders an aligned summary.

Correctness is checked *before* timings are trusted: if the engines disagree the
row is marked ``FAILED`` and a short diff is printed, but the suite continues.
"""

from __future__ import annotations

import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

import pyarrow as pa

# Absolute tolerance for float comparison (rounding differences between engines).
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-9

# Cells are canonicalized by rounding to this many decimals so equal-within-noise
# values hash into the same bucket for the multiset comparison.
_ROUND_DECIMALS = 6
# Two values that are genuinely equal (agree to ~1e-9) can still round into
# adjacent buckets, leaving them exactly one grid step (1e-6) apart. The pairwise
# float tolerance must therefore be at least one grid step, or that boundary
# produces a false mismatch. Real divergences in these queries are >= 1e-3, so a
# 1.5-step floor stays far from masking them.
_GRID_ATOL = 1.5 * 10**-_ROUND_DECIMALS


# --------------------------------------------------------------------------- #
# Result canonicalization (for correctness checks)
# --------------------------------------------------------------------------- #
def _canon_scalar(v):
    """Map a cell to a hashable, comparison-friendly form.

    Floats are rounded to absorb cross-engine rounding noise; NaN folds to a
    sentinel so two NaNs compare equal.
    """
    if v is None:
        return None
    if isinstance(v, Decimal):
        # DuckDB returns SUM over integers as a fixed-point Decimal; fold it into
        # the same numeric grid as ints/floats so it compares equal.
        return ("f", round(float(v), 6))
    if isinstance(v, float):
        if math.isnan(v):
            return ("nan",)
        # Round to a fixed grid coarser than the tolerance so equal-within-tol
        # values hash identically.
        return ("f", round(v, _ROUND_DECIMALS))
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, int):
        # An integer 4 and a float 4.0 from different engines should match.
        return ("f", round(float(v), _ROUND_DECIMALS))
    return v


def _rows_multiset(table: pa.Table) -> list[tuple]:
    """Return the table's rows as a sorted list of canonicalized tuples.

    Column *order* is normalized away by sorting column names, so two engines
    that emit the same columns in a different order still compare equal. Row
    order is normalized by sorting the row tuples (a multiset comparison).
    """
    cols = sorted(table.column_names)
    table = table.select(cols)
    pydict = table.to_pydict()
    n = table.num_rows
    rows = []
    for i in range(n):
        rows.append(tuple(_canon_scalar(pydict[c][i]) for c in cols))
    # Sort with a total order that tolerates None / mixed types by sorting on a
    # string projection (stable, deterministic, only used for multiset equality).
    rows.sort(key=lambda r: tuple(repr(x) for x in r))
    return rows


def _floats_close(a, b) -> bool:
    if isinstance(a, tuple) and isinstance(b, tuple) and a and b and a[0] == "f" and b[0] == "f":
        return math.isclose(a[1], b[1], rel_tol=FLOAT_RTOL, abs_tol=max(FLOAT_ATOL, _GRID_ATOL))
    return a == b


def results_match(reference: pa.Table, other: pa.Table) -> tuple[bool, str]:
    """Compare two tables as sorted row multisets. Returns (ok, message)."""
    if sorted(reference.column_names) != sorted(other.column_names):
        return False, (
            f"column mismatch: {sorted(reference.column_names)} vs {sorted(other.column_names)}"
        )
    if reference.num_rows != other.num_rows:
        return False, f"row count: {reference.num_rows} vs {other.num_rows}"

    ref_rows = _rows_multiset(reference)
    oth_rows = _rows_multiset(other)
    for i, (rr, oo) in enumerate(zip(ref_rows, oth_rows, strict=False)):
        if rr == oo:
            continue
        if len(rr) == len(oo) and all(_floats_close(a, b) for a, b in zip(rr, oo, strict=True)):
            continue
        return False, f"row {i} differs: {rr!r} vs {oo!r}"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def bench(fn: Callable[[], object], runs: int = 5) -> float:
    """Time ``fn`` best-of-``runs`` in milliseconds (one warm-up first)."""
    fn()  # warm up
    best = math.inf
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000.0
        best = min(best, dt)
    return best


@dataclass
class EngineResult:
    ms: float | None = None
    error: str | None = None
    correct: bool | None = None  # None until checked


@dataclass
class CompareResult:
    name: str
    engines: dict[str, EngineResult] = field(default_factory=dict)
    status: str = "OK"  # OK | FAILED | ERROR
    note: str = ""


def compare(
    name: str,
    fns: dict[str, Callable[[], pa.Table] | None],
    engines: list[str],
    runs: int = 5,
) -> CompareResult:
    """Run each engine's query, verify equality, and record timings.

    ``fns`` maps engine name -> callable returning a ``pyarrow.Table`` (or
    ``None`` to mark the case "n/a" for that engine). ``engines`` is the resolved
    lineup (and report order). Correctness is checked against the first engine that
    produced a result.
    """
    result = CompareResult(name=name)
    outputs: dict[str, pa.Table] = {}

    # First, execute each engine once to obtain a result (and catch failures).
    for engine in engines:
        fn = fns.get(engine)
        er = EngineResult()
        if fn is None:
            er.error = "n/a"
            result.engines[engine] = er
            continue
        try:
            out = fn()
            if not isinstance(out, pa.Table):
                out = pa.table(out) if isinstance(out, dict) else pa.Table.from_pandas(out)
            outputs[engine] = out
        except Exception as exc:
            er.error = f"{type(exc).__name__}: {exc}"
            er.ms = None
            tb = traceback.format_exc().strip().splitlines()
            er.error += " | " + tb[-1] if tb else ""
            result.engines[engine] = er
            continue
        result.engines[engine] = er

    # Correctness: compare every produced output to a reference.
    if outputs:
        ref_engine = next(iter(outputs))
        ref = outputs[ref_engine]
        mismatches = []
        for engine, out in outputs.items():
            ok, msg = results_match(ref, out)
            result.engines[engine].correct = ok
            if not ok:
                mismatches.append(f"{engine} != {ref_engine}: {msg}")
        if mismatches:
            result.status = "FAILED"
            result.note = " ; ".join(mismatches)
    else:
        result.status = "ERROR"
        result.note = "all engines failed"

    # Timing: only time engines that produced a result. Even on a correctness
    # FAILURE we time them (useful signal), but the row stays marked FAILED.
    for engine in outputs:
        fn = fns[engine]
        try:
            result.engines[engine].ms = bench(fn, runs=runs)
        except Exception as exc:
            result.engines[engine].error = f"timing failed: {exc}"

    if result.status == "OK" and any(e.error and e.error != "n/a" for e in result.engines.values()):
        # At least one engine errored out (but others agreed). Flag it.
        errs = [
            f"{name}: {e.error}"
            for name, e in result.engines.items()
            if e.error and e.error != "n/a"
        ]
        result.note = " ; ".join(errs)
        result.status = "PARTIAL"

    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_ms(er: EngineResult) -> str:
    if er.error == "n/a":
        return "n/a"
    if er.error:
        return "ERR"
    if er.ms is None:
        return "-"
    return f"{er.ms:.1f}"


def print_table(results: list[CompareResult], engines: list[str]) -> None:
    """Print an aligned table: query | per-engine ms | batcher/<engine> ratios | status.

    Columns are driven by ``engines`` (the resolved lineup), so the table adapts to
    whatever single-node or multi-node engines were selected. A ``b/<engine>`` ratio
    is shown for every comparator when Batcher is in the lineup.
    """
    has_batcher = "batcher" in engines
    comparators = [e for e in engines if e != "batcher"]
    headers = ["query"] + [f"{e}_ms" for e in engines]
    if has_batcher:
        headers += [f"b/{e}" for e in comparators]
    headers += ["status"]

    rows = []
    for r in results:
        cells = [r.name] + [_fmt_ms(r.engines.get(e, EngineResult())) for e in engines]
        if has_batcher:
            b = r.engines.get("batcher", EngineResult())
            for e in comparators:
                ce = r.engines.get(e, EngineResult())
                cells.append(f"{b.ms / ce.ms:.2f}x" if b.ms and ce.ms else "-")
        cells.append(r.status)
        rows.append(cells)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            if i == 0:
                out.append(cell.ljust(widths[i]))
            else:
                out.append(cell.rjust(widths[i]))
        return "  ".join(out)

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(fmt_row(headers))
    print(line)
    for row in rows:
        print(fmt_row(row))

    # Footnotes for any failed / partial rows.
    notes = [r for r in results if r.note]
    if notes:
        print()
        for r in notes:
            print(f"[{r.status}] {r.name}: {r.note}")
