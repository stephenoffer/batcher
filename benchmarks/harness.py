"""A tiny benchmarking + correctness-checking framework.

Each query is expressed once per engine as a zero-argument callable that returns
a ``pyarrow.Table``. ``compare`` runs all engines, verifies they produce the same
result (as a sorted multiset of rows, tolerant of float rounding), and records
best-of-N wall-clock timings. ``print_table`` renders an aligned summary.

Correctness is checked *before* timings are trusted: if the engines disagree the
row is marked ``FAILED`` and a short diff is printed, but the suite continues. The
comparison (``column_classes`` / ``to_rowset`` / ``rowsets_match``) is vectorized
over Arrow, because a row-wise one costs more than the queries it is gating.
"""

from __future__ import annotations

import json
import math
import re
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

__all__ = [
    "FLOAT_ATOL",
    "FLOAT_RTOL",
    "RESULT_PREFIX",
    "CompareResult",
    "EngineResult",
    "bench",
    "compare",
    "emit_result",
    "print_table",
    "results_match",
    "run_isolated",
]


# Absolute / relative tolerance for the pairwise float comparison.
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-9

# Floats are rounded to this many decimals *before* sorting, so two values that are
# equal within tolerance land on the same grid point and therefore sort together.
ROUND_DECIMALS = 6
# Rounding maps two genuinely-equal values (agreeing to ~1e-9) onto adjacent grid
# points at worst, leaving them one step (1e-6) apart. The pairwise tolerance must
# clear one step or that boundary is a false mismatch. Real divergences in these
# queries are >= 1e-3, far above this floor.
GRID_ATOL = 1.5 * 10**-ROUND_DECIMALS

# Comparison classes a column can be reconciled to, in widening order: a column's class
# across engines is the widest any engine assigns it.
_INT, _FLOAT, _BOOL, _STR = "int", "float", "bool", "str"


def _class_of(dtype: pa.DataType) -> str:
    """The comparison class of one engine's column type."""
    if pa.types.is_boolean(dtype):
        return _BOOL
    if pa.types.is_integer(dtype):
        return _INT
    if pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
        return _FLOAT
    return _STR


def _widen(a: str, b: str) -> str:
    """Reconcile two engines' classes for the same column.

    ``int`` widens to ``float`` (DuckDB's ``Decimal`` sum vs Batcher's ``int64``), and
    anything mixed with a non-numeric class falls back to string equality.
    """
    if a == b:
        return a
    if {a, b} == {_INT, _FLOAT}:
        return _FLOAT
    return _STR


# A derived column with no alias has no name in the query, so each engine invents one, and
# they disagree in ways that are pure spelling: DuckDB qualifies a built-in with its catalog
# and quotes it (``main."substring"(s_city, 1, 30)`` against ``substring(s_city, 1, 30)``) and
# parenthesizes sub-expressions it did not have to (``round((a / b), 2)`` against
# ``round(a / b, 2)``, ``((cast(a) / cast(b)) * 100)`` against ``cast(a) / cast(b) * 100``).
#
# `column_classes` already lowercased names for exactly this reason — the engines disagree on
# a generated name's *case* — but that covered only one of the three ways they disagree, so
# TPC-DS q2, q61, q79 and q85 were each reported as a correctness FAILURE over data that
# matched. Squeezing out the catalog prefix, the quotes, the whitespace and the parentheses
# leaves the one thing both engines do agree on, and a genuinely different column set still
# fails: two columns that squeeze to one name are two spellings of the same expression, and
# if they were not, the values would then disagree and the row would fail anyway.
_CATALOG_PREFIX = re.compile(r"\bmain\.")
_DROPPED_PUNCTUATION = str.maketrans("", "", ' "()')


def canonical_column_name(name: str) -> str:
    """The name a column is compared under, with each engine's spelling squeezed out."""
    return _CATALOG_PREFIX.sub("", name.lower()).translate(_DROPPED_PUNCTUATION)


def canonical_names(table: pa.Table) -> list[str]:
    """`table`'s column names canonicalized, or merely lowercased if that would collide.

    Two columns of one result squeezing to the same name would silently drop one of them from
    the comparison, which is the one outcome worse than the false failure this fixes. The
    lowercased fallback is exactly the behaviour that preceded canonicalization.
    """
    canonical = [canonical_column_name(n) for n in table.column_names]
    if len(set(canonical)) != len(canonical):
        return [n.lower() for n in table.column_names]
    return canonical


def column_classes(tables: list[pa.Table]) -> dict[str, str]:
    """The comparison class of every column, reconciled across all engines' outputs.

    Columns are keyed by `canonical_column_name`: SQL identifiers are case-insensitive, and
    for a *derived* column the engines invent different names for the same expression.

    Args:
        tables: Every engine's result for one query.

    Returns:
        Canonical column name mapped to its comparison class.
    """
    classes: dict[str, str] = {}
    for table in tables:
        for key, col_field in zip(canonical_names(table), table.schema, strict=True):
            cls = _class_of(col_field.type)
            classes[key] = _widen(classes[key], cls) if key in classes else cls
    return classes


def _canon_column(col: pa.ChunkedArray, cls: str) -> pa.Array:
    """Cast one column into its reconciled class, rounding floats onto the grid."""
    if cls == _FLOAT:
        out = pc.round(col.cast(pa.float64()), ndigits=ROUND_DECIMALS)
    elif cls == _INT:
        # int64 spans every signed width; a uint64 past 2^63 would need uint64, but no
        # benchmark column produces one and float64 would lose it silently.
        out = col.cast(pa.int64())
    elif cls == _BOOL:
        out = col
    else:
        out = pc.cast(col, pa.large_string(), safe=False)
    return out.combine_chunks() if isinstance(out, pa.ChunkedArray) else out


@dataclass
class RowSet:
    """One engine's result, canonicalized and sorted — ready for multiset comparison."""

    table: pa.Table
    classes: dict[str, str]


def to_rowset(table: pa.Table, classes: dict[str, str]) -> RowSet:
    """Canonicalize and sort ``table`` so two results compare as row multisets.

    Columns are reordered by lowercased name and cast to their reconciled class, then
    the whole table is sorted on every column. Any total order works for a multiset
    comparison as long as both sides use the same one, and sorting after rounding keeps
    equal-within-tolerance rows adjacent.

    Args:
        table: One engine's result.
        classes: The reconciled classes from :func:`column_classes`.

    Returns:
        The canonicalized, sorted rowset.
    """
    keyed = sorted(zip(canonical_names(table), table.column_names, strict=True))
    canon = pa.table({key: _canon_column(table.column(n), classes[key]) for key, n in keyed})
    if canon.num_rows > 1 and canon.num_columns:
        keys = [(n, "ascending") for n in canon.column_names]
        canon = canon.sort_by(keys)
    return RowSet(table=canon, classes=classes)


def _agree(cls: str, ref: pa.Array, oth: pa.Array) -> pa.Array:
    """A boolean mask, one entry per row: do the two columns agree on this row?

    Null-vs-null counts as agreement and null-vs-value as disagreement, so the mask
    alone decides the column. Floats compare within tolerance (and ``NaN == NaN``);
    every other class compares exactly.

    Stays in Arrow kernels rather than dropping to NumPy: a ``large_string`` column
    converts to a NumPy array of Python objects, which at benchmark scale costs more
    memory than the table being compared.
    """
    both_null = pc.and_(ref.is_null(), oth.is_null())
    if cls == _FLOAT:
        tol = pc.add(max(FLOAT_ATOL, GRID_ATOL), pc.multiply(FLOAT_RTOL, pc.abs(oth)))
        close = pc.fill_null(pc.less_equal(pc.abs(pc.subtract(ref, oth)), tol), False)
        nan_eq = pc.fill_null(pc.and_(pc.is_nan(ref), pc.is_nan(oth)), False)
        close = pc.or_(close, nan_eq)
    else:
        # `equal` is null wherever either side is null; only null-vs-null is agreement.
        close = pc.fill_null(pc.equal(ref, oth), False)
    return pc.or_(close, both_null)


def _column_diff(name: str, cls: str, ref: pa.Array, oth: pa.Array) -> str | None:
    """The first disagreement in one sorted column, or ``None`` when they agree."""
    if len(ref) == 0:
        return None
    agree = _agree(cls, ref, oth)
    if pc.all(agree).as_py() is True:
        return None
    row = int(np.flatnonzero(~agree.to_numpy(zero_copy_only=False))[0])
    return f"column {name!r} row {row}: {ref[row].as_py()!r} vs {oth[row].as_py()!r}"


def rowsets_match(ref: RowSet, oth: RowSet) -> tuple[bool, str]:
    """Compare two canonicalized rowsets. Returns ``(ok, message)``.

    Args:
        ref: The reference engine's rowset.
        oth: The rowset under test.

    Returns:
        ``(True, "ok")`` when the two are equal as row multisets, else ``(False, why)``.
    """
    if ref.table.column_names != oth.table.column_names:
        return False, f"column mismatch: {ref.table.column_names} vs {oth.table.column_names}"
    if ref.table.num_rows != oth.table.num_rows:
        return False, f"row count: {ref.table.num_rows} vs {oth.table.num_rows}"

    for name in ref.table.column_names:
        diff = _column_diff(
            name,
            ref.classes[name],
            ref.table.column(name).combine_chunks(),
            oth.table.column(name).combine_chunks(),
        )
        if diff is not None:
            return False, diff
    return True, "ok"


def results_match(reference: pa.Table, other: pa.Table) -> tuple[bool, str]:
    """Compare two engines' results as sorted row multisets. Returns ``(ok, message)``.

    The standalone entry point, for callers holding exactly two tables. The benchmark
    runner instead reconciles classes across the whole lineup once
    (:func:`column_classes`) and compares the resulting rowsets pairwise.

    Args:
        reference: The oracle engine's result.
        other: The result under test.

    Returns:
        ``(True, "ok")`` when the two are equal as row multisets, else ``(False, why)``.
    """
    ref_names = sorted(canonical_names(reference))
    oth_names = sorted(canonical_names(other))
    if ref_names != oth_names:
        return False, (
            f"column mismatch: {sorted(reference.column_names)} vs {sorted(other.column_names)}"
        )
    classes = column_classes([reference, other])
    return rowsets_match(to_rowset(reference, classes), to_rowset(other, classes))


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


# The correctness oracle, in preference order. DuckDB is the project's designated oracle
# (`.claude/rules/testing.md`); Batcher — the system under test — must never be the
# reference, or a comparator's bug is reported as Batcher's. This is not hypothetical:
# on TPC-H Q6 both Polars and Daft drop the `l_discount = 0.07` rows, and with Batcher as
# the reference the row was blamed on Batcher even though it agreed with DuckDB and with
# the published TPC-H answer. Any engine that produced a result may serve as a last
# resort, so a run without DuckDB still cross-checks.
_ORACLE_PREFERENCE = ("duckdb", "polars", "spark", "daft", "pyarrow")


def _reference_engine(outputs: dict[str, pa.Table]) -> str:
    """The engine whose result the others are checked against."""
    for candidate in _ORACLE_PREFERENCE:
        if candidate in outputs:
            return candidate
    return next(iter(outputs))


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

    # Correctness: compare every produced output to a reference. Column types are
    # reconciled across the whole lineup first, so each output is canonicalized once
    # (not once per comparison) and every engine is held to the same comparison class.
    if outputs:
        ref_engine = _reference_engine(outputs)
        names = {engine: sorted(canonical_names(t)) for engine, t in outputs.items()}
        classes = column_classes(
            [t for engine, t in outputs.items() if names[engine] == names[ref_engine]]
        )
        mismatches = []
        ref_rows = to_rowset(outputs[ref_engine], classes)
        for engine, out in outputs.items():
            if names[engine] != names[ref_engine]:
                ok, msg = False, f"column mismatch: {names[ref_engine]} vs {names[engine]}"
            else:
                ok, msg = rowsets_match(ref_rows, to_rowset(out, classes))
            result.engines[engine].correct = ok
            if not ok:
                # `msg` comes from `rowsets_match(ref, other)` and reads "<ref> vs <other>",
                # so the names must lead with `ref_engine` too. Written the other way round
                # this line reported every mismatch with the two engines' values SWAPPED —
                # which is how "Daft computes q6 wrong" got recorded as Batcher's bug and back
                # again. A diff that names the wrong culprit is worse than no diff.
                mismatches.append(f"{ref_engine} != {engine}: {msg}")
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


# --------------------------------------------------------------------------- #
# Per-case process isolation, so a query that kills the process costs one row
# --------------------------------------------------------------------------- #
#
# ``compare()`` catches an exception per engine, so a query that *raises* is already one
# ``ERROR`` row in a table that still reports every other query. Nothing catches a signal.
# A query the OOM killer takes, or one that aborts inside a native kernel, ends the whole
# runner — and with it every result after it, including the ones already computed.
#
# That is not a hypothetical failure mode here. On the Join Order Benchmark a per-query
# survey found 24 of the first 85 queries dying by ``SIGKILL`` rather than raising, spread
# through the suite rather than clustered, so no ``--skip`` list makes a full run reachable
# and the suite reports nothing instead of the three quarters that work.
#
# `run_isolated` runs each case in its own subprocess. The child does exactly what the
# in-process loop would do for that one case and prints its ``CompareResult`` as JSON; the
# parent reads it back. A child that dies without printing one becomes a ``KILLED`` row
# carrying the signal that killed it, which is the same shape ``ERROR`` already has and
# reports the same fact the survey had to reconstruct by hand.
#
# Two properties are deliberate:
#
# **Isolation is per case, not per engine.** The comparison is the unit of meaning — a
# timing without the oracle's answer beside it is not a result — so a child runs the whole
# lineup for one query.
#
# **There is no timeout.** A wall clock cannot distinguish a hang from a query that is
# merely slow, and this suite has both: TPC-DS q72 legitimately takes ~30 s single-node and
# scale-factor runs take minutes. Marking a slow-but-correct query as failed would be a
# worse error than the one this module fixes, so a hang still needs ``--skip`` or a human.

#: Marks the one stdout line a child uses to hand its result back. A prefix rather than
#: "parse the last line" because the engines print freely and a native library may write
#: to the same stream after the result is known.
RESULT_PREFIX = "__BENCH_RESULT__ "


def emit_result(result: CompareResult) -> None:
    """Print `result` on the wire the parent reads. Called in the child."""
    payload = {
        "name": result.name,
        "status": result.status,
        "note": result.note,
        "engines": {
            name: {"ms": er.ms, "error": er.error, "correct": er.correct}
            for name, er in result.engines.items()
        },
    }
    print(RESULT_PREFIX + json.dumps(payload), flush=True)


def _parse_result(line: str) -> CompareResult:
    """Rebuild a `CompareResult` from the child's wire line."""
    payload = json.loads(line[len(RESULT_PREFIX) :])
    result = CompareResult(
        name=payload["name"], status=payload["status"], note=payload.get("note", "")
    )
    for name, er in payload.get("engines", {}).items():
        result.engines[name] = EngineResult(
            ms=er.get("ms"), error=er.get("error"), correct=er.get("correct")
        )
    return result


def _child_argv(case: str) -> list[str]:
    """This process's command line, aimed at exactly one case.

    Rebuilt from ``sys.argv`` rather than from the parsed namespace so every flag the
    parent was given — engines, scale, source, memory cap, spill dir — reaches the child
    without this module having to know the CLI. Only ``--isolate`` is dropped, or the
    child would recurse.
    """
    argv = [a for a in sys.argv[1:] if a != "--isolate"]
    return [sys.executable, sys.argv[0], *argv, "--isolate-case", case]


def _death(returncode: int) -> str:
    """Describe how a child that printed no result died."""
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:  # pragma: no cover - an unknown signal number
            name = f"signal {-returncode}"
        return f"killed by {name}"
    return f"exited {returncode} without a result"


def run_isolated(case_names: list[str]) -> list[CompareResult]:
    """Run each named case in its own subprocess and collect the results.

    A child that dies without printing a result yields a ``KILLED`` row rather than
    ending the run. The child's own output is forwarded on failure only, because that
    is where the traceback or the allocator's last words are, and forwarding it always
    would bury the table.

    Args:
        case_names: Case names to run, in report order. The caller has already applied
            ``--family`` / ``--only`` / ``--skip``, so every name here is meant to run.

    Returns:
        One result per name, in the same order.
    """
    results: list[CompareResult] = []
    for i, case in enumerate(case_names, start=1):
        print(f"[{i}/{len(case_names)}] {case} ...", flush=True)
        proc = subprocess.run(
            _child_argv(case),
            capture_output=True,
            text=True,
            check=False,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(RESULT_PREFIX)), None)
        if line is None:
            note = _death(proc.returncode)
            print(f"    {note}", flush=True)
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
            for entry in tail:
                print(f"    | {entry}", flush=True)
            results.append(CompareResult(name=case, status="KILLED", note=note))
            continue
        results.append(_parse_result(line))
    print()
    return results
