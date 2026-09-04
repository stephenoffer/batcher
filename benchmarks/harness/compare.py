"""Result canonicalization, the equality gate, and the timed comparison itself.

Two engines' answers to one query are compared as **row multisets**: column names are
canonicalized (engines invent different names for a derived column), types are reconciled
across the whole lineup, floats are rounded onto a grid, and both sides are sorted. That
makes the comparison independent of the row order an engine happens to produce — which is
correct for a query that asked for no order, and is exactly why `order` exists for the ones
that did.

Correctness is checked *before* timings are trusted: if the engines disagree the row is
marked ``FAILED`` and a short diff is printed, but the suite continues. The comparison is
vectorized over Arrow, because a row-wise one costs more than the queries it is gating.
"""

from __future__ import annotations

import math
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .divergences import explain
from .names import canonical_names
from .order import order_violation

# Absolute / relative tolerance for the pairwise float comparison.
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-9

# Floats are rounded to this many decimals *before* sorting, so two values that are
# equal within tolerance land on the same grid point and therefore sort together.
ROUND_DECIMALS = 6
# Rounding maps two values that agree to ~1e-9 *in absolute terms* onto adjacent grid
# points at worst, leaving them one step (1e-6) apart, and the pairwise tolerance must
# clear one step or that boundary is a false mismatch.
#
# That reasoning holds at small magnitudes and NOT at large ones, which is worth stating
# rather than leaving to be rediscovered: a TPC-H revenue sum is ~1e9, where agreeing to
# 1e-9 *relative* means differing by about one whole unit — roughly a million grid steps.
# What actually carries the comparison there is `FLOAT_RTOL * |x|` in `_agree`, not this
# constant. The residual exposure is that `to_rowset` sorts on every column, so two rows
# whose float keys fall within that difference could sort differently on the two sides and
# be compared against each other. It needs near-duplicate aggregates to bite and none of
# these queries produce them, but it is a property of the design rather than a guarantee.
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
# `duckdb_arrow` sits beside `duckdb`: it is the same engine on a different binding, so it
# is exactly as good an oracle. Leaving it out was a quiet hole — `--engines
# batcher,duckdb_arrow` is the *documented* like-for-like invocation, and with no listed
# engine present the fallback below picked the only remaining output, which is Batcher. The
# system under test became its own oracle on a lineup the README tells people to run. `ray`
# is last because it has no SQL surface and answers only the native cases, not because its
# answers are suspect.
#
# This list is hand-maintained while the adapters self-register, so an engine added to
# `engines/lineup.py` and not added here silently becomes ineligible. That is what happened.
_ORACLE_PREFERENCE = ("duckdb", "duckdb_arrow", "polars", "spark", "daft", "pyarrow", "ray")


def degenerate_reason(table: pa.Table) -> str | None:
    """Why `table` carries no information, or ``None`` when it carries some.

    The equality gate is a comparison, and a comparison between two empty answers succeeds.
    `rowsets_match` checks `num_rows` first, so `0 == 0` matches; `_column_diff` then returns
    `None` for a zero-length column. Both are individually right and compose into **"the
    engines agree because none of them did anything"** — reported `OK`, and timed, at the
    fastest speed any query can be answered.

    Four shapes reach it and all four passed: zero rows, zero *columns*, every value null,
    and a single null row. Zero columns is the sharpest — `pa.table({})` against
    `pa.table({})` is a result that is broken under any reading, and also the cheapest
    possible one to produce. The failure modes that land here are the ordinary ones: a wrong
    path, an empty glob, a truncated write, a data load that silently produced nothing.
    """
    if table.num_columns == 0:
        return "zero columns"
    if table.num_rows == 0:
        return "zero rows"
    if all(table.column(c).null_count == table.num_rows for c in table.column_names):
        return "every value null"
    return None


def _reference_engine(outputs: dict[str, pa.Table]) -> str:
    """The engine whose result the others are checked against.

    Falls back to whatever produced a result when no preferred oracle ran — and **says so**,
    because the fallback's most likely value is Batcher itself and a silent self-check reads
    exactly like a passing one. A lineup with no independent engine is a real state
    (`--engines batcher` alone); it is not a correctness gate, and the run should not look
    like it was.
    """
    for candidate in _ORACLE_PREFERENCE:
        if candidate in outputs:
            return candidate
    reference = next(iter(outputs))
    print(
        f"  !! no independent oracle in this lineup; checking against {reference!r}, "
        "which is the engine under test — these rows are not correctness-gated"
    )
    return reference


def _release_engine(engine: str, lineup: list[str]) -> None:
    """Let `engine` give up cluster-wide resources before the next one is timed.

    Only when it shares the lineup: a single-engine run should keep whatever warm state it
    would have in production. See `engines.base.Engine.release` for what this is for — a
    Batcher session fleet reserving the cluster left Daft unable to start a worker.

    Best-effort in every direction. This is teardown between measurements; nothing here is
    worth failing a benchmark row for.
    """
    if len(lineup) < 2:
        return
    try:
        import engines as engines_mod

        engines_mod.get(engine).release()
    except Exception:  # pragma: no cover - teardown must never fail a case
        pass


def compare(
    name: str,
    fns: dict[str, Callable[[], pa.Table] | None],
    engines: list[str],
    runs: int = 5,
    ordered_by: Sequence[tuple[str | int, bool]] = (),
) -> CompareResult:
    """Run each engine's query, verify equality, and record timings.

    ``fns`` maps engine name -> callable returning a ``pyarrow.Table`` (or
    ``None`` to mark the case "n/a" for that engine). ``engines`` is the resolved
    lineup (and report order). Correctness is checked against the first engine that
    produced a result.

    ``ordered_by`` are the query's outermost ``ORDER BY`` terms. The equality check below
    compares row *multisets* — it sorts both sides — so on its own it cannot tell a sorted
    result from an unsorted one, and an engine that skipped the sort entirely would be
    reported as correct and then timed on the work it did not do. Each result is therefore
    additionally checked for monotonicity in its own order; see ``sort_order``.
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
            _release_engine(engine, engines)
            continue
        result.engines[engine] = er
        _release_engine(engine, engines)

    # Correctness: compare every produced output to a reference. Column types are
    # reconciled across the whole lineup first, so each output is canonicalized once
    # (not once per comparison) and every engine is held to the same comparison class.
    if outputs:
        ref_engine = _reference_engine(outputs)
        names = {engine: sorted(canonical_names(t)) for engine, t in outputs.items()}
        classes = column_classes(
            [t for engine, t in outputs.items() if names[engine] == names[ref_engine]]
        )
        mismatches: list[str] = []
        divergent: list[str] = []
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
                known = explain(name, ref_engine, engine, msg)
                if known is not None:
                    # A *recorded* semantic difference, not a failure. The row still does
                    # not read OK and its ratio is still withheld — see `divergences`.
                    divergent.append(
                        f"{ref_engine} != {engine}: {msg} [known: {known.reason} "
                        f"({known.citation})]"
                    )
                else:
                    mismatches.append(f"{ref_engine} != {engine}: {msg}")
        # Order, per engine and against the query rather than against another engine: the
        # multiset comparison above sorted both sides, so this is the only thing standing
        # between a skipped `ORDER BY` and a timed win on it.
        for engine, out in outputs.items():
            violation = order_violation(out, list(ordered_by))
            if violation is not None:
                result.engines[engine].correct = False
                mismatches.append(f"{engine}: {violation}")
        # Agreement between engines that all returned nothing is not agreement. It is not
        # a *defect* either — TPC-DS q17 legitimately returns zero rows at sf1, measured —
        # so this is its own status rather than a failure: never `OK`, no ratio, reason
        # printed, and the run does not fail on it. A reader sees "every engine returned
        # nothing" where they would otherwise have seen a fast pass.
        degenerate = {e: degenerate_reason(o) for e, o in outputs.items()}
        if not mismatches and all(degenerate.values()):
            kinds = sorted(set(degenerate.values()))
            result.status = "DEGENERATE"
            result.note = (
                f"every engine returned a result carrying no information ({', '.join(kinds)}) "
                "— they agree, but nothing was compared"
            )
        if mismatches:
            result.status = "FAILED"
            result.note = " ; ".join(mismatches + divergent)
        elif divergent:
            # Every difference on this row is one the suite has recorded, with a citation
            # naming which engine is right. That is not agreement and it is not a defect:
            # `DIVERGENT` says so, keeps the row out of `OK`, and keeps the ratio withheld.
            result.status = "DIVERGENT"
            result.note = " ; ".join(divergent)
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
        _release_engine(engine, engines)

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
