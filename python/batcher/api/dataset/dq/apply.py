"""Applying an accumulated `ds.dq` chain: lower it, count it, split it, annotate it.

The accessor is the surface; this is the machinery under all four terminal actions. It
exists as its own module for one structural reason: `drop`, `quarantine`, `annotate`, and
`validate` must agree exactly on what a violation is, and the only way to guarantee that is
for them to share one lowering (`prepared`) and one counting pass (`validate`). When they
did not, the report said a duplicated key was one violation while `drop` removed three rows.

Everything here lowers to relational operators that already exist — FILTER, a keyless
AGGREGATE, ``count() OVER (PARTITION BY keys)``, and a LEFT JOIN against distinct reference
keys. No new IR means no separate distributed semantics: the same chain runs on one core or
a hundred machines, and the split stays a provably total partition of the input.
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any

from batcher._internal import events
from batcher._internal.errors import PlanError
from batcher.api.dataset.dq.constraints import (
    AggregateConstraint,
    Constraint,
    ReferenceConstraint,
    RowConstraint,
    SchemaConstraint,
    UniqueConstraint,
)
from batcher.api.dataset.dq.report import ConstraintResult, ValidationReport
from batcher.plan.expr_ir import Col, Expr, count, lit, nullif, when
from batcher.plan.functions.string.building import concat_ws

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "ROW_KINDS",
    "annotate",
    "prepared",
    "provably_clean",
    "reject_unfilterable",
    "restore",
    "schema_gate",
    "validate",
    "validity",
]

ROW_KINDS = (RowConstraint, UniqueConstraint, ReferenceConstraint)
"""The constraint kinds that decide a *row*, and so can be dropped or quarantined."""


def reject_unfilterable(constraints: tuple[Constraint, ...], action: str) -> None:
    """Refuse a row-splitting action over a constraint that no row can violate.

    A relation-level constraint (a row count, a mean, a freshness bound) has no violating
    row to remove, so `drop`/`quarantine`/`annotate` cannot honour it. Silently ignoring it
    is the failure this exists to prevent: the chain reads as enforced, the terminal quietly
    enforces a subset of it, and the contract is weaker than the code says it is.

    Args:
        constraints: The accumulated chain.
        action: The terminal action's name, for the message.

    Raises:
        PlanError: If any relation-level constraint is present.
    """
    offenders = [c.name for c in constraints if isinstance(c, AggregateConstraint)]
    if offenders:
        raise PlanError(
            f"{action}() cannot act on the relation-level constraint(s) "
            f"{', '.join(offenders)}: they are true or false of the whole table, so there is "
            "no violating row to remove. Check them with .validate() or .fail(), and keep "
            "the row-level constraints in the chain you drop or quarantine on."
        )


def schema_gate(constraints: tuple[Constraint, ...], action: str) -> None:
    """Raise if an enforced schema constraint is unsatisfied — before any row work.

    A missing column cannot be quarantined; a row-level chain written against it would fail
    for every row, and report the wrong cause. So the schema verdict is a hard gate at the
    terminal rather than a per-row concern.
    """
    from batcher._internal.errors import DataQualityError

    broken = [c for c in constraints if isinstance(c, SchemaConstraint) and not c.satisfied]
    blocking = [c for c in broken if c.enforced]
    if blocking:
        detail = "; ".join(f"{c.name}: {c.detail}" if c.detail else c.name for c in blocking)
        raise DataQualityError(
            f"{action}() cannot run: the schema contract is not met — {detail}.",
            violations={c.name: 1 for c in blocking},
        )


def prepared(
    ds: Dataset, constraints: tuple[Constraint, ...], *, enforced_only: bool = True
) -> tuple[Dataset, list[tuple[str, Expr]]]:
    """Lower every row-deciding constraint onto `ds`, one validity expression each.

    Uniqueness and referential integrity are not row expressions on their own — one needs a
    window count over the key partition, the other a join against the reference — so this
    adds the columns they need and hands back the names to drop afterwards.

    Args:
        ds: The dataset the chain was built against.
        constraints: The accumulated chain.
        enforced_only: Whether to skip `warn`-severity constraints, as the row-splitting
            actions do; `annotate` passes False so a warning still shows up per row.

    The helper columns it adds are not handed back, because naming them is not how they get
    removed: `restore` projects the result to the input's own columns in the input's own
    order, which drops them and undoes the reordering a join causes in the same step.

    Returns:
        The prepared dataset and the ``(name, validity)`` pairs in declaration order.
    """
    terms: list[tuple[str, Expr]] = []
    for i, c in enumerate(constraints):
        if not isinstance(c, ROW_KINDS) or (enforced_only and not c.enforced):
            continue
        if isinstance(c, RowConstraint):
            terms.append((c.name, c.valid))
        elif isinstance(c, UniqueConstraint):
            ds, valid = _unique_validity(ds, c, i)
            terms.append((c.name, valid))
        else:
            ds, valid = _reference_validity(ds, c, i)
            terms.append((c.name, valid))
    return ds, terms


def _unique_validity(ds: Dataset, c: UniqueConstraint, index: int) -> tuple[Dataset, Expr]:
    """A row is valid iff its key combination occurs exactly once in the relation."""
    if "__dq_one" not in ds.schema.names:
        # A constant non-null column to COUNT over each key partition: COUNT(1)
        # OVER (PARTITION BY keys) is the per-key row count (==1 iff unique).
        ds = ds.with_columns(__dq_one=lit(1))
    h = f"__dq_uniq_{index}"
    ds = ds.window(partition_by=list(c.keys), order_by=[], functions={h: ("count", "__dq_one")})
    return ds, Col(h) == 1


def _reference_validity(ds: Dataset, c: ReferenceConstraint, index: int) -> tuple[Dataset, Expr]:
    """A row is valid iff any key part is NULL, or the key resolves in the reference.

    The reference keys are made distinct before the join, which is what keeps this a
    row-preserving LEFT JOIN: a duplicated key on the reference side would otherwise
    multiply the rows being checked.
    """
    prefix = f"__dq_ref{index}_"
    marker = f"{prefix}hit"
    right_on = [f"{prefix}k{i}" for i in range(len(c.ref_columns))]
    ref = c.reference_keys(prefix).with_columns(**{marker: lit(True)})
    joined = ds.join(ref, left_on=list(c.columns), right_on=right_on, how="left")
    any_null = reduce(lambda a, b: a | b, (Col(k).is_null() for k in c.columns))
    return joined, any_null | Col(marker).is_not_null()


def restore(frame: Dataset, columns: list[str], *, extra: str | None = None) -> Dataset:
    """Project `frame` back to the input's columns, in the input's order.

    Projecting by name rather than dropping the helpers by name, because dropping them is
    only half the job. A uniqueness check adds a window column and a referential one adds a
    join, and **a join reorders**: the key it joined on comes back first, so
    ``ds.dq.references("ref", to=...).drop()`` returned every right row under a schema whose
    columns had moved. Nothing raises on that — it is the same names and the same types in a
    different order — and it reaches a positional consumer (a `to_parquet` appending to an
    existing table, a downstream `union`) as silently wrong data.

    Args:
        frame: The lowered dataset, carrying whatever helper columns the chain needed.
        columns: The input relation's column names, in its own order.
        extra: A column to append after them, for `annotate`.

    Returns:
        The dataset projected to `columns` (plus `extra`).
    """
    wanted = [*columns, extra] if extra else list(columns)
    if list(frame.schema.names) == wanted:
        return frame  # nothing was added, so do not add a projection either
    return frame.select(*wanted)


def validity(terms: list[tuple[str, Expr]]) -> Expr:
    """One total boolean: TRUE exactly for a row that satisfies every lowered constraint.

    Forced to non-null so that ``valid`` and ``NOT valid`` partition the input exactly — a
    user `check()` may evaluate to NULL, and a NULL validity is a violation, not a third
    outcome that lands in neither side of a quarantine.

    Args:
        terms: The ``(name, validity)`` pairs from `prepared`.

    Returns:
        The conjunction of every validity, or a TRUE literal when there are none.
    """
    if not terms:
        return lit(True)
    combined = reduce(lambda a, b: a & b, (expr for _name, expr in terms))
    return when(combined).then(lit(True)).otherwise(lit(False))


def validate(ds: Dataset, constraints: tuple[Constraint, ...]) -> ValidationReport:
    """Execute the chain and return one `ConstraintResult` per constraint, in order.

    Row-level counts and every relation-level aggregate are measured in a **single keyless
    aggregate**, because they are all reductions over the same relation and there is no
    reason to scan it twice. Uniqueness and referential integrity need their own shuffle
    each and take one extra pass apiece.

    Args:
        ds: The dataset the chain was built against.
        constraints: The accumulated chain.

    Returns:
        The `ValidationReport`.
    """
    runtime = tuple(c for c in constraints if not isinstance(c, SchemaConstraint))
    measured: dict[int, Any] = {}
    rows = 0
    if runtime and not provably_clean(ds, constraints):
        rows, measured = _measure(ds, constraints)
    report = ValidationReport(
        tuple(_result(c, i, measured, rows) for i, c in enumerate(constraints))
    )
    _publish(report)
    return report


def _publish(report: ValidationReport) -> None:
    """Put each constraint's result on the event bus, so `observe` can chart the contract.

    A report answers "is today's data good". It cannot answer "has this constraint's
    violation count been climbing all week", which is the question that catches an upstream
    change while it is still a warning. Publishing gives that series to every sink already
    listening, at the cost of one dict per constraint on a path that just ran an aggregate.

    Passing constraints are published too: a series that appears only when something breaks
    has no baseline to compare against.
    """
    if not events.listening():
        return
    for r in report.results:
        events.publish(
            events.DQ,
            name=r.name,
            constraint=r.name,
            check=r.kind,
            severity=r.severity,
            violations=r.violations,
            rows=r.rows,
            ok=r.ok,
            value=r.value,
        )


def provably_clean(ds: Dataset, constraints: tuple[Constraint, ...]) -> bool:
    """Whether metadata proves every constraint holds — see `api.dataset.meta.prove`.

    A schema constraint is already decided, so it is answered here rather than handed to a
    prover that would see an unrecognized kind and conservatively give up — which would have
    made adding one schema check to a chain silently disable the metadata shortcut for the
    whole chain.

    Args:
        ds: The dataset the chain was built against.
        constraints: The accumulated chain.

    Returns:
        True when nothing can violate the contract, so no execution is needed.
    """
    from batcher.api.dataset.meta.prove import constraints_provably_hold

    runtime: list[Constraint] = []
    for c in constraints:
        if isinstance(c, SchemaConstraint):
            if not c.satisfied:
                return False
        else:
            runtime.append(c)
    return constraints_provably_hold(ds, runtime)


def _measure(ds: Dataset, constraints: tuple[Constraint, ...]) -> tuple[int, dict[int, Any]]:
    """Run the counting passes, returning the row count and each constraint's measurement."""
    aggs: dict[str, Expr] = {"__dq_rows": count()}
    for i, c in enumerate(constraints):
        if isinstance(c, RowConstraint):
            aggs[f"__dq_v{i}"] = when(c.valid).then(lit(0)).otherwise(lit(1)).sum()
        elif isinstance(c, AggregateConstraint):
            aggs[f"__dq_a{i}"] = c.value
    row = ds.agg(**aggs).to_pydict()
    rows = int(row["__dq_rows"][0] or 0)
    measured: dict[int, Any] = {}
    for i, c in enumerate(constraints):
        if isinstance(c, RowConstraint):
            measured[i] = int(row[f"__dq_v{i}"][0] or 0)
        elif isinstance(c, AggregateConstraint):
            measured[i] = row[f"__dq_a{i}"][0]
        elif isinstance(c, UniqueConstraint):
            measured[i] = _duplicate_row_count(ds, c)
        elif isinstance(c, ReferenceConstraint):
            measured[i] = _orphan_count(ds, c)
    return rows, measured


def _result(c: Constraint, index: int, measured: dict[int, Any], rows: int) -> ConstraintResult:
    """One constraint's result, from what the measuring pass found for it."""
    if isinstance(c, SchemaConstraint):
        return ConstraintResult(
            c.name, 0 if c.satisfied else 1, severity=c.severity, kind="schema", detail=c.detail
        )
    if isinstance(c, AggregateConstraint):
        value = measured.get(index)
        held = c.holds(None if value is None else float(value))
        return ConstraintResult(
            c.name,
            0 if held else 1,
            severity=c.severity,
            kind="aggregate",
            value=None if value is None else float(value),
        )
    kind = {UniqueConstraint: "unique", ReferenceConstraint: "reference"}.get(type(c), "row")
    return ConstraintResult(
        c.name,
        int(measured.get(index, 0)),
        rows=rows,
        severity=c.severity,
        mostly=c.mostly,
        kind=kind,
    )


def _duplicate_row_count(ds: Dataset, unique: UniqueConstraint) -> int:
    """How many *rows* the uniqueness constraint rejects — not how many keys repeat.

    `ValidationReport.total_violations` is documented as the number of violating rows, and
    every row-wise constraint reports one. This counted the duplicated *groups* instead,
    so it under-reported by the size of each group and by an unbounded factor: over
    ``[1, 1, 1, 2]``, `drop` removes three rows and `quarantine` rejects three, while the
    report said `1`. A key repeated a thousand times still said `1`.

    Summing the group sizes matches what `drop`/`quarantine` do — they keep a row iff
    ``count() OVER (PARTITION BY keys) == 1`` — so the non-raising report and the
    non-raising split now agree, which is what anyone reading a monitoring dashboard next
    to a dead-letter sink is entitled to assume.

    Args:
        ds: The dataset the chain was built against.
        unique: The uniqueness constraint to measure.

    Returns:
        The number of rows whose key combination occurs more than once.
    """
    duplicated = (
        ds.group_by(*unique.keys)
        .agg(__dq_n=count())
        .filter(Col("__dq_n") > 1)
        .agg(__dq_rows=Col("__dq_n").sum())
        .to_pydict()["__dq_rows"]
    )
    # `sum` over no group is NULL, which is zero violating rows.
    return int(duplicated[0]) if duplicated and duplicated[0] is not None else 0


def _orphan_count(ds: Dataset, c: ReferenceConstraint) -> int:
    """How many rows carry a non-null key with no match in the reference relation."""
    declared = reduce(lambda a, b: a & b, (Col(k).is_not_null() for k in c.columns))
    right_on = [f"__dq_ref_k{i}" for i in range(len(c.ref_columns))]
    ref = c.reference_keys("__dq_ref_")
    checked = ds.filter(declared)
    return checked.join(ref, left_on=list(c.columns), right_on=right_on, how="anti").count()


def annotate(ds: Dataset, constraints: tuple[Constraint, ...], column: str) -> Dataset:
    """Add `column`, naming the constraints each row failed — empty text when it passed.

    The column a dead-letter sink actually needs. A quarantined row on its own says only
    that *something* rejected it, which turns triage into re-running the checks one at a
    time; carrying the verdict with the row makes "which rule, how often, since when" a
    `group_by` rather than an investigation.

    Warn-severity constraints are named too: their whole purpose is to be observed before
    they are enforced.

    Args:
        ds: The dataset the chain was built against.
        constraints: The accumulated chain.
        column: The name of the column to add.

    Returns:
        A lazy `Dataset` with the extra text column.
    """
    reject_unfilterable(constraints, "annotate")
    schema_gate(constraints, "annotate")
    columns = list(ds.schema.names)
    frame, terms = prepared(ds, constraints, enforced_only=False)
    if not terms:
        return frame.with_columns(**{column: lit("")})
    # `concat_ws` skips NULL arguments, so a passing constraint contributes nothing and no
    # separator is doubled. A NULL validity (only a user `check` can produce one) takes the
    # `otherwise` branch and is named, matching the rule that a NULL validity is a violation.
    #
    # `nullif(x, x)` is how a typed NULL is spelled — `lit(None)` has no type to lower to, and
    # `api.multi_group` reaches for the same idiom for the same reason.
    empty = lit("")
    parts = [when(expr).then(nullif(empty, empty)).otherwise(lit(name)) for name, expr in terms]
    out = frame.with_columns(**{column: concat_ws(",", *parts)})
    return restore(out, columns, extra=column)
