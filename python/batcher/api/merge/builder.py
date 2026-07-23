"""`MergeBuilder` — the fluent ``MERGE INTO`` surface, and the legacy keyword shorthand.

``ds.write.merge_into(target, on=…)`` opens a builder; each ``when_*`` names a population
and returns the actions legal for it, so a clause that cannot exist cannot be spelled (an
insert has no ``delete()``; a not-matched-by-source clause has no ``insert()``). Clauses
apply **in the order they are added**, first match wins — SQL's rule.

    updates.write.merge_into("warehouse/orders", on="id") \\
        .when_matched(source_col("op") == lit("D")).delete() \\
        .when_matched().update(amount=source_col("amount")) \\
        .when_not_matched().insert_all() \\
        .when_not_matched_by_source().update(is_current=lit(False)) \\
        .execute()

`Writer.merge`'s keyword form (``when_matched="update"``) is the two-clause special case of
exactly this, translated by `simple_clauses`, so there is one merge implementation and not
two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError, suggestion
from batcher.api.merge.clauses import (
    CLAUSE_LABEL,
    CLAUSE_METHODS,
    MATCHED,
    NOT_MATCHED,
    NOT_MATCHED_BY_SOURCE,
    MergeClause,
    legal_actions,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest
    from batcher.plan.expr_ir import Expr

__all__ = ["MergeBuilder", "MergeWhen", "simple_clauses"]

_WHEN_MATCHED = ("update", "delete")
_WHEN_NOT_MATCHED = ("insert", "ignore")


def simple_clauses(when_matched: str, when_not_matched: str) -> list[MergeClause]:
    """The clause list for the keyword shorthand — a plain keyed upsert.

    ``when_matched="update"`` is ``UPDATE SET *``; ``when_not_matched="insert"`` is
    ``INSERT *``; ``"delete"``/``"ignore"`` drop the respective clause's row. Nothing here
    touches the not-matched-by-source population, so target rows the source never mentions
    survive untouched — which is what an upsert means.
    """
    if when_matched not in _WHEN_MATCHED:
        hint = suggestion(when_matched, _WHEN_MATCHED)
        raise PlanError(
            f"merge(): when_matched must be one of {_WHEN_MATCHED}, got {when_matched!r}."
            + (f" {hint}" if hint else "")
        )
    if when_not_matched not in _WHEN_NOT_MATCHED:
        hint = suggestion(when_not_matched, _WHEN_NOT_MATCHED)
        raise PlanError(
            f"merge(): when_not_matched must be one of {_WHEN_NOT_MATCHED}, "
            f"got {when_not_matched!r}." + (f" {hint}" if hint else "")
        )
    clauses = [MergeClause(kind=MATCHED, action=when_matched)]
    if when_not_matched == "insert":
        clauses.append(MergeClause(kind=NOT_MATCHED, action="insert"))
    return clauses


def _columns(
    action: str,
    values: Mapping[str, Expr] | None,
    named: dict[str, Expr],
    *,
    star: str,
) -> dict[str, Expr]:
    """The columns a clause writes, from either spelling — and never an empty set.

    An empty ``update()`` is almost always a mistake, and it is an expensive one: it reads
    as "write nothing", but SQL's ``UPDATE SET *`` (write *everything*) is one keystroke
    away. Refusing it and naming `star` is how a slip becomes an error instead of a silently
    empty clause.
    """
    merged = {**(values or {}), **named}
    if not merged:
        raise PlanError(
            f"merge(): {action}() needs at least one column — pass them as keywords "
            f"(e.g. {action}(amount=source_col('amount'))) or as a mapping. "
            f"To write every column from the source, use {star}()."
        )
    overlap = sorted(set(values or {}) & set(named))
    if overlap:
        raise PlanError(f"merge(): {action}() got column(s) {overlap} both ways; pick one")
    return merged


class MergeWhen:
    """The actions legal for one merge population — returned by a `MergeBuilder.when_*`.

    Each action records a clause and hands the `MergeBuilder` back, so clauses chain. This
    exists so the type of the population decides which actions are even callable, rather
    than a runtime check on a string.
    """

    __slots__ = ("_builder", "_condition", "_kind")

    def __init__(self, builder: MergeBuilder, kind: str, condition: Expr | None) -> None:
        self._builder = builder
        self._kind = kind
        self._condition = condition

    def update(self, values: Mapping[str, Expr] | None = None, **named: Expr) -> MergeBuilder:
        """Set the named target columns; columns not named keep their current value.

        Write the columns as keywords (``update(amount=source_col("amount"))``), or as a
        mapping when a column name is not a Python identifier — the same two spellings
        `Dataset.rename` accepts, for the same reason.

        Args:
            values: Target column name → the expression producing its new value.
            named: The same, as keyword arguments.

        Returns:
            The builder, so clauses chain.
        """
        return self._add("update", _columns("update", values, named, star="update_all"))

    def update_all(self) -> MergeBuilder:
        """Replace every target column with the source column of the same name (``UPDATE SET *``).

        Returns:
            The builder, so clauses chain.
        """
        return self._add("update", None)

    def delete(self) -> MergeBuilder:
        """Remove the matched row from the target.

        Returns:
            The builder, so clauses chain.
        """
        return self._add("delete", None)

    def insert(self, values: Mapping[str, Expr] | None = None, **named: Expr) -> MergeBuilder:
        """Insert a row, writing the named target columns; unnamed columns become NULL.

        Args:
            values: Target column name → the expression producing its value.
            named: The same, as keyword arguments.

        Returns:
            The builder, so clauses chain.
        """
        return self._add("insert", _columns("insert", values, named, star="insert_all"))

    def insert_all(self) -> MergeBuilder:
        """Insert the source row as-is, column for column by name (``INSERT *``).

        Returns:
            The builder, so clauses chain.
        """
        return self._add("insert", None)

    # Delta/Spark spellings (`whenMatchedUpdateAll` / `whenNotMatchedInsertAll`), so a
    # merge ported from Delta keeps reading. Thin aliases of the snake_case actions.
    def updateAll(self) -> MergeBuilder:
        """Delta spelling of `update_all` (``UPDATE SET *``).

        Returns:
            The builder, so clauses chain.
        """
        return self.update_all()

    def insertAll(self) -> MergeBuilder:
        """Delta spelling of `insert_all` (``INSERT *``).

        Returns:
            The builder, so clauses chain.
        """
        return self.insert_all()

    def __repr__(self) -> str:
        """Show which population this clause targets."""
        guard = "" if self._condition is None else " (guarded)"
        return f"MergeWhen({CLAUSE_LABEL.get(self._kind, self._kind)}{guard})"

    def _add(self, action: str, values: dict[str, Expr] | None) -> MergeBuilder:
        if action not in legal_actions(self._kind):
            raise PlanError(
                f"merge(): a {CLAUSE_LABEL.get(self._kind, self._kind)} clause cannot "
                f"{action}(); use {CLAUSE_METHODS.get(self._kind, '?')}"
            )
        clause = MergeClause(
            kind=self._kind, action=action, condition=self._condition, values=values
        )
        return self._builder._append(clause)


class MergeBuilder:
    """A ``MERGE INTO`` under construction — populations, ordered clauses, then `execute`.

    Obtained from ``ds.write.merge_into(target, on=…)``, where ``ds`` is the **source**
    (the change set) and `target` is the table being written.

    Examples:
        .. doctest::

            >>> import tempfile, os
            >>> import batcher as bt
            >>> from batcher import lit, source_col
            >>> path = os.path.join(tempfile.mkdtemp(), "t.parquet")
            >>> _ = bt.from_pydict({"id": [1, 2], "v": [10, 20]}).write.parquet(path)
            >>> changes = bt.from_pydict({"id": [2, 3], "v": [99, 30]})
            >>> _ = (
            ...     changes.write.merge_into(path, on="id")
            ...     .when_matched()
            ...     .update({"v": source_col("v")})
            ...     .when_not_matched()
            ...     .insert_all()
            ...     .execute()
            ... )
            >>> sorted(zip(*bt.read.parquet(path).collect().to_pydict().values()))
            [(1, 10), (2, 99), (3, 30)]
    """

    __slots__ = ("_clauses", "_format", "_keys", "_opts", "_prune", "_source", "_target")

    def __init__(
        self,
        source: Dataset,
        target: str,
        keys: list[str],
        *,
        prune: bool = True,
        format: str | None = None,
        opts: dict[str, Any] | None = None,
    ) -> None:
        if not keys:
            raise PlanError(
                "merge_into(): needs at least one key column to match rows on — "
                "pass on='id' (or on=['a', 'b'] for a composite key)"
            )
        self._source = source
        self._target = target
        self._keys = keys
        self._clauses: list[MergeClause] = []
        self._prune = prune
        self._format = format
        self._opts = opts or {}

    def __repr__(self) -> str:
        """Show the target, the merge keys, and the clauses added so far."""
        if not self._clauses:
            clauses = "no clauses yet"
        else:
            clauses = ", ".join(f"{c.kind}:{c.action}" for c in self._clauses)
        return f"MergeBuilder(target={self._target!r}, on={self._keys}, [{clauses}])"

    def when_matched(self, condition: Expr | None = None) -> MergeWhen:
        """Clause for rows whose key exists in **both** sides (``WHEN MATCHED``).

        Args:
            condition: An optional guard (SQL's ``AND …``); the clause applies only where
                it is true. It may read both `source_col` and `target_col`.

        Returns:
            The actions legal here: `update`, `update_all`, `delete`.
        """
        return MergeWhen(self, MATCHED, condition)

    def when_not_matched(self, condition: Expr | None = None) -> MergeWhen:
        """Clause for source keys absent from the target (``WHEN NOT MATCHED``).

        Args:
            condition: An optional guard. There is no target row here, so it may read
                `source_col` only.

        Returns:
            The actions legal here: `insert`, `insert_all`.
        """
        return MergeWhen(self, NOT_MATCHED, condition)

    def when_not_matched_by_source(self, condition: Expr | None = None) -> MergeWhen:
        """Clause for target keys absent from the source (``WHEN NOT MATCHED BY SOURCE``).

        This is the population a plain upsert ignores: rows the change set never mentions.
        Acting on them is how a full-snapshot load expires departed rows, how SCD-2 closes
        a version, and how a soft delete is applied.

        Note that a merge carrying such a clause **cannot skip any target file** — every
        row of the table is a candidate, so the whole table is rewritten. That is inherent
        to the clause (Delta and Snowflake pay the same cost), not a limitation here.

        Args:
            condition: An optional guard. There is no source row here, so it may read
                `target_col` only.

        Returns:
            The actions legal here: `update`, `update_all`, `delete`.
        """
        return MergeWhen(self, NOT_MATCHED_BY_SOURCE, condition)

    def execute(self) -> WriteManifest:
        """Run the merge and commit it.

        Returns:
            A `WriteManifest` of the data files the merge wrote.

        Raises:
            PlanError: If no ``when_*`` clause was added — a clauseless merge would
                touch nothing, which is almost always a forgotten clause.
        """
        if not self._clauses:
            raise PlanError(
                "merge_into(): no clauses were added, so the merge would do nothing. Add "
                "at least one, e.g. .when_matched().update_all().when_not_matched().insert_all()"
            )
        from batcher.api.merge.execute import run_merge

        return run_merge(
            self._source,
            self._target,
            self._keys,
            self._clauses,
            prune=self._prune,
            format=self._format,
            opts=self._opts,
        )

    def _append(self, clause: MergeClause) -> MergeBuilder:
        self._clauses.append(clause)
        return self

    @property
    def clauses(self) -> list[MergeClause]:
        """The clauses added so far, in the order they will be tried."""
        return list(self._clauses)
