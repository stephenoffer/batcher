"""The scaffolding every `ds.meta` accessor stands on: get the facts, or execute instead.

Two rules, held in one place so a hundred shortcuts cannot each get them slightly wrong.

**A shortcut never breaks a query.** Metadata analysis is an optimisation, so every failure
mode of it — a streaming source with no finite answer, a `map_batches` pipeline the IR cannot
describe, a connector that raises while reading its own footer — must degrade to "no facts",
never to an exception. `facts()` returns None there and the caller executes.

**A shortcut never changes an answer.** `answer(value, fallback)` is the only way a shortcut
reaches the user: a non-None metadata value is returned as-is (it is `Provenance.EXACT`-gated,
so it *is* the executed answer), and a None runs `fallback`, which is the same query the
engine would have run anyway. That is what makes `ds.meta.col("x").is_unique()` safe to
believe: it is either a footer read or a `COUNT(DISTINCT)`, and it cannot tell you which.

Layer: `api`, the conductor — the one layer allowed to ask Kyber a question *and* run the
executor when Kyber declines.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from batcher._internal.errors import PlanError
from batcher._internal.logging import note_suppressed
from batcher.plan.schema import suggest_columns

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.kyber.shortcuts import Facts

T = TypeVar("T")

__all__ = ["MetaBase", "answer"]


def answer(value: T | None, fallback: Callable[[], T]) -> T:
    """Return the metadata answer, or run `fallback` when metadata could not prove one.

    The single seam between the fast path and the correct path. `value` is `None` exactly
    when Kyber declined, and declining is always allowed — so `fallback` must be a query that
    computes the *same* thing, not a cheaper approximation of it.
    """
    return fallback() if value is None else value


class MetaBase:
    """Shared plumbing for the metadata accessors: the dataset, and its cached facts.

    The cache is what makes a wide namespace cheap. Estimating a plan means rewriting it
    through the optimizer and propagating statistics — a few milliseconds, which is nothing
    against a scan and everything against a field read. So it happens once per accessor tree,
    and the thirty questions a caller then asks are thirty dictionary lookups.

    Keyed by the *enrichment* requested, because some facets (an exact distinct count, an
    exact sum) are computed lazily by the source on demand: asking for `n_unique` pays for a
    distinct pass, and asking for `min` must not.
    """

    __slots__ = ("_cache", "_ds")

    def __init__(self, ds: Dataset) -> None:
        """Bind to a dataset; prefer reaching this through `Dataset.meta`."""
        self._ds = ds
        self._cache: dict[tuple, Facts | None] = {}

    def facts(
        self,
        *,
        ndv: Iterable[str] = (),
        mean: Iterable[str] = (),
        total: Iterable[str] = (),
    ) -> Facts | None:
        """This dataset's provable facts, or None when metadata analysis cannot run.

        The three keyword arguments name the columns whose *lazily computed* exact facets are
        wanted (distinct count, average, sum). They are opt-in because each costs a pass over
        an in-memory column the first time it is asked for — cheap, cached forever, and
        pointless if nobody asked.
        """
        key = (tuple(sorted(ndv)), tuple(sorted(mean)), tuple(sorted(total)))
        if key not in self._cache:
            self._cache[key] = self._compute(*key)
        return self._cache[key]

    def _compute(
        self, ndv: tuple[str, ...], mean: tuple[str, ...], total: tuple[str, ...]
    ) -> Facts | None:
        """Collect source statistics, enrich them, and distil the facts — never raising."""
        from batcher.api.terminal.metadata_answer._core import _metadata_answerable, _source_stats
        from batcher.api.terminal.metadata_answer.enrich import enrich_in_memory

        ds = self._ds
        if not _metadata_answerable(ds._plan, ds._sources):
            return None
        try:
            from batcher import core
            from batcher.kyber.shortcuts import facts_for

            stats = _source_stats(ds._sources, None)
            stats = enrich_in_memory(ds._sources, stats, ndv=ndv, mean=mean, total=total)
            return facts_for(ds._plan, ds._sources, stats, core.default_hub())
        except Exception as exc:  # a shortcut must never break a runnable query
            note_suppressed("api", "compute a metadata fact", exc)
            return None

    def ask(
        self,
        shortcut: Callable[..., T | None],
        *args: Any,
        ndv: Iterable[str] = (),
        mean: Iterable[str] = (),
        total: Iterable[str] = (),
    ) -> T | None:
        """Put one question to a Kyber shortcut, or None when there are no facts to ask about.

        Every accessor method is one `ask` (the fast path) paired with one `answer` (the
        fallback), which is why they all read the same and why none of them can invent a
        third behaviour.
        """
        facts = self.facts(ndv=ndv, mean=mean, total=total)
        return None if facts is None else shortcut(facts, *args)

    def source_stats(self) -> list:
        """The connectors' declared `SourceStatistics`, or an empty list if unavailable."""
        from batcher.api.terminal.metadata_answer._core import _source_stats

        try:
            return _source_stats(self._ds._sources, None)
        except Exception:  # a connector that cannot describe itself describes nothing
            return []

    def require_column(self, column: str) -> str:
        """Validate that `column` is an output column, else raise `PlanError`."""
        available = self._ds._plan.available_columns()
        if column not in available:
            hint = suggest_columns(column, available)
            raise PlanError(f"meta: unknown column {column!r}{hint}")
        return column

    def require_columns(self, columns: Sequence[str]) -> tuple[str, ...]:
        """Validate every name in `columns`, returning them as a tuple."""
        return tuple(self.require_column(c) for c in columns)

    def _column_facts(self, column: str, **enrich: Any) -> Any:
        """The facts for one validated column — an all-unknown bundle when there are none."""
        from batcher.kyber.shortcuts import ColumnFacts

        facts = self.facts(**enrich)
        return ColumnFacts(name=column) if facts is None else facts.col(column)
