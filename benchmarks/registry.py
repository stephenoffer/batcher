"""Benchmark registry: the single index every query registers itself into.

A benchmark is one logical query expressed once per engine. Each lives in a
family module under ``suites/`` and registers itself through a ``suite(...)``
handle, so the file count stays flat as the suite grows from tens to thousands of
queries (the maintainability pattern the rest of the codebase uses: grouped-by-
family modules plus a registry, never a god file and never one-file-per-query).

A family module looks like::

    from registry import suite

    joins = suite("joins", dataset="synthetic")

    @joins.case("join-left")
    def _(ctx):
        return {
            "batcher": lambda: ...,
            "duckdb":  lambda: ...,
            "polars":  lambda: ...,
        }

The callable a case returns per engine takes no arguments and returns a
``pyarrow.Table``; ``None`` marks an engine that does not express the query. The
``ctx`` is the prepared data context for the case's ``dataset`` (see
``contexts.py``), carrying the per-engine table handles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

# A query expressed per engine: engine name -> zero-arg callable -> table (or None).
EngineQueries = dict[str, Callable[[], "pa.Table"] | None]
# A case builder: given its data context, return the per-engine queries.
CaseBuilder = Callable[[Any], EngineQueries]


def sql_case(query: str) -> CaseBuilder:
    """A case builder that fans one SQL string out across every SQL-capable engine.

    The context exposes ``sql_runners()`` — a mapping of engine name to a
    pre-registered ``query -> pa.Table`` callable — so the standard suites
    (TPC-H / TPC-DS / ClickBench) express each query exactly once. Engines without a
    SQL surface (PyArrow, Ray Data) are simply absent from the mapping and show as
    ``n/a``, never a wrong answer.
    """

    def build(ctx: Any) -> EngineQueries:
        return {name: (lambda run=run: run(query)) for name, run in ctx.sql_runners().items()}

    return build


@dataclass(frozen=True)
class Case:
    """One registered benchmark: a name, its family, its dataset, and its builder."""

    family: str
    name: str
    dataset: str
    build: CaseBuilder


class Registry:
    """Process-wide collection of registered benchmark cases."""

    def __init__(self) -> None:
        self._cases: list[Case] = []
        self._names: set[str] = set()

    def add(self, case: Case) -> None:
        if case.name in self._names:
            raise ValueError(f"duplicate benchmark name: {case.name!r}")
        self._names.add(case.name)
        self._cases.append(case)

    def select(
        self,
        *,
        dataset: str | None = None,
        family: str | None = None,
        name: str | None = None,
    ) -> list[Case]:
        """Return cases filtered by dataset (exact), family (exact), and name (substring)."""
        out = self._cases
        if dataset is not None:
            out = [c for c in out if c.dataset == dataset]
        if family is not None:
            out = [c for c in out if c.family == family]
        if name is not None:
            out = [c for c in out if name in c.name]
        return list(out)

    def datasets(self) -> list[str]:
        # Preserve first-seen order so output groups read predictably.
        seen: dict[str, None] = {}
        for c in self._cases:
            seen.setdefault(c.dataset, None)
        return list(seen)

    def families(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self._cases:
            seen.setdefault(c.family, None)
        return list(seen)


# The one registry every suite module writes into.
REGISTRY = Registry()


@dataclass(frozen=True)
class Suite:
    """A registrar bound to one family and dataset, so each case only names itself."""

    family: str
    dataset: str

    def case(self, name: str) -> Callable[[CaseBuilder], CaseBuilder]:
        def register(build: CaseBuilder) -> CaseBuilder:
            REGISTRY.add(Case(family=self.family, name=name, dataset=self.dataset, build=build))
            return build

        return register

    def sql(self, name: str, query: str) -> None:
        """Register a SQL benchmark — one query string, fanned across SQL engines."""
        REGISTRY.add(
            Case(family=self.family, name=name, dataset=self.dataset, build=sql_case(query))
        )


def suite(family: str, *, dataset: str) -> Suite:
    """Open a registrar for ``family`` over ``dataset`` (the benchmark, see ``context.py``)."""
    return Suite(family=family, dataset=dataset)
