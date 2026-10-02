"""Split a vendored ``.sql`` query file and register each statement as a SQL case.

TPC-DS and JOB both ship their statements as one generated file with a ``-- @query <name>``
delimiter before each (written by ``tools/vendor_tpcds_queries.py`` and
``tools/vendor_job_queries.py``). This is the one reader for that format. Underscore-named,
so ``import_submodules`` does not import it as a suite.
"""

from __future__ import annotations

from collections.abc import Mapping

from registry import CaseBuilder, EngineQueries, Suite, sql_case

# The delimiter the vendoring tools write before each statement.
_MARKER = "-- @query "


def load_queries(path: str) -> dict[str, str]:
    """Split the vendored ``.sql`` file into ``{case name -> statement}``.

    Args:
        path: The vendored query file.

    Returns:
        Each query's case name mapped to its SQL text, in file order.
    """
    queries: dict[str, str] = {}
    name: str | None = None
    lines: list[str] = []
    with open(path) as fh:
        text = fh.read()
    for line in text.splitlines():
        if line.startswith(_MARKER):
            if name is not None:
                queries[name] = "\n".join(lines).strip()
            name = line[len(_MARKER) :].strip()
            lines = []
        elif name is not None:
            lines.append(line)
    if name is not None:
        queries[name] = "\n".join(lines).strip()
    return queries


def _refusing(query: str, refused: Mapping[str, str]) -> CaseBuilder:
    """The plain SQL fanout, with each engine in `refused` raising its reason instead of running.

    For an engine that does not merely lose a case but **cannot survive it**: the harness catches
    an exception per engine, but not a `SIGKILL`, and an OOM kill takes every engine's result
    for the case with it (under `--isolate`) or the rest of the suite (without). Raising keeps
    the fact in the report as that engine's error, with the measured reason beside it. It is the
    operator suite's `cannot_run`, for the vendored suites. An engine that is merely slow must
    keep its runner and be timed.
    """
    plain = sql_case(query)

    def build(ctx: object) -> EngineQueries:
        fns = plain(ctx)
        for engine, reason in refused.items():
            if engine in fns:

                def refuse(reason: str = reason) -> object:
                    raise RuntimeError(reason)

                fns[engine] = refuse
        return fns

    return build


def register_vendored(
    suite: Suite,
    path: str,
    *,
    count: int,
    vendor_tool: str,
    refuse: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, str]:
    """Load ``path``, check it holds exactly ``count`` queries, and register each on ``suite``.

    A truncated or half-written vendored file would otherwise shrink the benchmark silently:
    fewer cases register, every gate stays green, and the suite quietly stops being itself.

    Args:
        suite: The registrar the cases go to.
        path: The vendored query file.
        count: How many queries the benchmark defines.
        vendor_tool: The script that regenerates ``path``, named in the error.
        refuse: ``{case name -> {engine -> reason}}`` for engines a case kills rather than
            slows (see `_refusing`). A name that is not a case of the file is an error, so an
            entry cannot outlive the query it was written for.

    Returns:
        The loaded ``{case name -> statement}``.
    """
    queries = load_queries(path)
    if len(queries) != count:
        raise RuntimeError(
            f"{path} holds {len(queries)} queries, expected {count} — re-run `python {vendor_tool}`"
        )
    refuse = refuse or {}
    unknown = set(refuse) - set(queries)
    if unknown:
        raise RuntimeError(f"refusals name cases {sorted(unknown)} that {path} does not hold")
    for name, query in queries.items():
        if name in refuse:
            suite.sql_with_builder(name, query, _refusing(query, refuse[name]))
        else:
            suite.sql(name, query)
    return queries
