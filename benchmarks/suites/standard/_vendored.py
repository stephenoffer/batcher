"""Split a vendored ``.sql`` query file and register each statement as a SQL case.

TPC-DS and JOB both ship their statements as one generated file with a ``-- @query <name>``
delimiter before each (written by ``tools/vendor_tpcds_queries.py`` and
``tools/vendor_job_queries.py``). This is the one reader for that format. Underscore-named,
so ``import_submodules`` does not import it as a suite.
"""

from __future__ import annotations

from registry import Suite

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


def register_vendored(suite: Suite, path: str, *, count: int, vendor_tool: str) -> dict[str, str]:
    """Load ``path``, check it holds exactly ``count`` queries, and register each on ``suite``.

    A truncated or half-written vendored file would otherwise shrink the benchmark silently:
    fewer cases register, every gate stays green, and the suite quietly stops being itself.

    Args:
        suite: The registrar the cases go to.
        path: The vendored query file.
        count: How many queries the benchmark defines.
        vendor_tool: The script that regenerates ``path``, named in the error.

    Returns:
        The loaded ``{case name -> statement}``.
    """
    queries = load_queries(path)
    if len(queries) != count:
        raise RuntimeError(
            f"{path} holds {len(queries)} queries, expected {count} — re-run `python {vendor_tool}`"
        )
    for name, query in queries.items():
        suite.sql(name, query)
    return queries
