"""Function-surface censuses: run a reference engine and Batcher side by side.

Reading a competitor's source says what it *has*; only executing both says what Batcher
*answers*, which is the thing a migrating user experiences. Each script here enumerates
one reference engine's own function catalogue, synthesizes a call per signature, runs it
in both engines, and sorts the results into three buckets:

* **match** — Batcher agrees with the reference.
* **gap** — the reference answered and Batcher raised.
* **mismatch** — both answered, differently. This is the valuable bucket: a gap is an
  honest refusal, a mismatch is a wrong answer nobody has noticed.

Run one and read the summary line on stderr::

    python tools/census/duckdb_functions.py > /tmp/census.json
    python tools/census/spark_examples.py  > /tmp/spark.json
    python tools/census/polars_methods.py  > /tmp/polars.json

The oracles differ by what each engine makes available:

``duckdb_functions.py``
    The live engine. `duckdb_functions()` lists every scalar and aggregate; 478 have a
    signature the script can synthesize an argument for.

``spark_examples.py``
    Spark's own source, and **no JVM is needed** — every builtin carries an
    ``@ExpressionDescription`` with a runnable example and its expected output, and
    ``FunctionRegistry.scala`` maps each expression class to its SQL name. Reads them out
    of the checkout under ``/mnt/shared_storage/ref/spark``.

``polars_methods.py``
    The live library: every zero-argument method on ``pl.Expr`` and its ``.str``/``.dt``/
    ``.list`` namespaces, against the same method name here.

**Re-run after every change, not just at the end.** A wave that closes twenty gaps can
open a mismatch — a name wired to a function whose semantics differ — and only a re-run
sees it. That is how `get_bit` was caught being answered with Spark's bit order for
DuckDB's bit-string function.

The running record of what these censuses found and what closed it lives in
``docs/architecture/internals/competitor_parity_census.md``.
"""

from __future__ import annotations

__all__: list[str] = []
