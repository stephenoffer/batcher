"""H2O.ai db-benchmark: the standard groupby and join workload for dataframe engines.

`db-benchmark <https://github.com/h2oai/db-benchmark>`_ is where Polars, DuckDB, data.table,
pandas, Spark and Dask all publish comparable numbers, and it probes a different axis than
TPC-H does: no snowflake schema and no subqueries, just one wide table aggregated by keys of
very different cardinality, and one LHS joined against three RHS tables spanning six orders
of magnitude in size. That makes it the direct test of group-by state management and of
join build-side selection, which is exactly where an adaptive optimizer either pays off or
does not.

Two family modules, matching the benchmark's two tasks: ``groupby`` (its ten questions) and
``join`` (its five). Both read the tables ``datagen.h2o_tables`` builds to the benchmark's
published generator spec. Family modules are auto-discovered, as in the other suites.
"""

from __future__ import annotations

from discover import import_submodules

import_submodules(__name__)
