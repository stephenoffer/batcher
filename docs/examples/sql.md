# SQL

This page covers the scripts that run SQL over Datasets, and the ones that pin down where
the SQL surface differs from what a reader might assume.

## SQL and the DataFrame API are the same thing

A query is parsed into the logical plan the DataFrame API builds, so `bt.sql` returns a lazy
Dataset rather than a materialized table. That means the two spellings interoperate freely
and you can move between them mid-pipeline.

```python
import batcher as bt
from batcher import col

sales = bt.from_pydict(
    {"region": ["west", "east", "west"], "amount": [10, 20, 30]}
)

summary = bt.sql(
    "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY region",
    sales=sales,
)

# Still a Dataset, so the DataFrame API picks up where the query left off.
biggest = summary.filter(col("total") > 15).sort("total", descending=True)
assert biggest.to_pydict()["region"] == ["west", "east"]
```

`ds.sql` is the same thing scoped to one Dataset, which it calls `self`. `bt.Session` gives a
query its own catalog, which is what you want when two parts of a process register tables
under the same names.

## Where it surprises people

Group and order by the alias rather than by an ordinal. `ORDER BY 1` against a computed
projection does not resolve, and the error names the column it could not find.

Three-valued logic behaves as the standard requires, which is to say it catches people out.
`COUNT(*)` counts rows while `COUNT(column)` counts non-nulls; a comparison against null is
null, so neither `= 10` nor `<> 10` keeps a null row; and `IS NULL` is the only test that
finds them. `examples/sql_queries/null_semantics_in_sql.py` asserts all three against a real
left join.

The parser takes a dialect, so a query written for another engine can be run before it is
ported. That matters for a migration: you prove the old query still returns the same rows
here, and only then rewrite it.

## Every script on this page

The table below lists the SQL scripts in path order.

<!-- library-table: sql_queries -->
| Script | Shows |
| --- | --- |
| `examples/sql_queries/aggregates_and_having.py` | GROUP BY and HAVING in SQL, and the DataFrame equivalent |
| `examples/sql_queries/basics.py` | SQL over Datasets: bt.sql with named table bindings |
| `examples/sql_queries/ctes_and_views.py` | Naming intermediate results: CTEs in a query, views in a catalog |
| `examples/sql_queries/date_and_string_functions.py` | Date and string functions in SQL over real data |
| `examples/sql_queries/joins_and_subqueries.py` | Joins, CTEs and subqueries in SQL over real TPC-H tables |
| `examples/sql_queries/mixing_sql_and_dataframe.py` | Moving between SQL and the DataFrame API mid-pipeline |
| `examples/sql_queries/null_semantics_in_sql.py` | Three-valued logic in SQL, and where it surprises people |
| `examples/sql_queries/set_operations_and_cases.py` | UNION, CASE and IN, written as SQL over real tables |
| `examples/sql_queries/spark_dialect.py` | Reading SQL written for another engine |
| `examples/sql_queries/sql_over_files.py` | Querying a file directly from SQL |
| `examples/sql_queries/window_frames_in_sql.py` | Window frames spelled out in SQL |
| `examples/sql_queries/window_functions.py` | Window functions in SQL, with the frame spelled out |
<!-- /library-table -->
