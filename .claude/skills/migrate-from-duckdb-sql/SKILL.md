---
name: migrate-from-duckdb-sql
description: Write new SQL against Batcher or port an existing DuckDB/SQL workload onto it — the bt.sql/bt.Session/ds.sql entry points, the honest list of supported vs unsupported SQL constructs, the SQL-to-DataFrame translation table for the gaps, the connection-to-lazy-plan concept shifts, and how to verify against DuckDB itself. Invoke when moving SQL/DuckDB code onto Batcher, writing new SQL against it, or checking whether a SQL construct is supported.
---

# Write or migrate SQL on Batcher

Batcher reads **DuckDB syntax by default** and runs it through the same optimizer and
engine as the DataFrame API — there is no separate SQL engine, and the result of a query
is a lazy `Dataset` you can keep building on. So most ports start by pasting the SQL in
unchanged. This skill covers what survives that paste, what does not, and how to prove
the port is correct.

DuckDB is also this repo's **differential correctness oracle**
(`tests/differential/conftest.py::assert_same`), which makes equivalence checking
unusually easy: run both, compare multisets.

## Writing new SQL (not just porting)

Nothing here is porting-only. If you are authoring fresh SQL against Batcher, the same
three facts are the whole story:

- **`bt.sql(sql, **tables)`** — one-off query, each keyword binds a name in the query to a
  `Dataset` / `pyarrow.Table`. **`bt.Session()`** — a persistent name→`Dataset` catalog with
  `register` / `sql` / `table` / `list` / `drop` / `clear` when you have several statements
  or want `CREATE VIEW`. **`ds.sql("SELECT … FROM self")`** — query one Dataset directly.
  **`bt.register_function(name, fn)`** — your own function callable from SQL.
- **The result is a lazy `Dataset`, not rows.** SQL and DataFrame lower to the same plan and
  the same optimizer, so mixing them is free: `bt.sql(...).filter(...).write.parquet(...)`.
  Write whichever is clearer per step.
- **Check the construct before you write it.** The two tables below are the supported and
  unsupported inventories — read the unsupported one first when a query uses anything past
  plain select/join/group-by, so you write the working spelling rather than debugging the
  rejected one.

## Run the SQL first

```python
import batcher as bt

orders = bt.read.parquet("orders.parquet")

out = bt.sql(
    "SELECT region, SUM(amount) AS total "
    "FROM orders WHERE status = 'paid' GROUP BY region ORDER BY total DESC",
    orders=orders,
)
print(out.to_pydict())
```

Each keyword binds a table name in the query to a `Dataset` or a `pyarrow.Table`. The
result is lazy — chain `.filter(...)` / `.with_columns(...)` / another `bt.sql` onto it
and only the final terminal op executes.

For a persistent catalog (the `duckdb.connect()` analogue), use a `Session`:

```python
s = bt.Session()                          # Session(*, dialect="duckdb")
s.register("orders", orders)              # name -> Dataset
s.sql("CREATE VIEW paid AS SELECT * FROM orders WHERE status = 'paid'")
totals = s.sql("SELECT region, SUM(amount) AS t FROM paid GROUP BY region")
s.list()          # -> registered table names
s.table("paid")   # -> the Dataset behind a name
s.drop("paid"); s.clear()
```

`bt.sql` / `bt.register_function` share one process-global default session, so a
`CREATE TABLE/VIEW AS` from `bt.sql` is visible to a later `bt.sql`. Use an explicit
`Session` when you want isolation. `ds.sql("SELECT ... FROM self")` runs a query against a
single Dataset bound as `self` (`table_name=` renames it). `dialect=` on either call
selects any other sqlglot read dialect.

Register Python functions with
`bt.register_function(name, fn)` / `s.register_function(name, fn, result_type=...)` —
vectorized over Arrow arrays by default.

## What is supported

Most of an analytics workload. Verified working: `SELECT`/`WHERE`/`GROUP BY`/`HAVING`/
`ORDER BY`/`LIMIT`/`OFFSET`; `WITH` CTEs; `UNION`/`UNION ALL`/`INTERSECT`/`EXCEPT`;
`SELECT DISTINCT` and `DISTINCT ON (...)`; `CASE`, `CAST`/`TRY_CAST`, `COALESCE`,
`NULLIF`, `GREATEST`/`LEAST`, `LIKE`/`ILIKE`, `IN`, `BETWEEN`; `GROUP BY ALL`, positional
`GROUP BY`/`ORDER BY`, grouping by SELECT alias; **`ROLLUP` / `CUBE` / `GROUPING SETS`**
(expanded to a UNION ALL, with `GROUPING()`); `INNER`/`LEFT`/`RIGHT`/`FULL`/`CROSS` joins
plus `USING` and `NATURAL`; scalar, `IN`, `NOT IN`, `EXISTS`/`NOT EXISTS` subqueries
including **correlated** ones (decorrelated to joins, with the COUNT-bug handled);
`OVER (PARTITION BY … ORDER BY … ROWS …)` window functions and named `WINDOW w AS`;
`agg(...) FILTER (WHERE …)`; DuckDB star modifiers `EXCLUDE`/`REPLACE`/`RENAME` and
`COLUMNS('regex')`; `TABLESAMPLE`; `VALUES`; `EXPLAIN [ANALYZE]`; and DDL/DML —
`CREATE [OR REPLACE] TABLE|VIEW AS`, `DROP TABLE`, `INSERT`, `UPDATE`, `DELETE`.

**Vector search in SQL.** The two-argument vector functions run in SQL, so retrieval is a
plain query — `SELECT id FROM docs ORDER BY list_cosine_similarity(emb, [0.1, 0.2]) DESC
LIMIT 10`. Supported (DuckDB-matched): `list_cosine_similarity`, `list_cosine_distance`,
`list_distance` (L2), `list_dot_product`/`list_inner_product`, `l1_distance`, and the bare
`cosine_similarity` / `cosine_distance` / `euclidean_distance` / `dot_product` spellings;
element-wise vector math via `list_add`/`list_subtract`/`list_multiply`. The query vector is
an array literal (`[…]`) or another list column. Elementwise **ML activations** are also
callable in SQL — `sigmoid`, `relu`, `softplus`, `logit`, and the modern
`silu`/`swish`, `gelu`, `mish`, `hardsigmoid`, `hardswish` — so a scoring/feature
step (e.g. `SELECT sigmoid(w0 + w1*f) AS p`) stays in the query. These are Batcher
extensions, not DuckDB functions, and each matches its `torch.nn.functional` counterpart.

## What is not — rewrite these as DataFrame calls

Be honest with the user about these; each raises a clear `NotImplementedError`/`PlanError`
rather than returning a wrong answer.

| SQL construct | Status | Do this instead |
|---|---|---|
| `read_parquet('f.parquet')` and friends | **no file-scanning table functions** | `bt.read.parquet("f.parquet")` and bind it: `bt.sql("… FROM t", t=ds)` |
| `ASOF JOIN` | **parsed** — `ON l.k = r.k AND l.t >= r.t` lowers to `join_asof` | the DataFrame form `left.join_asof(right, on="ts", by="symbol", tolerance="5m")` adds a staleness cap and `direction="nearest"`, which the SQL clause cannot express |
| Non-equi / theta join (`ON a > b` only) | equi-only engine | equality conjunct + a `WHERE` residual, or pre-filter |
| `PIVOT` / `UNPIVOT` | **parsed** with an explicit `IN (...)` value list | `ds.pivot(...)` / `ds.unpivot(...)` (`ds.melt`, `ds.crosstab`) when the values are discovered rather than listed |
| `QUALIFY` on a window not in `SELECT` | partial | project the window with an alias, then `QUALIFY alias = 1` — or `.with_columns(rn=…over(…)).filter(bt.col("rn") == 1)` |
| `LATERAL`, `UNNEST` in `FROM` | **parsed** — `FROM t, UNNEST(arr)` and `LATERAL (SELECT …)` both lower; `UNNEST` adds the element column (named `unnest`, or by `AS u(x)`) beside the list, as DuckDB does | `ds.explode("col")` / `ds.unnest("struct_col")` for the DataFrame form |
| `WITH RECURSIVE` | body translated once — **wrong answer risk** | rewrite as an explicit loop of `Dataset` unions in Python |
| `MERGE INTO` | unsupported DML | `ds.write.delta(uri, merge_on=["id"])` — one transactional call |
| `array_agg(DISTINCT x)`, `string_agg(DISTINCT x)` | rejected — the list aggregates have no dedup form | pre-aggregate the distinct values in a subquery |
| `SUM(DISTINCT x)` beside `AVG`/`STDDEV`/`VAR`/a quantile/a second `COUNT(DISTINCT y)` | rejected — those have no single-column mergeable partial to survive the dedup | compute them in a separate subquery and join (`SUM/AVG/MIN/MAX(DISTINCT x)` alone, or beside `COUNT`/`SUM`/`MIN`/`MAX`/`BOOL_*`/`BIT_*`/`PRODUCT`/`ANY_VALUE`, is fine) |
| Frame `EXCLUDE (TIES/GROUP/CURRENT ROW)` | rejected — honouring the frame while dropping `EXCLUDE` would be a wrong answer | rewrite the exclusion as a predicate, or use a `GROUPS` frame |
| `x > ANY (subquery)`, `x >= ALL (subquery)` | rejected — `> ALL` over a NULL is UNKNOWN, and the `max()` rewrite says TRUE | `x > (SELECT min(c) …)` for `ANY`; for `ALL` add `AND NOT EXISTS (SELECT 1 … WHERE c IS NULL)`. `= ANY`/`= SOME`/`<> ALL` need no rewrite — they are `IN`/`NOT IN` |
| `IN (subquery)` under `OR` | rejected — a semi-join drops the rows the `OR` keeps | write it as `EXISTS (SELECT 1 FROM s WHERE s.c = t.x)`, qualifying the outer column |
| Non-column `PARTITION BY`/window `ORDER BY` | **supported** — a computed key is hoisted into a hidden column | nothing; `PARTITION BY date_trunc('month', ts)` works |
| `INSERT … ON CONFLICT` / `RETURNING` | unsupported | `write.delta(..., merge_on=...)` |
| Scalar UDF in `GROUP BY` / agg arg / `ORDER BY` | rejected | compute it as a projected alias in a subquery first |
| Non-constant `LIKE`/`regexp_*`/`substr` arguments | constants only | restructure, or use the `.str` expression namespace |

## The DuckDB-specific shifts

- **There is no connection and nothing to close.** No `duckdb.connect()`, no cursor, no
  `con.execute(...).fetchall()`. `import batcher as bt` and the engine is in-process.
  `bt.Session` is the closest analogue and holds only a name→`Dataset` catalog.
- **A query returns a lazy plan, not a result.** `bt.sql(...)` executes nothing. DuckDB's
  `.fetchall()`/`.df()`/`.arrow()` become the terminals `to_pylist()` / `to_pandas()` /
  `to_arrow()` / `collect()`; `to_pydict()` is the columnar one. This is a feature: keep
  composing before you materialize, and the optimizer sees the whole query.
- **Files are read by the DataFrame API, not by the query.** DuckDB's habit of
  `FROM 'data/*.parquet'` or `read_parquet(...)` has no equivalent. Read with
  `bt.read.parquet(...)` (globs and `s3://` URLs work) and bind the result, or
  `s.register("t", bt.read.parquet(...))` once and reference `t` everywhere.
- **`EXPLAIN` works, but `ds.explain()` is better.** It returns the optimized plan as a
  string with row estimates tagged `exact` / `default` / `learned` plus a `decisions:`
  section. `ds.explain(analyze=True)` runs it; `ds.stats()` gives measured per-operator
  rows/time/bytes/spill after a run.
- **Types are Arrow types.** `CAST(x AS BIGINT)` works, and narrow numerics are normalized
  once at the FFI boundary (Int8/16/32 → Int64, Float16/32 → Float64), so a ported query
  may come back wider than DuckDB returned it. `DECIMAL` compares as float in the
  differential harness — do not assert exact decimal identity across the two engines.
- **Row order is incidental unless you `ORDER BY`.** DuckDB's order for an unordered query
  is also incidental, but ports that "matched yesterday" usually never had an `ORDER BY`.
- **Uncorrelated scalar subqueries are eagerly collected** and inlined as literals, and a
  CTE referenced more than once is materialized. Both are usually what you want; be aware
  if a CTE is enormous and referenced twice.

## Porting recipe

1. **Inventory the SQL.** List every statement, every source table, and every construct
   in the "not supported" table above. Those are the only ones needing a rewrite.
2. **Replace file scans with readers.** Every `read_parquet('...')` / `FROM 'x.csv'`
   becomes a `bt.read.*` call bound as a keyword or registered in a `Session`. Keep the
   paths byte-identical to the DuckDB script so both read the same input.
3. **Paste the query in.** `bt.sql(sql_text, **tables)`. Run it. Most queries work here
   and step 4 is empty.
4. **Rewrite what raised.** Use the translation table. The error messages are specific —
   they name the construct and usually name the replacement method.
5. **Keep it as SQL, or lower it deliberately.** SQL and DataFrame lower to the same plan,
   so there is no performance reason to convert a working query. Convert only where the
   DataFrame form is *clearer* — or where a construct forces it.
6. **Verify against DuckDB.** Run both on the same input and compare **order-independently**
   unless the query ends in `ORDER BY`:

   ```python
   import batcher as bt
   import duckdb  # doctest: +SKIP

   batcher_rows = sorted(map(tuple, zip(*ported.to_pydict().values())))
   duck_rows = sorted(duckdb.sql(original_sql).fetchall())  # doctest: +SKIP
   assert batcher_rows == duck_rows
   ```

   In-repo, mirror `tests/differential/conftest.py::assert_same` — a multiset comparison
   tolerant of int↔float, Decimal→float, and float rounding. Use `assert_same_ordered`
   when order is part of the contract.
7. **Check the plan, then the clock.** `print(ported.explain())` to confirm predicates and
   projections pushed into the scan, then `ported.stats()` for the measured profile.

## Gotchas / do-not

- **Do not assume a construct works because it parsed.** `WITH RECURSIVE` is the trap: the
  body translates once and returns a plausible, wrong answer. Rewrite it explicitly.
- **Do not trust printed output in the docs as an assertion.** The doc-example harness
  executes each block and fails on exception; the `# {...}` output comments beside them
  are not verified. Verify against DuckDB, not against a comment.
- **Do not `collect()` between statements** to feed the next one. Chain the `Dataset` or
  use a `Session` view — collecting throws away cross-statement optimization.
- **Do not port a DuckDB `PRAGMA`/`SET` into Batcher config one-for-one.** Threads,
  memory limits, and partition counts are chosen adaptively; `bt.Config` /
  `bt.config_context` exist, but set them only when a measurement says to.
- **Do not leave a scalar UDF in a `GROUP BY` key.** It is rejected for a reason — project
  it to a column in a subquery first.
- **Do not read a file inside the SQL string.** There is no `read_parquet` here, and the
  error (`unknown table ''`) is confusing enough that it is worth checking first.

## See also

- `docs/user-guide/analyze/sql.md` — the SQL surface; `docs/api/relational/sql.md` — the feature table.
- `docs/tutorials/foundations/sql-to-dataframe.md`, `docs/benchmarks/comparisons/vs-duckdb.md`.
- `docs/user-guide/analyze/joins.md`, `docs/user-guide/analyze/aggregations.md`, `docs/user-guide/analyze/window-functions.md`, `docs/user-guide/transform/columns/udfs.md`, `docs/user-guide/operate/tuning/explain-plans.md`.
- Skills: `write-a-batcher-pipeline` (the DataFrame API you rewrite gaps into),
  `migrate-from-spark`, `migrate-from-polars-or-pandas`, `run-a-distributed-job`.
