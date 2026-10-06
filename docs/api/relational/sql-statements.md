# SQL statements

This page covers the SQL that changes or describes a session's catalog, rather than
querying it: the DDL, the DML clauses beyond a plain `INSERT`, `UPDATE` and `DELETE`, and
the three ways to ask what tables exist.

It continues {doc}`SQL </api/relational/sql>`, which covers the query surface and the
session the examples below assume:

```python
import batcher as bt

s = bt.Session()
s.register("events", bt.from_pydict({"id": [1, 2, 3], "amount": [30.0, 40.0, 5.0]}))
```

## Defining tables and views with SQL

`CREATE TABLE/VIEW ... AS` and `DROP TABLE` register and unregister a lazy dataset as a session table or view. Nothing is materialized until a terminal operation runs it. A *qualified* name, such as `CREATE TABLE sales.orders AS ...`, creates a stored table in the session's catalog instead, and so does an unqualified `CREATE TABLE` after `USE` has moved the session to a catalog namespace. `CREATE SCHEMA`, `USE`, `INSERT INTO sales.orders` and `SHOW DATABASES` work against the catalog. {doc}`/user-guide/moving-data/catalogs-and-tables` covers those statements.

`CREATE TEMP TABLE` and `CREATE TEMPORARY TABLE` always register a session table, after a `USE` too, because a temporary table lasts as long as the session. A qualified `TEMP` name is refused. To persist a table, create it without `TEMP`:

```python
s.sql("CREATE VIEW big_events AS SELECT id, amount FROM events WHERE amount > 25")
print(s.sql("SELECT * FROM big_events ORDER BY id").to_pydict())
# {'id': [1, 2], 'amount': [30.0, 40.0]}
```

## Running several statements

{py:meth}`Session.sql <batcher.Session.sql>` runs one statement. {py:meth}`Session.execute_script <batcher.Session.execute_script>` runs a `;`-separated script in order and returns one result per statement. It isn't atomic: when a statement fails, the ones before it stay applied, and the error carries a note saying how many completed.

```python
scratch = bt.Session()
results = scratch.execute_script(
    "CREATE TABLE t AS SELECT 1 AS x; INSERT INTO t VALUES (2); SELECT sum(x) AS total FROM t"
)
print(results[-1].to_pydict())
# {'total': [3]}
```

## MERGE INTO

`MERGE INTO` is the lakehouse DML statement, and it runs through the same engine as
{py:meth}`write.merge_into <batcher.Dataset.write>`: the SQL is translated into the same
`WHEN` clauses and composed by the same function, so the two spellings cannot disagree.

```python
s.sql("CREATE TABLE stock AS SELECT * FROM (VALUES (1, 10), (2, 20)) AS v(sku, qty)")
s.sql("CREATE TABLE delivery AS SELECT * FROM (VALUES (2, 5), (3, 7)) AS v(sku, qty)")

s.sql("""
    MERGE INTO stock USING delivery ON stock.sku = delivery.sku
    WHEN MATCHED THEN UPDATE SET qty = stock.qty + delivery.qty
    WHEN NOT MATCHED THEN INSERT (sku, qty) VALUES (delivery.sku, delivery.qty)
""")
print(s.sql("SELECT * FROM stock ORDER BY sku").to_pydict())
```

All three clause populations are supported, each with an optional `AND` condition:
`WHEN MATCHED`, `WHEN NOT MATCHED`, and `WHEN NOT MATCHED BY SOURCE`. A matched or
by-source clause may `UPDATE SET` or `DELETE`. A not-matched clause may `INSERT`.
`UPDATE SET *` and `INSERT *` take every column from the same-named source column. The
`USING` side may be a table or a subquery.

`ON` must be column equalities between the two sides, joined by `AND`. The key columns may
have different names, which is the usual shape of a change feed keyed by its own column:

```python
s.sql("CREATE TABLE feed AS SELECT * FROM (VALUES (1, 3), (4, 9)) AS v(item_id, qty)")
s.sql("""
    MERGE INTO stock USING feed ON stock.sku = feed.item_id
    WHEN MATCHED THEN UPDATE SET qty = feed.qty
    WHEN NOT MATCHED THEN INSERT (sku, qty) VALUES (feed.item_id, feed.qty)
""")
print(s.sql("SELECT * FROM stock ORDER BY sku").to_pydict())
# {'sku': [1, 2, 3, 4], 'qty': [3, 25, 7, 9]}
```

```{important}
The engine matches a source row to a target row by key columns, so a non-equality, or an
equality that does not span the two sides, has no key to be expressed as and is refused.
Filter the source in a `USING (SELECT ...)` subquery instead. With differently named keys,
`INSERT *` and `UPDATE SET *` are refused too: they take columns by name, so the key would
come from a source column the `ON` didn't name. List the columns.
```

`MERGE INTO` a catalog table takes the path `DELETE` and `UPDATE` on one take: the merged
rows are computed by the same rewrite, collected through the driver, and written back with
`mode="overwrite"`. It is not an incremental write. A catalog that can't overwrite a table
refuses it. `MERGE ... RETURNING` is refused.

## Upserts with ON CONFLICT

`INSERT ... ON CONFLICT (k) DO NOTHING` and `DO UPDATE SET ...` are an upsert, and Batcher
runs them as the merge they are: an inserted row whose key is already in the table is a
matched row, and every other row is inserted. In `DO UPDATE SET`, `excluded.col` is the row
being inserted and a bare or table-qualified column is the existing row. An optional
`WHERE` limits which conflicting rows are updated:

```python
s.sql("CREATE TABLE prices AS SELECT * FROM (VALUES (1, 10.0), (2, 20.0)) AS v(sku, price)")
s.sql("""
    INSERT INTO prices VALUES (2, 25.0), (3, 30.0)
    ON CONFLICT (sku) DO UPDATE SET price = excluded.price WHERE excluded.price > prices.price
""")
print(s.sql("SELECT * FROM prices ORDER BY sku").to_pydict())
# {'sku': [1, 2, 3], 'price': [10.0, 25.0, 30.0]}
```

Three things differ from an engine with declared constraints. Batcher tables have no primary
key, so the conflict target `(k)` is required, where DuckDB infers it. Two inserted rows
with the same key are refused before anything changes, because which one wins would depend
on row order. Rows with a NULL key never conflict, as SQL defines it, so each is inserted.
`ON CONFLICT` acts on session tables only.

## Returning the changed rows

`RETURNING` on an `INSERT`, `UPDATE` or `DELETE` of a session table makes the statement
return the rows it touched instead of the table's new state, projected by any select list.
`INSERT` returns the inserted rows, `UPDATE` the updated rows with their new values, and
`DELETE` the deleted rows. The table is changed either way:

```python
gone = s.sql("DELETE FROM prices WHERE price < 20 RETURNING sku, price * 2 AS doubled")
print(gone.to_pydict())
# {'sku': [1], 'doubled': [20.0]}
```

`RETURNING` on a catalog table, on `MERGE`, and together with `ON CONFLICT` is refused.

## Deleting by another table

`DELETE FROM t USING s WHERE ...` deletes every row of `t` for which some row of `s`
satisfies the condition. Batcher answers it as the `EXISTS` it means, so a row matched by
several rows of `s` is deleted once:

```python
s.sql("CREATE TABLE recalled AS SELECT * FROM (VALUES (2), (2), (9)) AS v(sku)")
s.sql("DELETE FROM prices USING recalled WHERE prices.sku = recalled.sku")
print(s.sql("SELECT sku FROM prices ORDER BY sku").to_pydict())
# {'sku': [3]}
```

The condition correlates through equalities between plain columns, the same rule as a
correlated `EXISTS` in a query. A condition that reads only the target, such as `AND
prices.price > 0`, can sit beside them.

## Listing and describing tables

`SHOW TABLES` and `DESCRIBE` are the two statements a SQL client issues before it issues a
query: a BI tool fills its table picker from the first, and a schema browser or a SQLAlchemy
reflection reads the second. Both return ordinary relations, so you can filter and join them
like any other result.

```python
print(s.sql("SHOW TABLES").to_pydict())
```

`DESCRIBE` returns DuckDB's six columns, so a client written against DuckDB reads it
unchanged:

```python
print(s.sql("DESCRIBE events").to_pydict()["column_name"])
```

`key`, `default` and `extra` are always null. Batcher has no primary keys, column defaults
or storage attributes to report, and a plausible value there would be worse than an empty
one.

```{note}
`column_type` carries Batcher's own type names, not DuckDB's: an integer column reads
`int64` where DuckDB says `BIGINT`. The engine stores Arrow and {py:attr}`Dataset.schema
<batcher.Dataset.schema>` reports Arrow, so printing a DuckDB spelling here would describe
storage that does not exist. The shape is DuckDB's; the content is this engine's.
```

### The ANSI spelling

A SQLAlchemy reflection and several BI tools read `information_schema` rather than `SHOW` or
`DESCRIBE`. Four views are served, answered from the same session, so the spellings cannot
disagree about what exists:

```python
s.sql("CREATE VIEW cheap AS SELECT sku FROM prices WHERE price < 50")
q = "SELECT table_name, table_type FROM information_schema.tables WHERE table_name IN ('events', 'cheap')"
print(s.sql(q + " ORDER BY table_name").to_pydict())
# {'table_name': ['cheap', 'events'], 'table_type': ['VIEW', 'BASE TABLE']}
```

The following table lists the columns of each view:

| View | Columns |
|---|---|
| `information_schema.tables` | `table_catalog`, `table_schema`, `table_name`, `table_type` |
| `information_schema.columns` | `table_catalog`, `table_schema`, `table_name`, `column_name`, `ordinal_position`, `column_default`, `is_nullable`, `data_type` |
| `information_schema.views` | `table_catalog`, `table_schema`, `table_name`, `view_definition` |
| `information_schema.schemata` | `catalog_name`, `schema_name` |

These are the columns a reflection selects. DuckDB's views are wider; the extra columns are
null for an Arrow relation, and padding them out would be inventing a shape rather than
reporting one.

`table_type` is `VIEW` for a `CREATE VIEW` and `BASE TABLE` for a session table, whether it
came from `CREATE TABLE ... AS` or {py:meth}`Session.register <batcher.Session.register>`.
`view_definition` is the query the view stores, as Postgres reports it, rather than DuckDB's
full `CREATE VIEW` statement. Session tables and views are reported in catalog `batcher`,
schema `main`, because they belong to the session rather than to an attached catalog.
`schemata` lists that schema and every namespace of every attached catalog. Catalog tables
are not listed in `tables` or `columns`; `SHOW TABLES` lists the current namespace's.

Any other `information_schema` view is refused with an error naming the four.

### Explaining a query

`EXPLAIN <query>` returns the planned operator tree as a one-row relation, in DuckDB's
`explain_key`/`explain_value` shape, and `EXPLAIN ANALYZE` runs the query and reports what
it measured. The query is read in the session's dialect and resolves views and catalog
tables as it would without `EXPLAIN`, and a syntax error in it raises the same
{py:exc}`PlanError <batcher.PlanError>` the bare query would. Explaining a statement
that changes the catalog, such as `EXPLAIN DELETE ...`, is refused rather than run:

```python
plan = s.sql("EXPLAIN SELECT sku FROM cheap").to_pydict()
print(plan["explain_key"])
# ['plan']
```

## See also

- {doc}`SQL </api/relational/sql>`: the query surface these statements sit beside.
- {doc}`Dataset </api/relational/dataset>`: `write.merge_into`, the builder `MERGE INTO`
  translates into.
