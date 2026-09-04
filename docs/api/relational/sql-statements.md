# SQL statements

This page covers the SQL that changes or describes a session's catalog, rather than
querying it: the DDL, `MERGE INTO`, and the three ways to ask what tables exist.

It continues {doc}`SQL </api/relational/sql>`, which covers the query surface and the
session the examples below assume:

```python
import batcher as bt

s = bt.Session()
s.register("events", bt.from_pydict({"id": [1, 2, 3], "amount": [30.0, 40.0, 5.0]}))
```

## Defining tables and views with SQL

`CREATE TABLE/VIEW ... AS` and `DROP TABLE` register and unregister a **lazy** dataset in the session catalog. Nothing is materialized until a terminal operation runs it:

```python
s.sql("CREATE VIEW big_events AS SELECT id, amount FROM events WHERE amount > 25")
print(s.sql("SELECT * FROM big_events ORDER BY id").to_pydict())
# {'id': [3, 4, 5], 'amount': [30.0, 40.0, 50.0]}
```

## MERGE INTO

`MERGE INTO` is the lakehouse DML statement, and it runs through the same engine as
{py:meth}`write.merge_into <batcher.Dataset.write>`: the SQL is translated into the same
`WHEN` clauses and composed by the same function, so the two spellings cannot disagree.

```python
s.sql("CREATE TABLE stock AS SELECT * FROM (VALUES (1, 10), (2, 20)) AS v(sku, qty)")
s.sql("CREATE TABLE delivery AS SELECT * FROM (VALUES (2, 5), (3, 7)) AS v(sku, qty)")

s.sql('''
    MERGE INTO stock USING delivery ON stock.sku = delivery.sku
    WHEN MATCHED THEN UPDATE SET qty = stock.qty + delivery.qty
    WHEN NOT MATCHED THEN INSERT (sku, qty) VALUES (delivery.sku, delivery.qty)
''')
print(s.sql("SELECT * FROM stock ORDER BY sku").to_pydict())
```

All three clause populations are supported, each with an optional `AND` condition:
`WHEN MATCHED`, `WHEN NOT MATCHED`, and `WHEN NOT MATCHED BY SOURCE`. A matched or
by-source clause may `UPDATE SET` or `DELETE`; a not-matched clause may `INSERT`. The
`USING` side may be a table or a subquery.

```{important}
`ON` must be equalities of the form `target.k = source.k`, on the same column name on both
sides, joined by `AND`. The engine matches a source row to a target row by key column, so a
join on differing names or on any non-equality has no key to be expressed as, and is
refused. Rename the columns to match, or filter the source before merging.
```

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
`DESCRIBE`. Two views are served, answered from the same session catalog, so the three
spellings cannot disagree about what exists:

```python
print(s.sql("SELECT table_name FROM information_schema.tables ORDER BY table_name").to_pydict())
```

`information_schema.tables` carries `table_catalog`, `table_schema`, `table_name` and
`table_type`. `information_schema.columns` carries those first three plus `column_name`,
`ordinal_position`, `column_default`, `is_nullable` and `data_type` — the columns a
reflection actually selects. DuckDB's views are wider; the extra columns are null for an
Arrow relation, and padding them out would be inventing a shape rather than reporting one.

`table_type` is always `BASE TABLE`. Batcher does not distinguish a table from a view:
`CREATE TABLE … AS`, `CREATE VIEW … AS` and {py:meth}`Session.register
<batcher.Session.register>` all bind a lazy `Dataset`, so there is one kind of thing in the
catalog and one value to report for it.

## See also

- {doc}`SQL </api/relational/sql>`: the query surface these statements sit beside.
- {doc}`Dataset </api/relational/dataset>`: `write.merge_into`, the builder `MERGE INTO`
  translates into.
