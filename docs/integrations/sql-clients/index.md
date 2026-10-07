# SQL clients

This section covers the tools that drive Batcher from the outside: a Python program written against the standard database API, SQLAlchemy, dbt, Ibis, and any client that speaks Arrow Flight SQL. Each one runs SQL through a {py:class}`Session <batcher.Session>`, the same object `bt.sql` uses, so the tables, views, functions, and catalogs you set up in Python are the ones the tool sees.

None of these adapters adds transactions. Batcher runs every statement to completion as it arrives, the way an autocommit connection behaves elsewhere, and each adapter states plainly what that means for its tool rather than pretending a rollback happened.

The following table maps each tool to its adapter:

| Tool | Adapter | Extra | Status |
|---|---|---|---|
| Any PEP 249 client | {doc}`batcher.dbapi <dbapi>` | none | Tested in this repository |
| SQLAlchemy 2.0 | {doc}`the batcher:// dialect <sqlalchemy>` | `sqlalchemy` | Tested against SQLAlchemy 2.0 here |
| dbt | {doc}`the dbt adapter pilot <dbt>` | `dbt` | Pilot, not yet run against dbt-core |
| Ibis | {doc}`the Ibis bridge <ibis>` | `ibis` | Pilot, not yet run against Ibis |
| Flight SQL clients | {doc}`the Flight SQL service <flight-sql>` | `flightsql` | Pilot, not yet run against a Flight SQL driver |

For the API reference, see {doc}`/api/operations/sql-clients`.

```{toctree}
:hidden:

dbapi
sqlalchemy
dbt
ibis
flight-sql
```
