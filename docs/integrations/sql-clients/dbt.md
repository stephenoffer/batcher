# dbt

The Batcher dbt adapter is a pilot. It covers the `table` and `view` materializations, `dbt run`, and `dbt test`, over a Batcher catalog that can persist tables on disk.

:::{warning}
Not yet verified against a live dbt-core installation; see tests/PENDING_VERIFICATION.md. It is written against dbt-adapters 1.x, and its own code is tested against stand-ins for the dbt classes it extends.
:::

## Configure a profile

Install the `dbt` extra, which pulls in dbt-core and dbt-adapters. dbt finds an adapter by importing `dbt.adapters.<type>`, and Batcher ships that module, so a profile names `type: batcher`. The adapter's code lives in `batcher.integrations.dbt`.

```yaml
my_project:
  target: dev
  outputs:
    dev:
      type: batcher
      database: warehouse        # the catalog name relations render under
      schema: analytics          # a namespace in that catalog
      path: /data/warehouse      # optional: keep tables on disk between runs
      threads: 4
```

With `path`, the target's catalog is a directory catalog, so a table built by `dbt run` is still there for the next invocation. Without it, the catalog is in memory and lasts as long as the dbt process. Every dbt thread shares one session per target, so a model one thread builds is visible to the next.

## How models materialize

A `table` model runs `create or replace table warehouse.analytics.<model> as (...)` and lands in the catalog. A `view` model runs `create or replace view <model> as (...)`. Both replace in place. dbt's default materializations build an intermediate relation and rename it over the target, and Batcher has no rename.

dbt's generic and singular tests are `SELECT` statements, and they run unchanged. Listing schemas and relations, reading a relation's columns, and creating or dropping schemas and relations are answered from the session's catalog.

## Requirements and limitations

There are no transactions. The adapter sends no `BEGIN` or `COMMIT`, and each statement takes effect when it runs.

Batcher keeps views in the session rather than in a catalog. So a view renders as its bare name, exists only inside the dbt process that built it, and two views of one name in different schemas collide. `dbt build` runs and tests in one process, so views are tested there. A separate `dbt test` process sees the tables but not the views.

Seeds, snapshots, incremental models, `dbt docs generate`, and query cancellation aren't part of the pilot. A relation rename is refused with an error rather than emulated.

## See also

- {doc}`/api/relational/sessions-and-catalogs`: the catalogs a profile's `database` names.
- {doc}`dbapi`: the connection each dbt thread holds.
