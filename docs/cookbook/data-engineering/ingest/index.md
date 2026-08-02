# Getting data in

These recipes cover the arrival path: a source drops records somewhere, and you have to pick
them up exactly once, in the right order, including the ones that show up late.

Start from the complete ETL pipeline if you are building the path from nothing. The other
three are the problems that appear once it is running.

| Recipe | The problem |
|---|---|
| {doc}`ETL pipeline <etl-pipeline>` | A whole extract, transform, and load, small enough to read in one sitting |
| {doc}`Incremental ingest <incremental-ingest>` | Picking up only the files that arrived since last time |
| {doc}`CDC pipeline <cdc-pipeline>` | Applying a change feed of inserts, updates, and deletes to a table |
| {doc}`Late-arriving data <late-arriving-data>` | Events that happened on Tuesday and reached you on Thursday |

## See also

- {doc}`/cookbook/data-engineering/modeling/index`: shaping what you ingested into tables people query.
- {doc}`/user-guide/moving-data/index`: the reader and writer reference behind these recipes.

```{toctree}
:hidden:

etl-pipeline
incremental-ingest
cdc-pipeline
late-arriving-data
```
