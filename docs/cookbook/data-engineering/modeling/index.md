# Shaping the tables

These recipes cover turning what arrived into tables people can query: combining sources,
absorbing a schema someone changed without telling you, and keeping a history of a dimension
that moves.

| Recipe | The problem |
|---|---|
| {doc}`Multi-source join <multi-source-join>` | Facts in Parquet, a dimension in a CSV, and a lookup in a database |
| {doc}`Schema evolution <schema-evolution>` | An upstream column appears, disappears, or changes type |
| {doc}`Slowly changing dimensions <slowly-changing-dimensions>` | Keeping the old row when an attribute changes, rather than overwriting it |

## See also

- {doc}`/cookbook/data-engineering/maintenance/index`: keeping those tables correct and cheap over time.
- {doc}`/user-guide/moving-data/lakehouse`: the transactional table behavior these recipes rely on.

```{toctree}
:hidden:

multi-source-join
schema-evolution
slowly-changing-dimensions
```
