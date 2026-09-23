# Map columns

This page describes how to build a map column and how to read one back, using {py:func}`map_from_arrays <batcher.map_from_arrays>` and the {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>` accessor.

A map holds key-value pairs inside one column, and each row carries its own set of keys. That makes it the right shape for sparse attributes, such as tags, labels, or HTTP headers, where a hundred keys are possible and each row carries three. A struct fixes its fields in the type, so `.struct.field("x")` is a name resolution and asking for a field the type lacks is an error. A map's keys are data, so `.map.get("x")` is a per-row lookup and a missing key is an ordinary null.

```python
import batcher as bt
```

## Building a map

{py:func}`map_from_arrays <batcher.map_from_arrays>` pairs a column of key lists with a column of value lists, one map per row. The name is Spark's. SQL spells the same constructor `map(keys, values)`, as DuckDB does.

```python
events = bt.from_pydict(
    {
        "id": [1, 2],
        "k": [["source", "region"], ["source"]],
        "v": [["web", "eu"], ["mobile"]],
    }
)
attrs = events.select("id", attributes=bt.map_from_arrays(bt.col("k"), bt.col("v")))
print(attrs.to_pydict()["attributes"])
# [[('source', 'web'), ('region', 'eu')], [('source', 'mobile')]]
```

The two lists are paired positionally, so they must be the same length in every row.

## Reading a map

The {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>` accessor reads a map column, whether it was built above or arrived from Arrow. A missing key gives null rather than an error, so a lookup composes into a filter like any other expression.

```python
read = attrs.select(
    "id",
    keys=bt.col("attributes").map.keys(),
    source=bt.col("attributes").map.get("source"),
    has_region=bt.col("attributes").map.contains("region"),
    n=bt.col("attributes").map.len(),
)
print(read.to_pydict()["source"], read.to_pydict()["has_region"])
# ['web', 'mobile'] [True, False]
```

`keys`, `values` and `entries` return lists, `get` returns the bare value, `len` counts the entries, and `contains` is a predicate. A null map row stays null through all of them, while an empty map returns an empty list, so a filter on `.map.len() == 0` selects only the empty ones.

## Turning entries into rows

{py:meth}`.map.keys() <batcher.plan.expr_ir.namespaces.collections._MapNamespace.keys>` and {py:meth}`.map.values() <batcher.plan.expr_ir.namespaces.collections._MapNamespace.values>` return lists that are positionally aligned with each other. When a key has to travel with its value, reach for {py:meth}`.map.entries() <batcher.plan.expr_ir.namespaces.collections._MapNamespace.entries>` instead of zipping those two lists. It returns one `{key, value}` struct per entry, so the pairing is structural and survives anything that reorders the list later.

Explode the entry list and each map row becomes one row per key, which is the usual way to group or join on keys that vary by row:

```python
long = (
    attrs.select("id", e=bt.col("attributes").map.entries())
    .explode("e")
    .select("id", key=bt.col("e").struct.field("key"), value=bt.col("e").struct.field("value"))
)
print(long.to_pydict())
# {'id': [1, 1, 2], 'key': ['source', 'region', 'source'], 'value': ['web', 'eu', 'mobile']}
```

A map cannot be a grouping, join, or distinct key itself, because its entries have no canonical order. Key on something derived from it, such as the exploded `key` column above. {doc}`The type system <type-system>` explains the rule.

## Reading a map that came from Arrow

Arrow needs the map type stated explicitly. A dict passed to {py:func}`from_pydict <batcher.from_pydict>` infers a *struct*, not a map, so a column read this way must be built with `pa.map_`.

```python
import pyarrow as pa

table = pa.table(
    {
        "id": pa.array([1, 2]),
        "attributes": pa.array(
            [[("source", "web")], [("source", "mobile"), ("region", "us")]],
            type=pa.map_(pa.string(), pa.string()),
        ),
    }
)
from_arrow = bt.from_arrow(table).select("id", n=bt.col("attributes").map.len())
print(from_arrow.to_pydict()["n"])
# [1, 2]
```

## Requirements and limitations

Three inputs raise rather than being coerced, matching DuckDB. Each one has a plausible wrong answer that a permissive constructor would return instead, which is why the refusal is worth more than the convenience:

| Input | Why it is refused |
|---|---|
| A null key | Arrow's map key field is non-nullable, so the pair has nowhere to go. |
| A duplicate key within one row | Keeping the first or the last is a guess, and the two differ. |
| Key and value lists of different lengths | Truncating to the shorter one silently drops data. |

A null *value* is legal, and a null list on either side yields a null map, which is distinct from the empty map. Duplicate detection is per row, so the same key appearing in two different rows is ordinary data.

One divergence from DuckDB is deliberate. `map(NULL, NULL)` with a bare untyped SQL `NULL` answers a null map in DuckDB and raises here, because an untyped `NULL` carries no key or value type to build the map's fields from and inventing one would put a guessed schema into the plan. A null *list column*, which does carry a type, behaves as DuckDB's does.

## See also

- {doc}`/user-guide/transform/columns/expression-accessors`: the general-purpose accessors this sits beside.
- {doc}`/user-guide/transform/columns/type-system`: nested types, and why a map cannot be a key.
- {doc}`/user-guide/transform/rows/transformations`: `explode` and `unnest` in full.
- [`examples/expr_collections/map_columns.py`](https://github.com/stephenoffer/batcher/blob/main/examples/expr_collections/map_columns.py): the same material as a runnable script, including the refusals.
