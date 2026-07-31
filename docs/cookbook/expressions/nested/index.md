# Lists, structs, and JSON

The nested types, which the engine handles in place rather than by exploding to rows and regrouping.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/expressions/nested/lists_basics` | Indexing, slicing, joining, and flattening |
| {doc}`/cookbook/expressions/nested/lists_transforms` | Transforming inside a list column, without exploding it first |
| {doc}`/cookbook/expressions/nested/lists_aggregate` | Reducing a list column to one value per row |
| {doc}`/cookbook/expressions/nested/lists_set_operations` | Union, intersection, difference, and overlap |
| {doc}`/cookbook/expressions/nested/lists_vectors` | Similarity, distance, and normalization on embedding vectors |
| {doc}`/cookbook/expressions/nested/structs_and_maps` | Nested records without flattening the table |
| {doc}`/cookbook/expressions/nested/json_columns` | Reading JSON held in a string column, without parsing it in Python |

```{toctree}
:hidden:

lists_basics
lists_transforms
lists_aggregate
lists_set_operations
lists_vectors
structs_and_maps
json_columns
```
