# Lists, structs, and JSON

Lists, structs, maps, and JSON text, all handled in place by the engine rather than by exploding to rows and regrouping. Start with list basics for indexing and `explode`, then list transforms for mapping over elements with an expression. Is your list column an embedding? Read the vectors recipe.

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
