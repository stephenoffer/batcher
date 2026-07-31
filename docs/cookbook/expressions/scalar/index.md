# The scalar algebra

The expression core: arithmetic, branching, null and type handling, and the two reductions that are not per-row.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/expressions/scalar/numeric_math` | Arithmetic and math functions on numeric columns |
| {doc}`/cookbook/expressions/scalar/conditionals` | Branching inside an expression, and the SQL null helpers |
| {doc}`/cookbook/expressions/scalar/nulls_and_casting` | The two places a pipeline quietly changes its answer |
| {doc}`/cookbook/expressions/scalar/column_selectors` | Naming columns by type or pattern instead of one at a time |
| {doc}`/cookbook/expressions/scalar/horizontal` | Reducing across columns instead of down rows |
| {doc}`/cookbook/expressions/scalar/aggregates` | Counts, positions, quantiles, and approximations |
| {doc}`/cookbook/expressions/scalar/window_functions` | Per-row values computed from a window of related rows |
| {doc}`/cookbook/expressions/scalar/sorting_and_ranking` | Sorting and ranking, including the edge cases that hide bugs |

```{toctree}
:hidden:

numeric_math
conditionals
nulls_and_casting
column_selectors
horizontal
aggregates
window_functions
sorting_and_ranking
```
