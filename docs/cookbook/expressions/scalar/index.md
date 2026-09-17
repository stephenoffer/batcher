# The scalar algebra

The expression core that every other accessor builds on: arithmetic, branching, null and type handling, column selectors, and the reductions that work across columns, down a column, or over a window. If you are new to the expression API, read numeric math and conditionals first. The nulls-and-casting and sorting recipes are the ones that save you from a quietly wrong report.

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
