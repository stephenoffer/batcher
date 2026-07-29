# Dataset cookbook

The Dataset verbs: joins, grouping, reshaping, deduplication, sampling, profiling, null handling, and the `meta` accessor.

Every page here embeds a complete, self-contained script from the
[`examples/dataset/`](https://github.com/batcher/batcher/tree/main/examples/dataset) directory.
The scripts build their own in-memory data and assert on their own output, and
`tests/docs/test_examples.py` runs all of them, so a page that stops matching the
engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`deduplication` | Deduplication: exact keys, whole rows, and keeping a chosen survivor |
| {doc}`dq_contracts` | Data-quality contracts: validate, fail, drop, or quarantine |
| {doc}`grouping` | Grouping: agg, multi-key rollups, and the cube/rollup/grouping-set variants |
| {doc}`iteration` | Getting results out: batches, rows, slices, and the single-value cases |
| {doc}`joins` | Join types, key spellings, and the as-of join for time series |
| {doc}`meta_columns` | Profiling one column: bounds, uniqueness, nulls, and constancy |
| {doc}`meta_comparison` | Asking about a join before running it, and reading approximate statistics |
| {doc}`meta_predicates` | Cheap yes/no questions about the data, and the column-check shorthands |
| {doc}`meta_schema` | Asking about a dataset's shape without executing it |
| {doc}`null_handling` | Dataset-level null handling: dropping, filling, and counting missing values |
| {doc}`profiling` | Profiling a table you have just been handed |
| {doc}`reshaping` | Reshaping: pivot, unpivot, explode, unnest, and set operations |
| {doc}`sampling_and_splits` | Sampling and splitting: reproducible subsets that do not leak |
| {doc}`sql_interface` | SQL over the same engine, and mixing SQL with DataFrame verbs |

```{toctree}
:hidden:

deduplication
dq_contracts
grouping
iteration
joins
meta_columns
meta_comparison
meta_predicates
meta_schema
null_handling
profiling
reshaping
sampling_and_splits
sql_interface
```
