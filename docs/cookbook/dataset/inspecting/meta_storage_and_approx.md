# Storage layout and sketches

`ds.meta.storage` says what a scan would read before it reads anything: the files, the row groups, the recorded bytes, and the partition keys. `ds.meta.approx` reads the sketches earlier runs recorded, and the `ds.meta.col(...).check` membership questions answer from the column bounds whenever those bounds rule a value out.

The script writes a day-partitioned Parquet tree in twelve small appends and lists its files with `num_files()` and `files()`. It then runs the small-files check: the average recorded bytes per file says whether the directory needs compacting before anyone scans it repeatedly. The membership checks `contains`, `never_equals`, `any_in`, `none_in`, and `may_contain` run against the same tree, and a comparison with a value of the wrong type raises `PlanError`.

The last part compacts the tree into one file with `ds.write.parquet`, runs two ordinary queries on it, and reads the sketches they recorded back through `top_k`, `histogram`, and `selectivity`. Each estimate is asserted against the exact answer computed from the same rows. Behind a `map_batches` stage the planner cannot estimate anything, so `rows()` and `selectivity()` return `None` rather than `0.0`.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/meta_storage_and_approx.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_storage_and_approx.py
```

## See also

- {doc}`/user-guide/analyze/metadata-shortcuts`: the full `ds.meta` namespace, including how it behaves on a cluster.
- {doc}`/cookbook/dataset/inspecting/meta_comparison`: sizing a join before running it.
- {doc}`/cookbook/dataset/inspecting/meta_predicates`: cheap yes/no questions, and the column-check shorthands.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
