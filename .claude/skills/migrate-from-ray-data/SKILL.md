---
name: migrate-from-ray-data
description: Port a Ray Data pipeline to Batcher's public Python API — where the generated name-by-name reference lives, the object-store-free data plane, the defaults that silently change a result (batch format, save mode, shuffle seed, image decode, aggregate output names), the resource-parameter gaps on map_batches, and a recipe that ends by proving the port returns the same rows. Invoke when converting a ray.data script to Batcher, when asked for the Batcher equivalent of a Ray Data call, or when moving a Batcher pipeline back to Ray Data.
---

# Migrate from Ray Data

Ray Data and Batcher are both lazy Python APIs over Arrow batches that scale from a laptop to a
Ray cluster, so most verbs have a Batcher spelling. The port goes wrong in the defaults: a call
that renames cleanly can hand your function a different batch type, overwrite where Ray Data
appended, or name its output column differently. The user-facing page is
`docs/getting-started/migration/ray-data.md` (the concepts), and the name-by-name reference is
generated into `docs/getting-started/migration/ray-data/`. This skill is the procedure around
both, and must never contradict them.

## When to use

- A `ray.data` script (ETL, batch inference, or a training feed) needs to run on Batcher.
- You are asked what a specific Ray Data call is spelled as here.
- A ported pipeline needs its results proven equal to the Ray Data original, or a Batcher
  pipeline needs to go back to Ray Data.

Not for: taking an already-ported Batcher pipeline onto a cluster (that is
`run-a-distributed-job`) or model-serving detail (`build-an-ml-pipeline`).

## The one shift: bulk data leaves the object store

In Ray Data a `Dataset` is blocks held as Ray objects, so shuffles, splits, and repartitions move
blocks through the object store. Batcher uses Ray for scheduling only and moves Arrow batches
between workers over Arrow Flight with credit-based flow control. There is no object store to
size, no `ray.init()` to call before a single-node run, and distribution is an argument to the
terminal call (`ds.collect(distributed=True)`) rather than a property of the dataset. The
block and object-ref surface (`num_blocks`, `get_internal_block_refs`, `to_arrow_refs`,
`from_blocks`, ...) is `out_of_scope` in the registry for that reason.

## Where the name mappings live

Every Ray Data 2.58 public name has one row in the migration registry
(`python/batcher/_internal/migration/data/ray_data/`), rendered into
`docs/getting-started/migration/ray-data/`:

- `dataset.md`: `Dataset`, `GroupedData`, `DataIterator`.
- `expressions.md`: `ray.data.expressions.Expr` and its `.str`/`.list`/`.dt`/`.struct`/`.map`
  namespaces, plus `ray.data.aggregate`.
- `io.md`: the `ray.data` module (readers, `from_*` constructors, strategies) and the
  `Datasource`/`Datasink` protocols.
- `udfs-ai-multimodal.md`: `ray.data.llm` and `ray.data.preprocessors`.
- `execution-context.md`: `DataContext` fields and `ExecutionOptions`.
- `leaving-batcher.md`: Batcher spellings back to Ray Data.

Ray Data has more `mismatch` rows than canonical ones, so **read the status before renaming a
call**. Do not restate a mapping in this skill: fix the registry row and run
`just migration-docs`.

## Defaults that silently change a result

Each of these is a `mismatch` row: the Batcher spelling runs, raises nothing, and answers
differently. Check a port for every one of them.

- **Batch format.** Ray `map_batches` and `map_groups` hand `fn` `{column: ndarray}` by default;
  Batcher hands it a `pyarrow.RecordBatch` (`batch_format="pyarrow"`). Pass
  `batch_format="numpy"` to keep a NumPy function working, or rewrite it over Arrow.
- **Iteration.** Ray `iter_batches` yields `{column: ndarray}` batches of 256 rows with nulls as
  NaN; Batcher `ds.iter_batches()` yields Arrow record batches of the engine's size and has no
  `batch_format=`. `ds.ml.to_numpy_batches(batch_size=256)` is the closest port. Ray
  `iter_rows` yields dicts; Batcher yields tuples unless `named=True`.
  `ds.ml.iter_torch_batches` defaults `batch_size=None` and `prefetch_batches=2` where Ray uses
  256 and 1, so pass both.
- **Save mode.** Ray `write_parquet`/`write_csv` append into a directory by default; Batcher
  overwrites by default, and its file sinks do not support `append`. Pass `mode=` explicitly.
- **Shuffle.** Ray `random_shuffle()` draws a new permutation on every execution;
  `ds.shuffle(seed=0)` returns the same permutation every run. Pass a fresh seed per epoch.
- **Sampling.** Ray `random_sample` keeps each row independently; Batcher `sample(fraction)`
  keeps rows by a seeded hash of their values, so duplicate rows are kept or dropped together.
- **Image decode.** Ray `read_images` decodes into an `image` array column by default;
  `bt.read.images` defaults to `decode=False`. Pass `decode=True`.
- **JSON.** Ray `read_json` reads JSON documents, including a top-level array; `bt.read.json`
  reads newline-delimited JSON only.
- **Aggregates.** Ray `aggregate`, and `sum`/`mean`/`min`/`max` over several columns, are eager
  and key results as `"sum(x)"`; Batcher `agg` is lazy and names each column after its alias.
  `GroupedData.count()` adds a `count()` column of rows per group, where Batcher
  `GroupBy.count` counts non-null values per column; `group_by(k).len(name="count()")` is the
  row count.
- **Eager calls.** `take`, `take_all`, `take_batch`, `unique(column)`, and `materialize` execute
  immediately in Ray. Batcher is lazy until a terminal, so `take(n)` is
  `ds.limit(n).to_pylist()` and `materialize()` is `ds.cache()`, which fills on the first
  terminal rather than now.
- **Splits.** `split_at_indices`, `split_proportionately`, `split`, and `train_test_split`
  materialize in Ray; Batcher's parts are lazy and each re-executes the input, so `cache()`
  first when you consume every part. `ds.ml.train_test_split` assigns rows by a seeded hash,
  where Ray's default `shuffle=False` takes the last rows in dataset order.
- **Expressions.** Ray `round` rounds half to even; Batcher rounds half away from zero. Ray
  `str.find` is 0-based and returns -1; `str.position` is 1-based and returns 0. Ray
  `str.slice`/`list.slice` take a stop index; Batcher takes a length. `bt.range(n)` names its
  column `value`, where Ray names it `id`.
- **Configuration.** `DataContext.get_current().x = ...` mutates a process singleton; Batcher's
  `Config` is frozen and applied through `bt.set_config` / `bt.config_context`.

## Resource parameters are the main gap

`map_batches`, `map`, `flat_map`, `filter`, and `map_groups` have `param` or `mismatch` rows for
the Ray resource parameters. Batcher's `map_batches` takes `num_gpus=`, `concurrency=` as an int
or a `(min, max)` pair, and `fn_args`/`fn_kwargs`/`fn_constructor_args`/`fn_constructor_kwargs`;
the rows list what is still missing, including `num_cpus`, `memory`, `compute=ActorPoolStrategy`,
the three-tuple `concurrency`, and `ray_remote_args`. A class passed as `fn` is constructed once
per worker, which is the port of a Ray callable-class UDF. For a model rather than arbitrary
code, `ds.ml.infer` is the shorter path.

## Porting recipe

1. **Inventory the script.** List every read, every `map_batches`/`map`/`filter` and the batch
   format each function expects, every eager call (`take*`, `materialize`, `aggregate`,
   `unique`), and every write with its implied save mode.
2. **Run the codemod first.** `python -m batcher.migrate --from ray_data --to batcher <paths>`
   prints a diff and changes nothing until you add `--write`. The `ray_data` direction may not
   be implemented yet: the command then raises `ConfigError` naming the directions that are, and
   the `ray-data/index.md` page says the same. Port by hand from the generated pages in that
   case. Either way, run `python -m batcher.migrate --from batcher --to batcher <paths>` over
   any code that already calls Batcher, so no removed Batcher spelling survives the port.
3. **Drop the cluster setup.** Delete `ray.init()` and `DataContext` tuning for a single-node
   run; the `execution-context.md` rows say which settings have a `bt.Config` counterpart.
4. **Port the readers.** `ray.data.read_*` becomes `bt.read.<fmt>`. Keep the paths identical,
   and apply the image and JSON defaults above.
5. **Port the verbs top-down** from `dataset.md`. Prefer an expression over a callback wherever
   Ray accepts either: `ds.filter(bt.col("amount") > 10)` pushes into the scan, and a lambda
   cannot.
6. **Port the UDFs.** Set `batch_format=` to what each function expects, pass the class rather
   than an instance, and declare `input_columns` only when you are sure of every column the
   function reads, because an undeclared column can be pruned out from under it.
7. **Port the eager calls and the writes.** Replace each eager call with a `limit` plus a
   terminal, and pass `mode=` on every write.
8. **Verify** (below), then read `ds.explain()` and `ds.stats()`.

### Verifying equivalence

Compare results, not plans. `ds.equals(other)` executes both sides and ignores row order by
default; pass `ordered=True` only after an explicit `sort`, because an order-independent
comparison cannot see a sort bug. Against the Ray Data original, collect both sides and compare
as multisets of tuples, the way `tests/differential/conftest.py::assert_same` does:

```python
import batcher as bt

ray_rows = sorted(tuple(r.values()) for r in ray_ds.take_all())  # doctest: +SKIP
bt_rows = sorted(tuple(r.values()) for r in ported.to_pylist())
assert ray_rows == bt_rows
```

Then check that the distributed result matches the single-node one, which is the property
Batcher holds itself to: `ported.collect(distributed=True)` against `ported.collect()`.

## Going back

`docs/getting-started/migration/ray-data/leaving-batcher.md` maps Batcher spellings to Ray Data
for the rows where both engines compute the same thing; anything else is on the forward pages
with its difference. `python -m batcher.migrate --from batcher --to ray_data <paths>` is the
reverse codemod, subject to the same implemented-directions check. Data moves both ways without
a file: `bt.from_ray_dataset(ds)` reads a Ray `Dataset`, and `ds.to_ray_dataset()` returns one.
Apply the defaults list above in reverse, such as restoring Arrow as the batch format where a
function now expects it, and passing `mode=` on every Ray Data write.

## Gotchas / do-not

- **Do not rename `map_batches` and stop.** The function now receives Arrow, not NumPy, and the
  first sign is a wrong answer or an exception deep inside user code.
- **Do not assume a write appends.** Batcher overwrites by default and rejects `append` on file
  sinks.
- **Do not reuse one shuffle seed across epochs** when the original reshuffled every epoch.
- **Do not port `num_blocks`, `*_refs`, or object-store sizing.** There is nothing to port them
  to; stream with `ds.iter_batches()` and read measured execution from `ds.stats()`.
- **Do not `collect()` mid-chain** to imitate `materialize()`. Use `ds.cache()` where a result
  is consumed more than once, and nowhere else.

## See also

- `docs/getting-started/migration/ray-data.md` — the concepts, splitting, and batch inference.
- `docs/getting-started/migration/ray-data/index.md` — every Ray Data name, with its Batcher
  spelling, status, and what differs.
- `docs/getting-started/migration/differences.md` — laziness, write-mode defaults, integer
  overflow.
- `docs/integrations/compute/ray.md` — cluster setup and the Flight shuffle.
- Skills: `run-a-distributed-job`, `build-an-ml-pipeline`, `migrate-from-daft`.
