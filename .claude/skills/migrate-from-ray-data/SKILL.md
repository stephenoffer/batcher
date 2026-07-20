---
name: migrate-from-ray-data
description: Port a Ray Data pipeline to Batcher's public Python API — the operation translation, the batch-vs-row and actor/GPU concept shifts, how Ray becomes scheduling-only under Arrow Flight, and an ordered recipe that verifies the ported script returns the same rows. Invoke when converting `ray.data` code to Batcher, or when asked how a Ray Data idiom is spelled here.
---

# Migrate a Ray Data pipeline to Batcher

Use this when you have a working `ray.data` pipeline — usually a read, some
`map_batches`, a model, and a write — and want it on Batcher. The operation names are
deliberately close, so the port reads easy; the parts that are *not* the same are the
execution model and the relational surface Ray Data does not have. Read
`docs/integrations/ray.md` first — it is the reference this skill operationalizes.

## Translation table

| Ray Data | Batcher | Note |
|---|---|---|
| `ray.init()` | *(nothing)* | single-node never imports Ray; cluster is auto-detected |
| `ray.data.read_parquet(p)` | `bt.read.parquet(p)` | lazy; projection + predicate pushdown |
| `ray.data.read_csv/json/...` | `bt.read.csv/json/...`, or `bt.read(p)` | format inferred from the path |
| `ray.data.from_arrow(t)` | `bt.from_arrow(t)` | also `from_pydict`/`from_pylist`/`from_numpy` |
| `ray.data.from_pandas(df)` | `bt.from_pandas(df)` | `from_polars`, `from_huggingface`, `from_torch` too |
| an existing `ray.data.Dataset` | `bt.from_ray_dataset(rds)` | on-ramp; streams one Arrow block per batch |
| `ds.map_batches(fn, batch_format="pyarrow")` | `ds.map_batches(fn)` | `fn` gets a `pyarrow.RecordBatch`; parallel by default |
| `ds.map_batches(Cls, concurrency=n, num_gpus=1)` | `ds.ml.map_batches(Cls, concurrency=n, num_gpus=1)` | class ⇒ loaded once per worker |
| `ds.map(fn)` | `ds.map(fn)` — **but prefer `map_batches`/`Expr`** | per-row Python is the slow path |
| `ds.flat_map(fn)` | `ds.flat_map(fn)` | |
| `ds.filter(lambda r: r["a"] > 1)` | `ds.filter(bt.col("a") > 1)` | expression, runs in Rust |
| `ds.select_columns([...])` | `ds.select(...)` | |
| `ds.add_column("c", fn)` | `ds.with_columns(c=expr)` | |
| `ds.drop_columns([...])` | `ds.drop(...)` | |
| `ds.groupby("k").sum("v")` | `ds.group_by("k").agg(total=bt.col("v").sum())` | full aggregate surface, not 5 fixed ones |
| `ds.sort("k")` | `ds.sort("k", descending=False)` | |
| *(no join)* | `ds.join(o, on="k", how="left")`, `ds.join_asof(...)` | Ray Data has no real join |
| *(no window)* | `bt.rank().over(partition_by=.., order_by=..)`, `ds.window(...)` | |
| *(no SQL)* | `bt.sql(q, t=ds)` / `ds.sql(q)` | same optimizer as the DataFrame API |
| `ds.unique("c")` / `ds.distinct()` | `ds.distinct()` / `ds.n_unique()` | |
| `ds.random_shuffle()` | `ds.shuffle(seed=...)` | |
| `ds.repartition(n)` | `ds.repartition(n)` | output layout: `num_files=`/`by=`/`target_size_mb=` |
| `ds.limit(n)` / `ds.take(n)` | `ds.limit(n)` / `ds.head(n)` | |
| `ds.count()` / `ds.show()` | `ds.count()` / `ds.show()` | |
| `ds.take_all()` | `ds.collect()` (Arrow) / `ds.to_pylist()` | |
| `ds.iter_batches()` | `ds.iter_batches(batch_size=...)` | Arrow batches, bounded memory |
| `ds.iter_torch_batches()` | `ds.ml.iter_torch_batches(...)` | |
| `ds.streaming_split(n)` | `ds.ml.stream_loader(world_size=, rank=, ...)` | deterministic, resumable, world-size independent |
| `ds.write_parquet(p)` | `ds.write.parquet(p)` / `ds.write(p, mode=...)` | atomic + `resume=True` |
| `ds.stats()` | `ds.stats()` | measured rows/time/bytes/spill + bottleneck |
| *(no plan)* | `ds.explain()` | there is an optimizer to explain |

## Conceptual shifts that actually bite

- **`map_batches` takes a whole Arrow batch, and that is the only good spelling.**
  `fn` receives a `pyarrow.RecordBatch` (`batch_format="pyarrow"` is the default, not an
  option you remember to pass) and returns one. There is no `"numpy"`/`"pandas"` default
  quietly converting under you.

  ```python
  import batcher as bt
  import pyarrow.compute as pc

  ds = bt.from_pydict({"city": ["NYC", "LA"], "amount": [10, 20]})

  def add_fee(batch):
      return batch.append_column("fee", pc.multiply(batch.column("amount"), 0.05))

  print(ds.map_batches(add_fee, input_columns=["city", "amount"]).to_pydict())
  # {'city': ['NYC', 'LA'], 'amount': [10, 20], 'fee': [0.5, 1.0]}
  ```

  `input_columns` must name **every** column `fn` reads: projection pushdown prunes the
  scan to that list, so an omission is a correctness bug, not a perf nit.
- **There is no per-row hot path.** `ds.map(fn)` exists for the familiar spelling and is
  Python-per-row — the thing this engine is built to avoid. Anything expressible as
  columns should be an `Expr` (`ds.filter(bt.col("a") > 1)`,
  `ds.with_columns(c=bt.col("a") * 2)`), which lowers to Rust and JIT-compiles.
- **`num_workers` defaults to `"auto"`, so a batch transform is already parallel.** Ray
  Data's single-threaded-unless-you-say-otherwise default is not reproduced. Threads only
  help a GIL-releasing `fn` (Arrow/NumPy/torch); pass `multiprocessing=True` for a
  CPU-bound pure-Python `fn`.
- **Actors become `ds.ml`.** A stateful class + `concurrency` + `num_gpus` maps onto
  `ds.ml.map_batches(Cls, concurrency=..., num_gpus=...)` (class, not instance — loaded
  once per worker). For the common model shapes there are named entry points:
  `ds.ml.infer(model, num_gpus=, concurrency=)`, `ds.ml.embed(model)`,
  `ds.ml.classify/extract/generate`, and `batcher.ml.llm_generate(..., engine=vllm_engine(...))`.
  Batch size and `num_gpus` adapt from measurements — do not port your hand-tuned values.
- **Ray is scheduling only.** Tasks, actors, placement groups, and small control messages
  go through Ray; bulk Arrow batches move worker-to-worker over Arrow Flight
  (`bc-transport`) with credit-based flow control, bypassing the object store. That is
  why there is no object-store memory proportion to tune — and why `ray.put` on a
  `RecordBatch` is exactly the tax the design removes.
- **Distribution is a keyword, not a program.** `ds.collect(distributed="auto")` (the
  default) uses Ray when it detects a multi-node cluster and runs in-process otherwise.
  `True`/`False` force it. The result is identical either way: the distributed path
  composes the *same* mergeable `partial → shuffle → combine → finalize` primitives.
- **You now have an optimizer, so stop pre-optimizing.** Manual column pruning, hand-split
  stages, and pre-filter-then-repartition dances exist because Ray Data has no plan.
  Write the query declaratively and check `ds.explain()`.

## Porting recipe

1. **Inventory the pipeline.** List the source, each `map_batches`/`map`, each actor
   class, and the sink. Note anything using Ray primitives directly (`ray.put`,
   `ray.remote`, `ObjectRef` passing) — that is scheduling code that should disappear, not
   be translated.
2. **Replace the source.** `ray.data.read_X` → `bt.read.X` on the same paths. If the
   pipeline starts from an existing `ray.data.Dataset` you cannot delete yet, bridge with
   `bt.from_ray_dataset(rds)` — but treat it as temporary: everything upstream of the
   hand-off still pays Ray Data's per-task and object-store cost.
3. **Demote relational work out of Python.** Every `map_batches` that only filters,
   projects, derives, or aggregates becomes `filter` / `select` / `with_columns` /
   `group_by().agg()`. This is the step that produces most of the speedup.
4. **Port the surviving `map_batches` functions.** Drop `batch_format=`, take a
   `pyarrow.RecordBatch`, add `input_columns=[...]`, and delete `num_cpus`/parallelism
   tuning unless a measurement justifies it.
5. **Port actors to `ds.ml`.** Stateful class → `ds.ml.map_batches(Cls, ...)`; a
   recognizable model → `ds.ml.infer` / `ds.ml.embed`. Keep `concurrency`/`num_gpus` only
   where the model genuinely pins them.
6. **Port the sink.** `ds.write_parquet(p)` → `ds.write.parquet(p)`, or
   `ds.write(p, mode="append"|"overwrite"|"error"|"ignore", partition_by=[...])`.
   `resume=True` skips committed shards on a re-run; `max_rows_per_file=` bounds files.
7. **Verify equivalence.** Run both pipelines on the same input and compare
   **order-independently** unless the query ends in an explicit `sort` — neither engine
   promises order. The in-repo pattern is `tests/differential/conftest.py::assert_same`
   (multiset comparison, tolerant of int↔float and float rounding), with
   `assert_same_ordered` when order is part of the contract. Mirror it:

   ```python
   import batcher as bt

   ported = bt.from_pydict({"k": ["a", "b"], "v": [1, 2]})
   got = sorted(map(tuple, zip(*ported.to_pydict().values())))
   want = sorted([("a", 1), ("b", 2)])
   assert got == want
   ```

8. **Read the plan, then the profile.** `print(ds.explain())` to confirm pushdowns landed,
   `ds.stats()` for measured per-operator rows/time/bytes/spill and the bottleneck.

## Gotchas / do-not

- **Do not keep `ds.map(fn)` from a `ds.map` port.** It compiles, it is correct, and it is
  the slowest thing in the pipeline. Vectorize to `map_batches` or an `Expr`.
- **Do not under-declare `input_columns`.** A column the `fn` reads but does not declare
  can be pruned out from under it — a silent wrong answer, not an error. If you are not
  certain what the ported function touches, leave `input_columns=None` (the default),
  which keeps every column alive; a missed pruning opportunity is the cheap failure.
- **Do not assume block order.** Ray Data block order is incidental and so is Batcher's.
  Add an explicit `ds.sort(...)` if order matters; otherwise compare as a multiset.
- **Do not `take_all()`/`collect()` a large result** to loop over it in Python. Use
  `iter_batches()`, or push the work into the plan.
- **Do not route bulk data through Ray.** No `ray.put` on a `RecordBatch`, no passing
  batches as `ObjectRef`s between stages. Ray moves paths; Flight moves batches.
- **Do not assume `fn` runs exactly once.** A preempted worker recomputes its partition,
  so a `map_batches` `fn` with an external side effect (vector-DB insert, REST POST) can
  apply it twice. Make the sink an upsert on a stable key; pure transforms are already safe.
- **Do not reach for `distributed=True` reflexively.** On one node the in-process engine
  wins by a wide margin and distribution costs a shuffle. Leave it on `"auto"`.
- **Do not port tuned knobs.** Batch size, `num_gpus`, and worker counts are adaptive here;
  a value copied from a Ray Data config usually pins the engine below what it would pick.

## See also

- `docs/integrations/ray.md` — scheduling vs data plane, Flight shuffle, cluster config,
  failure modes.
- `docs/migration/index.md` — the full mapping tables and the ML/batch-inference surface.
- `docs/user-guide/{udfs,transformations,performance,writing-data}.md`;
  `docs/deep-dives/shuffle-flight.md`.
- Skills: `migrate-from-spark` (the PySpark port), `run-quality-gate` (if the port changes
  repo code).
