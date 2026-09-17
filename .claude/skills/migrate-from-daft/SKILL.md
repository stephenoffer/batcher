---
name: migrate-from-daft
description: Port a Daft workload to Batcher's public Python API — the relational verb mapping, Daft's URL/image/embedding expressions against Batcher's .image/.audio/.video/.json accessors and the batcher.ml inference pipeline, the UDF story, and Ray-backed distribution without the object store. Invoke when converting a Daft (especially multimodal or batch-inference) script to Batcher, or when asked for the Batcher equivalent of a Daft expression.
---

# Migrate from Daft

Daft and Batcher target the same shape of work: multimodal and ML-first pipelines
over a native columnar engine, distributed with Ray. The port is mostly mechanical.
Read `docs/getting-started/migration/daft/index.md` (every Daft name, generated from the
migration registry) and `docs/getting-started/migration/reading-and-writing.md` (the
`from_*`/`to_*` adapters)
and `docs/benchmarks/comparisons/vs-daft.md` (the scorecard) before promising a user a speedup.
Batcher wins multimodal ingest and top-N and ties aggregation. The join-heavy TPC-H
result is **hardware-dependent and has moved**: that page measures Daft ahead on a
16-core node, while a 96-core re-run at sf1 has Batcher ahead on 18 of the 19 queries
both engines answer. Quote the conditions, not a multiplier.

Every Batcher name below is verified against the live surface. If you need one this
skill doesn't list, check it — `python -c "import batcher as bt; print(bt.<name>)"`,
or `print([m for m in dir(bt.col('x').image) if not m.startswith('_')])` — never
invent it.

## When to use

- A Daft script (relational, multimodal, or batch-inference) needs to run on Batcher,
  or a ported pipeline needs its results proven equal to the Daft original.
- You are asked what a specific Daft expression maps to.

## Relational translation

Both engines are lazy: Daft builds a plan and runs on `collect()`/`show()`; Batcher's
`Dataset` does the same on `collect()`, `to_arrow()`, `to_pydict()`, `count()`,
`show()`, `iter_batches()`, or `write.*`. That model ports unchanged.

Every Daft 0.7.25 public name has one row in the migration registry
(`python/batcher/_internal/migration/data/daft/`), rendered into
`docs/getting-started/migration/daft/`: `dataframe.md` (`DataFrame`, `GroupedDataFrame`,
`Window`), `module.md` (the `daft` module: constructors, readers, session shortcuts, UDF
decorators), `functions-numeric.md`, `functions-strings.md`, `functions-temporal.md`,
`functions-nested.md` (on `Expression` and `daft.functions`), `udfs-ai-multimodal.md`,
`session-and-catalog.md`, and `types.md`. Read each row's status before renaming a call. A
`mismatch` returns a different answer under the Batcher spelling, such as `day_of_week`
numbering, 0-based `find`, or `decode_image` yielding pixels in Daft and a header
struct here. Fix a wrong row in the registry and run `just migration-docs`; don't restate it in
this skill.

## Multimodal and ML translation

This is where the two engines actually differ. Daft puts media work on `.url` and
`.image` expression namespaces; Batcher splits it between **typed accessors** for
pure per-value transforms and the **`ds.ml` pipeline** for anything that loads a
model or does network IO.

`udfs-ai-multimodal.md` maps each image, video, audio, file and AI function. A URL fetch is
`ds.ml.download(...)`, a Dataset-level stage rather than an expression, and a model call is
`ds.ml.infer` / `ds.ml.embed` / `ds.ml.generate`.

**Pass a class, not an instance**, to `ds.ml.infer` / `ds.ml.embed` /
`ds.map_batches`: the model is then constructed once per worker instead of being
pickled per batch. `num_gpus=` and `concurrency=` size the GPU actor pool; batch size
adapts under a VRAM cap rather than being a number you tune.

```python
import batcher as bt

# decode=True / size= appends a decoded `image` (H, W, 3) uint8 tensor column.
frames = (
    bt.read.images("s3://bucket/frames/", decode=True, size=(224, 224))
    .select("uri", "image")
    .ml.infer(Classifier, column="image", output_column="label", num_gpus=1, concurrency=4)
)
frames.write.parquet("s3://bucket/labels/")
```

## Scalar-function translation

The scalar functions are on the `functions-*.md` pages, one row per name on both `Expression`
and `daft.functions`.

**Do not reach for an epoch cast.** Daft's `timestamp_seconds` maps to `bt.from_epoch`, not to a
cast: `col("t").cast("timestamp")` compiles, runs, and is wrong, because Arrow reads a
bare integer as *microseconds*. `bt.from_epoch(c, "s")` is the port.

## The UDF story

`module.md` has the rows for `udf`, `cls`, `func`, and `method`. None of them is canonical:
Daft's decorators produce expression-level UDFs, and Batcher's `@bt.udf` wraps a batch function
applied to a whole `Dataset`, so a Daft UDF ports to `ds.map_batches(fn)` or, for a model,
`ds.ml.infer`.

`ds.map_batches(fn)` hands `fn` a pyarrow `RecordBatch` and expects one back —
vectorized Arrow compute inside, never a row loop. Declare `input_columns` (what you
read, so projection pushdown can prune the scan) and `output_columns` (the new
schema). Getting `input_columns` wrong is a **correctness** bug, not a slow query:
an undeclared column can be pruned out from under `fn`. Leave it `None` if unsure.

```python
import batcher as bt
import pyarrow.compute as pc


@bt.udf(output_columns=["price", "qty", "total"], input_columns=["price", "qty"])
def add_total(batch):
    total = pc.multiply(batch.column("price"), pc.cast(batch.column("qty"), "float64"))
    return batch.append_column("total", total)


out = add_total(bt.read.parquet("/tmp/orders"))
```

## Distribution: same Ray, different data plane

Both engines schedule on Ray. Daft's Ray runner moves partitions through the **Ray
object store**; Batcher uses Ray for task/actor scheduling and control-plane metadata
only, and moves bulk Arrow batches over **Arrow Flight (`bc-transport`) with
credit-based flow control**, bypassing the object store entirely. Practically:

- No `daft.context.set_runner_ray()` to call and no object-store memory proportion to
  tune. `ds.collect(distributed=True)` (or `"auto"`) is the switch; `num_workers=` /
  `num_partitions=` are the knobs, and `spill=True` keeps aggregation/join/sort inside
  a memory bound instead of failing.
- The distributed result is identical to single-node **by construction** — the same
  mergeable `partial → combine → finalize` operators run in both cases, not a second
  distributed implementation.

## Porting recipe

1. **Split the script into relational vs model stages.** The relational half ports
   verb-for-verb from the first table; the model half moves onto `ds.ml`.
2. **Run the codemod first.** `python -m batcher.migrate --from daft --to batcher <paths>`
   prints a diff and changes nothing until you add `--write`. The `daft` direction may not be
   implemented yet: the command then raises `ConfigError` naming the directions that are, and
   the `daft/index.md` page says the same. Port by hand from the generated pages in that case.
   Either way, run `python -m batcher.migrate --from batcher --to batcher <paths>` over any
   code that already calls Batcher, so no removed Batcher spelling survives the port.
3. **Replace the readers.** `daft.read_*` → `bt.read.<fmt>`; for media, prefer the
   dedicated `bt.read.images` / `bt.read.video` / `bt.read.point_cloud` over a manual
   path scan plus download.
4. **Convert URL fetch → decode → transform.** `.url.download()` becomes
   `ds.ml.download(...)`; the decode/resize/tensor chain stays as `.image` /`.audio` /
   `.video` accessor expressions, which lower to Rust and stay vectorized.
5. **Convert model UDFs to `ds.ml.infer` / `ds.ml.embed`**, passing the model *class*
   with `num_gpus=` and `concurrency=`. Only fall back to `ds.map_batches` when the
   stage is not a model call.
6. **`select` down to the columns each opaque stage reads, before that stage.** The
   optimizer cannot prune across a Python callback.
7. **Verify** (below), then read `ds.explain()` and `ds.stats()` — `stats()` reports
   measured rows/time/bytes/spill per operator and names the bottleneck.

### Verifying equivalence

Compare **order-independently** unless the query has an explicit `sort` (a plan with
no `ORDER BY` has no defined row order in either engine; conversely, a sorted query
must be compared *in order*, because an order-independent comparison cannot see a
sort bug). The in-repo pattern is `tests/differential/conftest.py::assert_same` —
normalize to pyarrow, rows to tuples, sort by a total order, compare as multisets
with int↔float and float-rounding tolerance. Reuse it:

```python
import batcher as bt

daft_rows = sorted(tuple(r.values()) for r in daft_df.to_pylist())
bt_rows = sorted(tuple(r.values()) for r in ported_ds.to_pylist())
assert daft_rows == bt_rows
```

For a multimodal pipeline, comparing decoded pixels is fragile — gate on **frame
counts and output tensor shapes** (that is what `benchmarks/scenarios/image_decode.py`
does), plus an exact comparison of the relational columns.

### When the two engines disagree, check which one is right

A ported query whose results differ is not automatically a porting bug. Two places where
Daft is the one that departs from SQL, both found by running it against DuckDB:

- **`sum(x) OVER (PARTITION BY k ORDER BY o)`.** SQL's default frame with an `ORDER BY`
  is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` — a *running* aggregate. Daft
  0.7.21 applies the whole partition instead, so on `v = [10, 20, 30]` it returns
  `60, 60, 60` where DuckDB and Batcher return `10, 30, 60`. A port of such a query will
  legitimately produce different numbers, and the new ones are the correct ones. Tell the
  user; do not "fix" the port to reproduce the old output.
- **TPC-H q6's float folding.** Daft folds `0.06 + 0.01` to `0.06999999999999999` and
  drops every `l_discount = 0.07` row (see `docs/benchmarks/comparisons/vs-daft.md`).

The general move: when a ported result differs, run the same query through DuckDB before
assuming the port is wrong.

## Going back

`docs/getting-started/migration/daft/leaving-batcher.md` maps Batcher spellings to Daft for the
rows where both engines compute the same thing; anything else is on the forward pages with its
difference. `python -m batcher.migrate --from batcher --to daft <paths>` is the reverse codemod,
subject to the same implemented-directions check. Batcher has no Daft exporter or importer
(`to_daft` and `from_daft` are registry gaps), so hand data back through Arrow, with
`daft.from_arrow(ds.to_arrow())`, or through Parquet files.

## Gotchas / do-not

- **Do not port a per-row Python UDF as a per-row Python UDF.** `ds.map` / `flat_map`
  / `@bt.udf(per_row=True)` cost a Python object per row and make the stage opaque to
  the optimizer. Reach for an `Expr` first, `map_batches` second.
- **Do not assume ordering without `sort`.** Neither engine preserves input order
  through a group-by, a shuffle, or a parallel scan.
- **Do not `collect()` a large multimodal result**, and do not insert a `collect()`
  mid-chain to "check" something — decoded frames are enormous, and each `collect` is
  a materialization barrier hiding the rest of the plan from the optimizer. Stream with
  `ds.iter_batches()` or terminate into `ds.write.*`.
- **Do not declare `input_columns` loosely.** An undeclared-but-read column can be
  pruned from the scan; that is a wrong answer, not a slow one. Unsure what the ported
  expression touches? Leave it `None` (the default) — every column stays alive.
- **Do not promise a blanket speedup**, in either direction. The join-heavy TPC-H
  comparison depends on core count and has reversed between the two machines measured;
  per-batch Python UDFs are still Daft's by ~2×. Quote `docs/benchmarks/comparisons/vs-daft.md`
  *with its conditions*, never a bare multiplier.
- **Do expect a ported SQL workload to change answers where Daft was wrong.** On TPC-H
  at sf1 the harness's DuckDB gate fails Daft on q6, q15 and q18 and Daft errors on q21
  and q22 — 5 of 22. If a ported query's numbers move, check DuckDB before assuming the
  port broke it.

## See also

- `docs/getting-started/migration/daft/index.md` — every Daft name, with its Batcher spelling,
  status, and what differs; `docs/getting-started/migration/reading-and-writing.md` — the
  `from_*`/`to_*` adapters.
- `docs/benchmarks/comparisons/vs-daft.md`, `docs/benchmarks/results/multimodal-ingest.md` — the measured
  comparison and the image/point-cloud pipelines.
- `docs/api/models/ml.md`, `docs/ml/` — the inference, embedding, and training-feed surface.
- `docs/user-guide/transform/columns/udfs.md` — when a UDF is justified and what it costs.
- `docs/architecture/deep-dives/distribution/shuffle-flight.md`, `docs/architecture/deep-dives/operators/mergeable-algebra.md` — why the
  distributed result equals the single-node one, without the object store.
- `/migrate-from-polars-or-pandas`, `/migrate-from-spark` — the sibling migration skills.
