---
name: migrate-from-daft
description: Port a Daft workload to Batcher's public Python API — the relational verb mapping, Daft's URL/image/embedding expressions against Batcher's .image/.audio/.video/.json accessors and the batcher.ml inference pipeline, the UDF story, and Ray-backed distribution without the object store. Invoke when converting a Daft (especially multimodal or batch-inference) script to Batcher, or when asked for the Batcher equivalent of a Daft expression.
---

# Migrate from Daft

Daft and Batcher target the same shape of work: multimodal and ML-first pipelines
over a native columnar engine, distributed with Ray. The port is mostly mechanical.
Read `docs/migration/transforming.md` and `docs/migration/reading-and-writing.md`
(the mapping tables and the `from_*`/`to_*` adapters)
and `docs/benchmarks/vs-daft.md` (the scorecard) before promising a user a speedup.
Batcher wins multimodal ingest and top-N and ties aggregation. The join-heavy TPC-H
result is **hardware-dependent and has moved**: that page measures Daft ahead on a
16-core node, while a 96-core re-run has Batcher ahead on 18 of the 19 queries both
engines answer (`docs/internals/daft_parity_ledger.md`). Quote the conditions, not a
multiplier.

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

| Daft | Batcher |
|---|---|
| `daft.read_parquet(p)` | `bt.read.parquet(p)` |
| `daft.read_warc(p)` | `bt.read.warc(p)` — one row per crawl record; `.warc.gz` transparent |
| `daft.read_huggingface("user/ds")` | `bt.from_huggingface(datasets.load_dataset("user/ds"))` — load first, then wrap |
| `daft.from_pydict(d)` | `bt.from_pydict(d)` |
| `daft.col("a")` | `bt.col("a")` |
| `df.select("a", "b")` | `ds.select("a", "b")` |
| `df.with_column("c", e)` | `ds.with_columns(c=e)` |
| `df.where(pred)` | `ds.filter(pred)` |
| `df.groupby("k").agg(...)` | `ds.group_by("k").agg(total=col("v").sum())` |
| `df.join(o, on="k")` | `ds.join(o, on="k", how="inner")` |
| `df.sort("a", desc=True)` | `ds.sort("a", descending=True)` |
| `df.limit(n)` | `ds.limit(n)` |
| `df.distinct()` | `ds.distinct()` |
| `df.explode("c")` | `ds.explode("c")` |
| `df.into_partitions(n)` | `ds.repartition(n)` |
| `df.collect()` | `ds.collect()` (pyarrow `Table`) |
| `df.to_arrow()` / `.to_pandas()` | `ds.to_arrow()` / `ds.to_pandas()` |
| `df.explain()` | `ds.explain()` (plus `ds.stats()` for *measured* per-op cost) |

## Multimodal and ML translation

This is where the two engines actually differ. Daft puts media work on `.url` and
`.image` expression namespaces; Batcher splits it between **typed accessors** for
pure per-value transforms and the **`ds.ml` pipeline** for anything that loads a
model or does network IO.

| Daft | Batcher |
|---|---|
| `col("url").url.download()` | `ds.ml.download("url", output_column="bytes")` |
| `col("bytes").image.decode()` | `col("bytes").image.decode()` — a struct, not just dimensions |
| `image_width(c)` / `image_height(c)` / `image_channel(c)` / `image_mode(c)` | one `col("c").image.decode()`, then `.struct.field("width" \| "height" \| "channels" \| "mode")`. Daft re-reads the header per call; Batcher reads it once |
| `col("img").image.resize(w, h)` | `col("img").image.resize(w, h)` |
| `crop(c, x, y, w, h)` | `col("c").image.crop(x, y, w, h)` — clips at the edge; `center_crop` is the padding one |
| `encode_image(c, fmt)` | `col("c").image.encode(fmt)` — `png`/`jpeg`/`bmp`/`gif` |
| `convert_image(c, mode)` | `col("c").image.convert(mode)` — `L`/`LA`/`RGB`/`RGBA` |
| image → tensor for a model | `col("img").image.to_tensor()` |
| audio decode / resample | `col("a").audio.decode()`, `.audio.resample(...)`, `.audio.to_waveform()` |
| video decode | `col("v").video.decode()` |
| `daft.read_parquet` over image paths | `bt.read.images(path, decode=True, size=(224, 224))`, `bt.read.video(...)`, `bt.read.point_cloud(...)` |
| `col("j").json.query(...)` | `col("j").json.extract_string(p)` / `extract_int` / `extract_float` / `extract_bool` |
| `json_array_length(c, p)` / `json_object_keys(c, p)` | `col("c").json.array_length(p)` / `.json.keys(p)` |
| `json_tuple(c, *keys)` | `col("c").json.values(p)` then `explode`, or one `extract_*` per key |
| — | `col("c").json.type_of(p)` and `.json.exists(p)`: route a field whose type varies, and tell an absent key from a JSON `null` |
| `.struct.get("f")` / map access | `col("s").struct.field("f")`, `col("m").map.get(k)` / `.keys()` / `.values()` |
| list/embedding ops | `col("e").list.cosine_distance(o)`, `.l2_distance`, `.dot`, `.normalize`, `.mean_pool` |
| `df.with_column("emb", embed_text(col("t")))` | `ds.ml.embed(model, column="t", output_column="emb", num_gpus=1)` |
| a model UDF over batches | `ds.ml.infer(model, column=..., num_gpus=..., concurrency=...)` |
| LLM generation UDF | `ds.ml.generate(...)` / `batcher.ml.llm_generate(..., engine=vllm_engine("..."))` |
| zero-shot labeling | `ds.ml.classify(engine, labels=[...])` |
| — | `ds.ml.near_duplicates("text")` / `ds.ml.drop_near_duplicates(...)`, `ds.ml.similarity_join(other, left_on=...)` |
| — | `ds.ml.stream_loader(batch_size=, world_size=, rank=)` — sharded, resumable training feed |

**Pass a class, not an instance**, to `ds.ml.infer` / `ds.ml.embed` /
`ds.ml.map_batches`: the model is then constructed once per worker instead of being
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

The functions where Batcher's spelling differs and the capability does not. Everything
absent from this table is either the same name or covered by the relational table above.

| Daft | Batcher |
|---|---|
| `to_snake_case(c)`, `to_camel_case(c)`, `to_kebab_case(c)`, `to_title_case(c)`, `to_upper_*` | one `col("c").str.to_case(style)`; the styles add `sentence`, `dot`, `train` |
| `compress(c, codec)` / `decompress(c, codec)` | `col("c").str.compress(codec)` / `.str.decompress(codec)`; adds zstd, brotli, lz4 |
| `try_compress` / `try_decompress` | not needed — `decompress` is already lenient, so a bad frame is null |
| `regexp_split(c, p)` | `col("c").str.regexp_split(p)` |
| `great_circle_distance(a, b, c, d)` | `bt.great_circle_distance(a, b, c, d, unit="km")`; also `m`/`mi`/`nm` |
| `make_date(y, m, d)` / `make_timestamp(...)` | `bt.make_date(...)` / `bt.make_timestamp(...)` |
| `timestamp_seconds(c)` / `timestamp_millis(c)` / `timestamp_micros(c)` | `bt.from_epoch(c, "s" \| "ms" \| "us" \| "ns")` |
| `date_from_unix_date(c)` | `bt.from_unix_date(c)` |
| `length_bytes(c)` | `col("c").str.len_bytes()` |
| `eq_null_safe(a, b)` | `a.eq_missing(b)` |
| `is_inf(c)` / `not_null(c)` | `c.is_infinite()` / `c.is_not_null()` |
| `columns_sum(...)` / `columns_max(...)` | `bt.sum_horizontal(...)` / `bt.max_horizontal(...)` |
| `date_format(c, f)` / `datepart(p, c)` / `date_trunc(u, c)` | `col("c").dt.strftime(f)` / `bt.date_part(p, c)` / `col("c").dt.truncate(u)` |
| `dot_product(a, b)` / `jaccard_similarity(a, b)` | `a.list.dot(b)` / `a.list.jaccard(b)` |
| `list_sum(c)` / `list_max(c)` / … | `col("c").list.sum()` / `.list.max()` / … |
| `monotonically_increasing_id()` | `ds.with_row_index()` — a `Dataset` method, not an expression |
| `random_int(lo, hi)` | `ds.with_random()` (seeded, `[0, 1)`) scaled and cast |
| `uuid()` | no equivalent, deliberately: a random value per row would make the sequential, parallel and distributed paths disagree, and those must be identical. Use `col("k").str.hash64()` for a surrogate key |

**Do not reach for an epoch cast.** Daft's `timestamp_seconds` has no Batcher alias on
purpose: `col("t").cast("timestamp")` compiles, runs, and is wrong, because Arrow reads a
bare integer as *microseconds*. `bt.from_epoch(c, "s")` is the port.

## The UDF story

| Daft | Batcher |
|---|---|
| `@daft.udf(return_dtype=...)` on a batch fn | `@bt.udf(output_columns=[...])`, applied to a `Dataset` |
| stateful class UDF (`__init__` + `__call__`) | the same class handed to `ds.map_batches(Cls, ...)` / `ds.ml.map_batches` |
| per-row UDF | `@bt.udf(per_row=True)`, or `ds.map`/`ds.flat_map` — avoid; see gotchas |
| SQL-callable UDF | `bt.register_function(name, fn, result_type=...)` |

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
2. **Replace the readers.** `daft.read_*` → `bt.read.<fmt>`; for media, prefer the
   dedicated `bt.read.images` / `bt.read.video` / `bt.read.point_cloud` over a manual
   path scan plus download.
3. **Convert URL fetch → decode → transform.** `.url.download()` becomes
   `ds.ml.download(...)`; the decode/resize/tensor chain stays as `.image` /`.audio` /
   `.video` accessor expressions, which lower to Rust and stay vectorized.
4. **Convert model UDFs to `ds.ml.infer` / `ds.ml.embed`**, passing the model *class*
   with `num_gpus=` and `concurrency=`. Only fall back to `ds.map_batches` when the
   stage is not a model call.
5. **`select` down to the columns each opaque stage reads, before that stage.** The
   optimizer cannot prune across a Python callback.
6. **Verify** (below), then read `ds.explain()` and `ds.stats()` — `stats()` reports
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
  drops every `l_discount = 0.07` row (see `docs/benchmarks/vs-daft.md`).

The general move: when a ported result differs, run the same query through DuckDB before
assuming the port is wrong.

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
  per-batch Python UDFs are still Daft's by ~2×. Quote `docs/benchmarks/vs-daft.md` and
  the parity ledger *with their conditions*, never a bare multiplier.
- **Do expect a ported SQL workload to change answers where Daft was wrong.** On TPC-H
  at sf1 the harness's DuckDB gate fails Daft on q6, q15 and q18 and Daft errors on q21
  and q22 — 5 of 22. If a ported query's numbers move, check DuckDB before assuming the
  port broke it.

## See also

- `docs/migration/transforming.md`, `docs/migration/reading-and-writing.md` — the full
  mapping tables and `from_*`/`to_*` adapters.
- `docs/benchmarks/vs-daft.md`, `docs/benchmarks/multimodal-ingest.md` — the measured
  comparison and the image/point-cloud pipelines.
- `docs/api/ml.md`, `docs/ml/` — the inference, embedding, and training-feed surface.
- `docs/user-guide/udfs.md` — when a UDF is justified and what it costs.
- `docs/deep-dives/shuffle-flight.md`, `docs/deep-dives/mergeable-algebra.md` — why the
  distributed result equals the single-node one, without the object store.
- `/migrate-from-polars-or-pandas`, `/migrate-from-spark` — the sibling migration skills.
