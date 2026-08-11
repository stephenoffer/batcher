"""The Spark/pandas/Ray Data naming and foreign-format-exporter half of the redirect table.

Split out of `_dataset_table` purely to keep each module within the structure limits; it
holds the three largest families — the camelCase Spark method names that map onto a Batcher
spelling, the `snake_case` Ray Data ones, and the pandas exporters and namespaces that are a
display or foreign-format concern. Merged back in `_dataset_table.DATASET_UNSUPPORTED`.
"""

from __future__ import annotations

__all__ = ["DATASET_EXPORTERS", "DATASET_NAMING", "DATASET_RAY_DATA"]


DATASET_NAMING: dict[str, str] = {
    "toPandas": "Spelled ds.to_pandas() here (PEP 8 naming throughout).",
    "toArrow": "Spelled ds.to_arrow() here (PEP 8 naming throughout).",
    "toJSON": "Write JSON with ds.write.json(path), or materialize rows with ds.to_pylist().",
    "toDF": "Rename columns with ds.rename({...}); a Dataset is already the frame.",
    "toLocalIterator": "Spelled ds.iter_batches() here; it streams Arrow batches.",
    "printSchema": "Spelled ds.schema here; ds.info() prints a readable summary.",
    "withColumn": "Spelled ds.with_columns(name=expr) here (PEP 8 naming throughout).",
    "withColumns": "Spelled ds.with_columns(a=expr1, b=expr2) here (PEP 8 naming throughout).",
    "withColumnRenamed": "Spelled ds.rename({'old': 'new'}) here (PEP 8 naming throughout).",
    "withColumnsRenamed": "Spelled ds.rename({'old': 'new', ...}) here (PEP 8 naming throughout).",
    "withMetadata": (
        "Column metadata is not exposed. Rename or cast with ds.rename(...) / ds.cast(...)."
    ),
    "selectExpr": "Use ds.sql('SELECT ... FROM self') or ds.select(<expressions>).",
    "groupBy": "Spelled ds.group_by(...) here (PEP 8 naming throughout).",
    "orderBy": "Spelled ds.sort(...) here.",
    "sortWithinPartitions": "Spelled ds.sort(...) here; ordering is global.",
    "unionAll": (
        "Spelled ds.union(other) here; union keeps duplicates (pass distinct=True to drop them)."
    ),
    "unionByName": "Spelled ds.union(other) here; ensure both sides expose the same columns.",
    "exceptAll": "Spelled ds.except_(other) here.",
    "intersectAll": "Spelled ds.intersect(other) here.",
    "subtract": "Spelled ds.except_(other) here (set difference).",
    "crossJoin": "Spelled ds.cross_join(other) here.",
    "dropDuplicates": "Spelled ds.distinct() (or ds.drop_duplicates()) here.",
    "where": "Spelled ds.filter(bt.col('x') > 0) here (Spark's `where` alias).",
    "approxQuantile": "Spelled ds.approx_quantile(column, [0.5]) here.",
    "sampleBy": (
        "Per-key sampling: ds.sample_per_group(by, n) for n rows per group, or "
        "ds.stratified_split(by=...)."
    ),
    "randomSplit": (
        "Split with ds.train_val_test_split(by=...) or ds.stratified_split(by=..., test_size=...)."
    ),
    "repartitionByRange": "Sort then repartition: ds.sort('key').repartition(n).",
    "colRegex": "Select columns by pattern with ds.select(bt.matches(r'^value_')).",
    "freqItems": "Approximate frequent values: ds.value_counts('col') or ds.group_by('col').len().",
    "observe": (
        "Measured per-operator metrics are ds.stats(); ds.explain(analyze=True) reports what ran."
    ),
    "createTempView": (
        "Batcher has no view registry. Pass the dataset into bt.sql('... FROM t', t=ds)."
    ),
    "createGlobalTempView": (
        "Batcher has no view registry. Pass the dataset into bt.sql('... FROM t', t=ds)."
    ),
    "createOrReplaceGlobalTempView": (
        "Batcher has no view registry. Pass into bt.sql('... FROM t', t=ds)."
    ),
    "registerTempTable": "Batcher has no view registry. Pass into bt.sql('... FROM t', t=ds).",
    "alias": "Self-joins disambiguate columns automatically (a suffix); no alias is needed.",
    "rdd": "Batcher has no RDD layer. Use ds.iter_batches() for Arrow batches.",
    "metrics": "Spelled ds.stats() here, which returns measured per-operator RunStats.",
    "isLocal": "Execution is in-process unless you pass collect(distributed=True).",
    "writeTo": "Write to a catalog table with ds.write.iceberg(table).",
    "writeStream": "Streaming writes use ds.write.* with a Trigger; see the streaming guide.",
    "createOrReplaceTempView": (
        "Batcher has no global view registry. Pass the dataset straight into "
        "bt.sql('SELECT * FROM t', t=ds), or use ds.sql('SELECT * FROM self')."
    ),
    "n_partitions": (
        "Partitioning is decided at execution, not carried on the plan, so a lazy "
        "Dataset has no partition count. ds.repartition(n) sets the output layout for "
        "the next write, and ds.explain(analyze=True) reports what actually ran."
    ),
    "memory_usage_deep": "Spelled ds.memory_usage() here; it is an estimate, not a measurement.",
    "hstack": (
        "Batcher has no positional column stacking (there is no row order to align "
        "on). Add columns with ds.with_columns(...), or join on a key with "
        "ds.join(other, on='key')."
    ),
}

# --- namespaces and exporters that are a display / foreign-format concern ------------
DATASET_EXPORTERS: dict[str, str] = {
    "to_records": "For a NumPy record array collect first: ds.to_pandas().to_records().",
    "to_feather": (
        "Write Arrow/Feather with ds.write.arrow(path); ds.to_arrow() gives a pyarrow.Table."
    ),
    "to_excel": "No Excel sink; collect first: ds.to_pandas().to_excel(path).",
    "to_hdf": "No HDF5 sink; collect first: ds.to_pandas().to_hdf(...).",
    "to_pickle": "No pickle sink; collect first: ds.to_pandas().to_pickle(...).",
    "to_stata": "No Stata sink; collect first: ds.to_pandas().to_stata(...).",
    "to_latex": "A display concern; collect first: ds.to_pandas().to_latex().",
    "to_markdown": "A display concern; collect first: ds.to_pandas().to_markdown().",
    "to_html": "A display concern; collect first: ds.to_pandas().to_html().",
    "to_string": "A display concern; use ds.show() for a preview, or ds.to_pandas().to_string().",
    "to_clipboard": "A display concern; collect first: ds.to_pandas().to_clipboard().",
    "to_xml": "No XML sink; collect first: ds.to_pandas().to_xml(...).",
    "to_sql": "Write to a database with ds.write.sql(uri, table=...).",
    "na": "Null handling is ds.fill_null(...), ds.drop_nulls(), and bt.col('x').is_null().",
    "stat": (
        "Statistics are ds.corr(...), ds.cov(...), ds.approx_quantile(...), and ds.crosstab(...)."
    ),
    "plot": "Plotting is a display concern; collect first: ds.to_pandas().plot().",
    "hist": "Plotting is a display concern; collect first: ds.to_pandas().hist().",
    "boxplot": "Plotting is a display concern; collect first: ds.to_pandas().boxplot().",
    "sparse": "Batcher has no sparse frame accessor; columns are dense Arrow arrays.",
    "add_prefix": "Prefix every column with ds.rename(lambda c: 'pre_' + c).",
    "add_suffix": "Suffix every column with ds.rename(lambda c: c + '_suf').",
}


# --- Ray Data spellings ---------------------------------------------------------------
# Ray Data is the one competitor whose users arrive already writing Python against a lazy,
# distributed `Dataset`, so the concepts port but the names frequently do not. The families
# below are grouped by why the name differs: a plain rename, a namespace move (`ds.write.*`,
# `ds.ml.*`), a lazy-vs-materialized difference, or a Ray-internal that has no counterpart
# here because Batcher's data plane bypasses the Ray object store.
DATASET_RAY_DATA: dict[str, str] = {
    # Relational verbs that only differ by name.
    "select_columns": "Spelled ds.select('a', 'b') here.",
    "drop_columns": "Spelled ds.drop('a', 'b') here.",
    "rename_columns": "Spelled ds.rename({'old': 'new'}) here.",
    "add_column": "Spelled ds.with_columns(name=bt.col('x') * 2) here.",
    "aggregate": "Spelled ds.agg(...) here, or ds.group_by('k').agg(...) for a grouped one.",
    "mix": "Spelled bt.concat([a, b]) here (or ds.union(other) for two).",
    "zip": (
        "There is no positional column-wise zip: a relation is an unordered multiset, so "
        "pairing rows by position is only defined once you name the order. Add the position "
        "and join on it: a.with_row_index('i').join(b.with_row_index('i'), on='i')."
    ),
    # Splitting and sampling.
    "split": (
        "Split by position with ds.split_at_indices([2, 5]), or by fraction with "
        "ds.split_proportionately([0.2, 0.5]). Both stay lazy and materialize nothing."
    ),
    "streaming_split": (
        "Batcher's loaders stream without a split step: ds.ml.stream_loader(...) feeds one "
        "worker, and ds.split_proportionately([...]) gives disjoint shards that stay lazy."
    ),
    "random_shuffle": "Spelled ds.shuffle(seed=0) here (a full, seeded shuffle).",
    "randomize_block_order": (
        "Spelled ds.shuffle(seed=0) here. Batcher shuffles rows rather than reordering "
        "blocks, so there is no weaker block-level variant to choose."
    ),
    "random_sample": "Spelled ds.sample_frac(0.1, seed=0) here, or ds.sample(n) for a row count.",
    "train_test_split": "Spelled ds.ml.train_test_split(0.2, seed=0) here.",
    "streaming_train_test_split": (
        "Spelled ds.ml.train_test_split(0.2, seed=0) here; it is already lazy, so both "
        "halves stream without a separate streaming variant."
    ),
    # Materialization: Ray Data returns eager results where Batcher stays lazy.
    "take": "Spelled ds.limit(n).to_pylist() here.",
    "take_all": "Spelled ds.to_pylist() here (ds.collect() for an Arrow table).",
    "take_batch": "Spelled ds.limit(n).to_arrow() here.",
    "materialize": (
        "Spelled ds.cache() here: it pins the computed result so downstream branches reuse "
        "it instead of recomputing. ds.persist() is the spill-backed form."
    ),
    "iterator": "Spelled ds.iter_batches() here; ds.iter_rows() yields dicts.",
    "iter_torch_batches": "Spelled ds.ml.iter_torch_batches(...) here.",
    "iter_tf_batches": "Spelled ds.ml.to_tf(...) here.",
    "iter_jax_batches": (
        "There is no JAX iterator. Iterate Arrow with ds.iter_batches() and convert each "
        "batch, or take NumPy batches with ds.ml.to_numpy_batches(...)."
    ),
    # Writers live on the `ds.write` namespace.
    "write_parquet": "Spelled ds.write.parquet(path) here (every sink is on ds.write).",
    "write_csv": "Spelled ds.write.csv(path) here.",
    "write_json": "Spelled ds.write.json(path) here.",
    "write_iceberg": "Spelled ds.write.iceberg(table) here.",
    "write_lance": "Spelled ds.write.lance(path) here.",
    "write_mongo": "Spelled ds.write.mongo(uri, ...) here.",
    "write_snowflake": "Spelled ds.write.snowflake(...) here.",
    "write_sql": "Spelled ds.write.sql(uri, table=...) here.",
    "write_kafka": "Spelled ds.write.kafka(...) here.",
    "write_datasink": "Custom sinks are ds.write.for_each_batch(fn); every sink is on ds.write.",
    "write_numpy": "No NumPy sink. Write a column with ds.write.parquet(path), or ds.to_numpy().",
    "write_images": (
        "No image sink. Encode to bytes with the .image accessor, then ds.write.parquet(path)."
    ),
    "write_tfrecords": (
        "No TFRecord sink. Write ds.write.parquet(path), which the loaders read directly."
    ),
    "write_webdataset": (
        "No WebDataset sink. Write ds.write.parquet(path); Batcher reads WebDataset shards on "
        "the way in with bt.read.webdataset(...)."
    ),
    "write_bigquery": "No BigQuery sink. Land Parquet with ds.write.parquet(path) and load it.",
    "write_clickhouse": "No ClickHouse sink. Use ds.write.sql(uri, table=...).",
    "write_turbopuffer": (
        "No Turbopuffer sink. Write vectors with ds.write.parquet(path), or push them with "
        "ds.write.for_each_batch(fn)."
    ),
    # Foreign frameworks.
    "to_spark": "No Spark bridge. Hand over a file: ds.write.parquet(path), then read it in Spark.",
    "to_daft": "No Daft bridge. Both speak Arrow: daft.from_arrow(ds.to_arrow()).",
    "to_dask": (
        "No Dask bridge. Collect first: ds.to_pandas(), or hand over ds.write.parquet(path)."
    ),
    "to_modin": "No Modin bridge. Collect first: ds.to_pandas().",
    "to_mars": "No Mars bridge. Collect first: ds.to_pandas().",
    "to_random_access_dataset": (
        "No random-access key lookup service. Keep the table and filter it: "
        "ds.filter(bt.col('key') == k), which pushes down to the scan."
    ),
    # Ray internals — absent by design, because the data plane bypasses the object store.
    "to_arrow_refs": (
        "There are no block object refs: bulk Arrow moves over Arrow Flight, not the Ray "
        "object store. Stream the data with ds.iter_batches()."
    ),
    "to_pandas_refs": "There are no block object refs. Collect with ds.to_pandas().",
    "to_numpy_refs": "There are no block object refs. Collect with ds.to_numpy().",
    "get_internal_block_refs": (
        "There are no block object refs; bulk Arrow moves over Arrow Flight rather than the "
        "Ray object store. Stream with ds.iter_batches()."
    ),
    "iter_internal_ref_bundles": "No internal ref bundles. Stream with ds.iter_batches().",
    "num_blocks": "Spelled ds.repartition(n) to set, and ds.stats() to read what ran.",
    "size_bytes": "Read it from ds.stats(), or ds.describe() for per-column detail.",
    "input_files": "Not exposed as a list. ds.explain() shows the scan, ds.stats() what it read.",
    "context": (
        "Configuration is process-wide, not a per-dataset context object: read bt.config "
        "and set it with bt.set_config(...) or the bt.config_context(...) block."
    ),
    "get_dataset_id": "Datasets carry no id. ds.stats() identifies a run.",
    "name": "Datasets carry no name. ds.explain() labels the plan.",
    "set_name": "Datasets carry no name, and are immutable besides. ds.explain() labels the plan.",
    "summary": "Spelled ds.describe() here; ds.stats() reports what execution did.",
    "get_stats_summary": "Spelled ds.stats() here.",
    "serialize_lineage": (
        "No lineage pickling: a Dataset is already a lazy plan, so pass it directly. "
        "ds.explain() prints it."
    ),
    "deserialize_lineage": "No lineage pickling; a Dataset is already a lazy plan.",
    "has_serializable_lineage": "No lineage pickling; a Dataset is already a lazy plan.",
}
