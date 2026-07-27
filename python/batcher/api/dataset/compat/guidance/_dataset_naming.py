"""The Spark/pandas naming and foreign-format-exporter half of the Dataset redirect table.

Split out of `_dataset_table` purely to keep each module within the structure limits; it
holds the two largest families — the camelCase Spark method names that map onto a Batcher
spelling, and the pandas exporters and namespaces that are a display or foreign-format
concern. Merged back in `_dataset_table.DATASET_UNSUPPORTED`.
"""

from __future__ import annotations

__all__ = ["DATASET_EXPORTERS", "DATASET_NAMING"]


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
