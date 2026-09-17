from pyspark.sql import functions as F
import batcher as bt

df = bt.read.parquet("s3://bucket/events/")
# batcher-migrate: PySpark `functions.date_format` needs a manual rewrite: Spark takes Java DateTimeFormatter patterns (yyyy-MM-dd) and renders in the session time zone; strftime takes %Y-%m-%d. Codemod translates the pattern
out = (
    df.with_columns(day=bt.col("ts").dt.strftime("%Y-%m-%d"))
    .with_columns(parsed=bt.col("raw").str.to_date("%d/%m/%Y"))
    .with_columns(stamp=F.date_format("ts", "yyyy-MM-dd HH:mm:ss.SSS"))
    .with_columns(year=bt.col("ts").dt.year())
)
# batcher-migrate: PySpark `DataFrame.printSchema` differs in Batcher (`Dataset.schema`): Spark prints a tree; Batcher exposes a pyarrow.Schema to print
out.printSchema()
