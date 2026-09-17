from pyspark.sql import functions as F
import batcher as bt

# batcher-migrate: PySpark `bt.read.parquet` has no exact PySpark spelling; left as written
df = bt.read.parquet("s3://bucket/events/")
# batcher-migrate: PySpark `functions.date_format` needs a manual rewrite: Spark takes Java DateTimeFormatter patterns (yyyy-MM-dd) and renders in the session time zone; strftime takes %Y-%m-%d. Codemod translates the pattern
# batcher-migrate: PySpark `Expr.dt.strftime` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Dataset.with_columns` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Expr.str.to_date` has no exact PySpark spelling; left as written
out = (
    df.with_columns(day=F.col("ts").dt.strftime("%Y-%m-%d"))
    .with_columns(parsed=F.col("raw").str.to_date("%d/%m/%Y"))
    .with_columns(stamp=F.date_format("ts", "yyyy-MM-dd HH:mm:ss.SSS"))
    .with_columns(year=F.year(F.col("ts")))
)
# batcher-migrate: PySpark `DataFrame.printSchema` differs in Batcher (`Dataset.schema`): Spark prints a tree; Batcher exposes a pyarrow.Schema to print
# batcher-migrate: PySpark `Dataset.printSchema` has no exact PySpark spelling; left as written
out.printSchema()
