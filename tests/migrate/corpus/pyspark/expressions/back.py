import batcher as bt
from pyspark.sql import functions as F

# batcher-migrate: PySpark `bt.from_pylist` has no exact PySpark spelling; left as written
df = bt.from_pylist([{"x": 1, "s": "apple"}, {"x": 5, "s": "banana"}])
# batcher-migrate: PySpark `bt.when` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Expr.then` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Dataset.with_columns` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Expr.str.substr` has no exact PySpark spelling; left as written
out = (
    df.with_columns(size=bt.when(F.col("x") > 2).then("big").otherwise("small"))
    .with_columns(upper=F.upper(F.col("s")))
    .with_columns(head=F.col("s").str.substr(1, 3))
    .with_columns(next=F.col("x") + F.lit(1))
    .with_columns(same=df["x"])
)
# batcher-migrate: PySpark `DataFrame.show` was rewritten to `Dataset.show`, which lacks: truncate= and vertical= display options (Spark shows 20 rows by default)
# batcher-migrate: PySpark `Dataset.show` has no exact PySpark spelling; left as written
out.show()
