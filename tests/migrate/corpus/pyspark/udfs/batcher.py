import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
import batcher as bt

df = bt.from_pylist([{"x": 1}, {"x": 2}])
# batcher-migrate: PySpark `types.IntegerType` is `Expr.cast` in Batcher, but this call does not carry over 1:1
# batcher-migrate: PySpark `functions.udf` differs in Batcher (`bt.udf`): bt.udf(fn)(col) raises instead of evaluating at plan time; port a row UDF as ds.map_batches over Arrow batches, or as an expression
plus_one = F.udf(lambda v: v + 1, IntegerType())


# batcher-migrate: PySpark `functions.pandas_udf` is `Dataset.map_batches` in Batcher, but this call does not carry over 1:1
@F.pandas_udf("long")
def times_two(s: pd.Series) -> pd.Series:
    return s * 2


out = df.with_columns(y=plus_one(bt.col("x"))).with_columns(z=times_two(bt.col("x")))
# batcher-migrate: PySpark `DataFrame.show` was rewritten to `Dataset.show`, which lacks: truncate= and vertical= display options (Spark shows 20 rows by default)
out.show()
