from pyspark.sql import functions as F
import batcher as bt

# batcher-migrate: PySpark `bt.from_pylist` has no exact PySpark spelling; left as written
df = bt.from_pylist([{"s": "a b", "xs": [1, 2]}, {"s": "c", "xs": []}])
# batcher-migrate: PySpark `functions.concat` differs in Batcher (`bt.concat / bt.concat_str`): Spark concat returns null when any argument is null: bt.concat_str(*cols, ignore_nulls=False). Array arguments are list concatenation
joined = df.select(F.concat(F.col("s"), F.lit("!")).alias("loud"))
# batcher-migrate: PySpark `functions.split` differs in Batcher (`Expr.str.split`): Spark split(str, pattern, limit) with limit > 0 is str.regexp_split(pattern, limit=limit); limit <= 0 means no limit. Java-only regex syntax needs a manual port
parts = df.select(F.split("s", " ").alias("parts"))
# batcher-migrate: PySpark `functions.explode` is `Dataset.explode` in Batcher, but this call does not carry over 1:1
exploded = df.select(F.explode("xs").alias("x"))
# batcher-migrate: PySpark `DataFrame.selectExpr` has no Batcher equivalent yet: SQL expression strings in select (bt.expr)
# batcher-migrate: PySpark `Dataset.selectExpr` has no exact PySpark spelling; left as written
computed = df.selectExpr("size(xs) as n")
# batcher-migrate: PySpark `functions.broadcast` is not provided by Batcher: no join-hint IR; Kyber picks the build side from measured cardinalities (revisit)
small = F.broadcast(df)
# batcher-migrate: PySpark `DataFrame.rdd` is not provided by Batcher: no RDD or JVM context
rdd = df.rdd
# batcher-migrate: PySpark `bt.from_pylist` has no exact PySpark spelling; left as written
other = bt.from_pylist([{"s": "a b", "n": 1}])
# batcher-migrate: PySpark `DataFrame.join` needs a manual rewrite: Spark how= spellings (left_outer, full_outer, leftsemi, left_anti, ...) as values; Column/non-equi on= conditions; full-outer key coalescing
# batcher-migrate: PySpark `Dataset.join` has no exact PySpark spelling; left as written
paired = df.join(other, df.s == other.s, "inner")
print(joined, parts, exploded, computed, small, rdd, paired)
