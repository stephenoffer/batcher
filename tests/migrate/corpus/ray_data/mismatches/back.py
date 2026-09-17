import ray.data
import ray.data.expressions

ds = ray.data.from_items([{"k": 1}, {"k": 1}, {"k": 2}])
# batcher-migrate: Ray Data `Dataset.random_shuffle` differs in Batcher (`Dataset.shuffle`): Ray random_shuffle() without seed draws a new permutation per execution; Batcher shuffle(seed=0) is deterministic, the same permutation every run. Pass a fresh seed per epoch.
# batcher-migrate: Ray Data `Dataset.random_shuffle` has no exact Ray Data spelling; left as written
shuffled = ds.random_shuffle()
# batcher-migrate: Ray Data `Dataset.materialize` differs in Batcher (`Dataset.cache`): Ray materialize() executes now and pins blocks in the object store, returning a MaterializedDataset; Batcher cache() is lazy and stores the Arrow result on the first terminal op. Port as: ds = ds.cache(); ds.count() to force it.
# batcher-migrate: Ray Data `Dataset.materialize` has no exact Ray Data spelling; left as written
pinned = ds.materialize()
# batcher-migrate: Ray Data `Dataset.unique` differs in Batcher (`Dataset.select / Dataset.distinct / Dataset.to_pylist`): Ray unique(column) is eager and returns a list of the column's distinct values (ignore_nulls=False keeps None); Batcher Dataset.distinct deduplicates rows and returns a Dataset. Port as: [r[c] for r in ds.select(c).distinct().to_pylist()].
# batcher-migrate: Ray Data `Dataset.unique` has no exact Ray Data spelling; left as written
values = ds.unique("k")
# batcher-migrate: Ray Data `Dataset.add_column` differs in Batcher (`Dataset.map_batches`): Ray add_column(col, fn) hands fn a pandas batch (batch_format='pandas' default) and appends its result as `col`. Batcher: map_batches(fn, batch_format='pandas') returning the batch with the column added, or with_columns when the column is an expression.
# batcher-migrate: Ray Data `Dataset.add_column` has no exact Ray Data spelling; left as written
widened = ds.add_column("k2", lambda frame: frame["k"] * 2)
print(shuffled, pinned, values, widened)
