# Chaining and persisting

This page covers sequencing preprocessors into one object, fitting a whole pipeline on
the training split, and saving the fitted state so serving applies exactly what training
learned.

## Chaining steps

`Chain` is the sklearn `Pipeline` equivalent. It fits each step on the **previous
step's output** and replays the fitted steps, in order, over any split. Doing this by
hand means fitting step *i* on data that steps *0..i-1* have already transformed. That
is easy to get subtly wrong, and the mistake leaks held-out statistics into training
features without ever failing.

```python
import batcher as bt
from batcher.ml import Chain, SimpleImputer, StandardScaler

ds = bt.from_pydict({"age": [10.0, 20.0, None, 40.0, 30.0, 50.0]})
train, test = ds.ml.train_test_split(0.3, seed=0)

chain = Chain(SimpleImputer(["age"]), StandardScaler(["age"])).fit(train)
train_x, test_x = chain.transform(train), chain.transform(test)
print(chain)
# Chain(SimpleImputer, StandardScaler)
```

Call `fit` on the training split only, then `transform` on both. A `Chain` is itself a
`Preprocessor`, so it nests. Its steps stay introspectable through `chain[0]` and
`len(chain)`, which is how you read a fitted step's learned state.

You can also sequence several preprocessors by hand. Fit each on the previous
step's output, then transform any split through the same fitted objects.

```python
from batcher.ml.preprocessors import StandardScaler, SimpleImputer
import batcher as bt

train = bt.from_pydict({"age": [20.0, 30.0, 40.0, 50.0], "income": [1.0, 2.0, 3.0, 4.0]})

imputer = SimpleImputer(["age"])
scaler = StandardScaler(["age", "income"])
train_scaled = scaler.fit_transform(imputer.fit_transform(train))
print(train_scaled.collect().column_names)
# ['age', 'income']
```

Each object keeps its fitted state, so the same steps transform held-out data with the
statistics learned on train:

```python
import batcher as bt
from batcher.ml.preprocessors import StandardScaler

train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
scaler = StandardScaler(["x"]).fit(train)

test = bt.from_pydict({"x": [6.0, 7.0]})
print([round(v, 3) for v in scaler.transform(test).collect().column("x").to_pylist()])
# [2.121, 2.828]
```

## Saving a fitted preprocessor

A preprocessor is only useful because its state is learned once and reused: the scaler
standardizing a request at serving time must hold the *training* set's mean. `save` writes
that state as plain JSON.

```python
import os
import tempfile

from batcher.ml.preprocessors import Preprocessor, StandardScaler

scaler = StandardScaler("x").fit(bt.from_pydict({"x": [1.0, 3.0]}))
path = os.path.join(tempfile.mkdtemp(), "scaler.json")
scaler.save(path)
print(Preprocessor.load(path).mean_)
```

JSON rather than a pickle, deliberately: the file is reviewable, diffable, portable to a
serving stack in another language, and safe to load from a store you do not fully control.
A cloud URI works wherever a local path does.

## Composing a pipeline

A real feature pipeline is several preprocessors in sequence. Fit each on the previous
step's output, then push any split through the *same* fitted objects so train and
validation share every learned statistic. The classic order is impute, then scale, then
encode.

```python
import batcher as bt
from batcher.ml.preprocessors import SimpleImputer, StandardScaler, OneHotEncoder

train = bt.from_pydict(
    {
        "age": [20.0, 30.0, None, 50.0],
        "income": [1.0, 2.0, 3.0, 4.0],
        "city": ["paris", "rome", "paris", "oslo"],
    }
)

imputer = SimpleImputer(["age"], strategy="median")
scaler = StandardScaler(["age", "income"])
encoder = OneHotEncoder(["city"])

# Fit each stage on the previous stage's output, on train only.
step1 = imputer.fit_transform(train)
step2 = scaler.fit_transform(step1)
prepared = encoder.fit_transform(step2)
print(prepared.collect().column_names)
# ['age', 'income', 'city_oslo', 'city_paris', 'city_rome']
```

Held-out data flows through the identical fitted objects. Use `transform`, never
`fit_transform`, so it inherits the training statistics:

```python
val = bt.from_pydict({"age": [None], "income": [2.5], "city": ["rome"]})
prepared_val = encoder.transform(scaler.transform(imputer.transform(val)))
print(prepared_val.collect().column_names)
# ['age', 'income', 'city_oslo', 'city_paris', 'city_rome']
```

## See also

- {doc}`/ml/preparing/preprocessors/index`: the fit/transform contract each step in a chain obeys.
- {doc}`/ml/preparing/preprocessors/feature-generation`: the steps a chain usually ends with.
- {doc}`/tutorials/feature-engineering`: the same workflow end to end.
