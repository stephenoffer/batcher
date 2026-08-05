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

## Taking the model with it

{py:class}`Chain <batcher.ml.preprocessors.Chain>` composes preprocessors and stops there,
which leaves you responsible for remembering which transforms a model was trained behind and
applying exactly those at serving time. That is where train/serve skew comes from, and it
fails silently: a model scored behind one fewer transform returns numbers, not an error.

{py:class}`Pipeline <batcher.ml.Pipeline>` owns both halves. `fit` fits each step on the
previous one's output and then the model on the fully transformed frame; `predict` replays
the identical sequence:

```python
import batcher as bt
from batcher.ml import LinearRegression, Pipeline, SimpleImputer, StandardScaler

train = bt.from_pydict({"x": [1.0, None, 3.0, 5.0], "y": [2.0, 4.0, 6.0, 10.0]})
pipe = Pipeline(
    SimpleImputer(["x"]),
    StandardScaler(["x"]),
    model=LinearRegression(["x"], "y"),
).fit(train)
print("prediction" in pipe.predict(train).columns)
# True
```

Because it is one object it saves as one file, so what ships to serving is the whole recipe
rather than a model plus a memo about what preceded it:

```python
import os
import tempfile

target = os.path.join(tempfile.mkdtemp(), "pipeline.json")
pipe.save(target)
served = Pipeline.load(target)
print([type(step).__name__ for step in served.steps])
# ['SimpleImputer', 'StandardScaler']
```

`transform` applies the steps without scoring, which is how you inspect what the model
actually sees. And because a `Pipeline` exposes `fit`/`predict`, it drops straight into
{py:func}`cross_val_score <batcher.ml.cross_val_score>` and
{py:func}`grid_search <batcher.ml.grid_search>` — which is the point of putting the
preprocessing inside it, since cross-validating a model whose scaler was fitted on the whole
frame measures nothing.

## Predicting several things at once

Every estimator takes one target column. A problem that predicts several things from the same features leaves you holding a list of models and the job of remembering which column each was fitted on, which is the same skew {py:class}`Pipeline <batcher.ml.Pipeline>` prevents one level up.

{py:class}`MultiOutputRegressor <batcher.ml.MultiOutputRegressor>` owns that list. Pass the estimator as a class, since each sub-model needs its own target:

```python
import batcher as bt
from batcher.ml import LinearRegression, MultiOutputRegressor

demand = bt.from_pydict(
    {
        "price": [10.0, 12.0, 14.0, 16.0, 18.0],
        "north": [100.0, 96.0, 92.0, 88.0, 84.0],
        "south": [50.0, 44.0, 38.0, 32.0, 26.0],
    }
)

model = MultiOutputRegressor(LinearRegression, ["price"], ["north", "south"]).fit(demand)
scored = model.predict(demand).to_pydict()
print(round(scored["prediction_north"][0], 1), round(scored["prediction_south"][0], 1))
# 100.0 50.0
print([round(m.coef_[0], 1) for m in model.estimators_])
# [-2.0, -3.0]
```

Each region gets its own slope, which a single shared model would have to average away. Predictions are appended as `prediction_<target>`, and scoring every target is one pass because each sub-model contributes an expression to the same frame.

{py:class}`MultiOutputClassifier <batcher.ml.MultiOutputClassifier>` is the same thing for labels, and it is worth being clear about which problem that is. Multi-*class* picks exactly one of several classes and belongs to {py:class}`OneVsRestClassifier <batcher.ml.multiclass.OneVsRestClassifier>`. Multi-*label* lets a row carry any number of independent tags at once, so a document can be both "finance" and "urgent"; each label is its own yes-or-no question with its own model and no argmax across them.

This is a wrapper rather than an optimization. Two different targets genuinely need two different fits, so the cost is one fit per target. Where a shared fit is possible it is worth preferring, which is what {py:class}`RidgeCV <batcher.ml.linear.RidgeCV>` does for a penalty path.

## See also

- {doc}`/ml/preparing/preprocessors/index`: the fit/transform contract each step in a chain obeys.
- {doc}`/ml/preparing/preprocessors/feature-generation`: the steps a chain usually ends with.
- {doc}`/tutorials/ml/feature-engineering`: the same workflow end to end.
