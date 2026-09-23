# Chaining and persisting

This page covers sequencing preprocessors into one object, fitting a whole pipeline on
the training split, and saving the fitted state so serving applies exactly what training
learned.

Use {py:class}`Chain <batcher.ml.preprocessors.Chain>` when the steps are all preprocessors, and
{py:class}`Pipeline <batcher.ml.Pipeline>` when a model comes after them. Sequencing the
objects by hand also works, as shown below, but it's the version you can get wrong without
anything failing.

## Chaining steps

`Chain` is the equivalent of a scikit-learn `Pipeline` without the model. It fits each step
on the previous step's output and replays the fitted steps, in order, over any split. By
hand, that means fitting step *i* on data steps *0..i-1* have already transformed. Get it
wrong and held-out statistics leak into the training features, with no error.

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
{py:class}`Preprocessor <batcher.ml.preprocessors.Preprocessor>`, so it nests. Its steps stay introspectable through `chain[0]` and
`len(chain)`, which is how you read a fitted step's learned state.

By default `Chain` is built with `cache=True`, which collects the whole training split to the driver once and fits every step against that copy. That trades one source scan per step for holding the split in driver memory, and it means the later fits run single-node. For a large or distributed training set, build the chain with `cache=False` so every step's fit is its own aggregate over the source, routed like any other fit.

## Sequencing by hand

This is what `Chain` does for you, spelled out once. Fit each step on the previous step's
output, then push any split through the same fitted objects so train and validation share
every learned statistic. The classic order is impute, then scale, then encode.

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

## Saving a fitted preprocessor

A fitted preprocessor is worth keeping because its state is learned once and reused. The
scaler standardizing a request at serving time must hold the training set's mean. `save`
writes that state as plain JSON.

```python
import os
import tempfile

from batcher.ml.preprocessors import Preprocessor, StandardScaler

scaler = StandardScaler("x").fit(bt.from_pydict({"x": [1.0, 3.0]}))
path = os.path.join(tempfile.mkdtemp(), "scaler.json")
scaler.save(path)
print(Preprocessor.load(path).mean_)
```

JSON rather than a pickle is deliberate. You can review and diff the file, read it from a
serving stack in another language, and load it from a store you don't fully control without
running its code. A cloud URI such as `s3://`, `gs://`, or `abfs://` works wherever a local
path does.

Every preprocessor that holds only data round-trips exactly, including the learned per-group tables of `GroupImputer` and `GroupStatEncoder`. A preprocessor built around a Python callable, meaning `FunctionTransformer`, `Tokenizer`, or `RFE`, has nothing JSON can hold for that argument, so `save` raises `PlanError` at save time rather than writing a file that `load` can't rebuild.

## Taking the model with it

{py:class}`Chain <batcher.ml.preprocessors.Chain>` stops at the preprocessors. You're left
remembering which transforms a model was trained behind and applying exactly those at
serving time. That's where train/serve skew comes from, and it fails silently: a model
scored behind one fewer transform returns numbers, not an error.

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

One object saves as one file, so the whole recipe ships to serving instead of a model plus
a memo about what came before it:

```python
import os
import tempfile

target = os.path.join(tempfile.mkdtemp(), "pipeline.json")
pipe.save(target)
served = Pipeline.load(target)
print([type(step).__name__ for step in served.steps])
# ['SimpleImputer', 'StandardScaler']
```

`transform` applies the steps without scoring, so you can inspect what the model sees. A
`Pipeline` exposes `fit` and `predict`, so it drops straight into
{py:func}`cross_val_score <batcher.ml.cross_val_score>` and
{py:func}`grid_search <batcher.ml.grid_search>`, which refit the preprocessing inside every
fold. Cross-validating a model whose scaler was fitted on the whole frame measures
nothing.

## Predicting several things at once

Every estimator takes one target column. A problem that predicts several things from the
same features leaves you holding a list of models and the job of remembering which column
each was fitted on, which is the same skew {py:class}`Pipeline <batcher.ml.Pipeline>`
prevents one level up.

{py:class}`MultiOutputRegressor <batcher.ml.MultiOutputRegressor>` owns that list. Pass the
estimator as a class, since each sub-model needs its own target:

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

Each region gets its own slope, which a single shared model would have to average away.
Predictions are appended as `prediction_<target>`, and scoring every target is one pass
because each sub-model contributes an expression to the same frame.

{py:class}`MultiOutputClassifier <batcher.ml.MultiOutputClassifier>` does the same for
labels, and it solves the multi-*label* problem. Multi-*class* picks exactly one of several
classes and belongs to {py:class}`OneVsRestClassifier
<batcher.ml.multiclass.OneVsRestClassifier>`. Multi-label lets a row carry any number of
independent tags, so a document can be both "finance" and "urgent". Each label is its own
yes-or-no question with its own model, and there's no argmax across them.

The wrapper saves bookkeeping, not compute. Two targets need two fits. Where one shared fit
can serve several answers, prefer it, as {py:class}`RidgeCV <batcher.ml.linear.RidgeCV>`
does for a penalty path.


## Asking a preprocessor about itself

Every preprocessor answers two questions without being run, which is what a pipeline
builder, a serializer, or a test needs before it calls `transform`.

{py:obj}`Preprocessor.is_fitted <batcher.ml.Preprocessor>` is `False` until `fit` has run
and `True` afterwards. It is the check to make before handing a preprocessor to something
that will call `transform`, because an unfitted one raises rather than guessing.
{py:meth}`get_params <batcher.ml.Preprocessor>` returns the constructor arguments the
instance is holding, so a fitted preprocessor can be described, logged, or rebuilt.

```python
import batcher as bt
from batcher.ml.preprocessors import StandardScaler

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 10.0]})
scaler = StandardScaler("x")
print(scaler.is_fitted, scaler.get_params())

scaler = scaler.fit(ds)
print(scaler.is_fitted)
```

`get_params` reports the configuration, never the learned state. The learned values live in
the trailing-underscore attributes (`mean_`, `scale_`, and so on), which is the same split
scikit-learn uses, so a parameter grid built from `get_params` cannot accidentally carry a
fitted model's data.

## See also

- {doc}`/ml/preparing/preprocessors/index`: the fit/transform contract each step in a chain obeys.
- {doc}`/ml/preparing/preprocessors/feature-generation`: the steps a chain usually ends with.
- {doc}`/ml/preparing/preprocessors/feature-selection`: selectors, which chain like any other step.
- {doc}`/ml/evaluation/model-selection`: cross-validating and tuning a `Pipeline`.
- {doc}`/getting-started/tutorials/ml/feature-engineering`: the same workflow end to end.
