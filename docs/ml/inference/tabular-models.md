# Score a tabular model

This page describes how to run a fitted XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX model over a Batcher {py:class}`Dataset <batcher.Dataset>`, and how to fit the classical estimators in `batcher.ml` without leaving the engine.

Tabular models are what most production ML runs, and their shape differs from a language model's. The input is dozens of numeric columns, the model is megabytes, and the bottleneck is feeding the model rather than the model itself. {py:meth}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>` is built for that shape.

## How it works

`ds.ml.predict` builds a load-once class UDF. The model is constructed once per worker, and each Arrow batch is assembled into one dense `(rows, features)` matrix that reaches the model in a single call. Nothing crosses the boundary a row at a time, and nothing is materialized on the driver.

The following diagram splits that work into what happens once and what happens per batch:

![When the query is built, on the driver, ds.ml.predict takes a fitted object or a path. It checks the features against the model's feature names where the model recorded them and raises on a mismatch, then resolves the output columns from the model's class or tree count. On each worker the model loads once, in the constructor, with its thread pool capped, and a path is fetched once per worker. Then each Arrow batch becomes one dense matrix in features= order with a null becoming NaN, the matrix goes to one model call with the chosen method=, and the output is appended as prediction columns before the next Arrow batch arrives. Nothing crosses the boundary a row at a time, and nothing is materialized on the driver.](/_static/diagrams/tabular_predict_flow.svg)

Three consequences follow.

The feature order is the contract. A tabular model scores by *position*, so the right columns in the wrong order produce confident nonsense with no error. Where the model records its own feature names, Batcher compares them and raises at plan time.

A null feature becomes NaN. XGBoost and LightGBM treat NaN as missing and learned a default direction for it during training. Pass `missing=` when your model was trained with a different sentinel.

The output schema is resolved before the query runs. Batcher reads the model's class count or tree count to decide how many columns the prediction produces. A model given as a path is opened once, and cached, to be measured.

## Score a fitted model

Pass the model and name the feature columns in training order:

```python
import batcher as bt
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]], [0, 1, 1])

ds = bt.from_pydict({"a": [0.5, 2.0], "b": [1.0, 3.0]})
scored = ds.ml.predict(model, features=["a", "b"])
print(scored.to_pydict()["prediction"])
```

The prediction is an ordinary column, so everything downstream composes:

```python
high_risk = scored.filter(bt.col("prediction") == 1)
print(high_risk.count())
```

## Choose what the model computes

`method=` means the same thing in every framework, so switching libraries doesn't mean rewriting the call. The table lists each value, what it returns, and which models support it:

| `method` | What you get | Available on |
|---|---|---|
| `"predict"` | The model's natural output: a label, or a value for a regressor. | all |
| `"predict_proba"` | Class probabilities, one column per class. | classifiers |
| `"raw"` | The untransformed margin or decision function. | all |
| `"leaf"` | The leaf index each tree routed the row to. | boosters |
| `"contrib"` | Per-feature SHAP contributions plus a bias term. | boosters |

Class probabilities become one column per class, named `prediction_0`, `prediction_1`, and so on:

```python
probabilities = ds.ml.predict(model, features=["a", "b"], method="predict_proba")
print(sorted(probabilities.columns))
```

Set `as_list=True` to get a single `List<Float64>` column instead, which is usually what you want before writing the result out.

## Load a model from storage

A path or cloud URI works wherever a model object does. The framework is detected from the file extension, and the file is fetched once per worker. Pass `framework=` when the extension is ambiguous or absent.

```python
# docs: skip
scored = ds.ml.predict(
    "s3://models/churn/booster.ubj",
    features=["tenure", "monthly_charges", "total_charges"],
)
```

## Explanations at batch scale

`method="contrib"` gives per-feature SHAP contributions for every row. A row-at-a-time explanation call can't answer that query over a whole table, and it is what turns "the model said 0.83" into "because tenure was low":

```python
# docs: skip
explained = ds.ml.predict(booster, features=feature_names, method="contrib")
top_driver = explained.select(
    *feature_names,
    driver=bt.greatest(*[bt.col(f"prediction_{i}") for i in range(len(feature_names))]),
)
```

## Scale it out

`ds.ml.predict` takes the same scheduling keywords as {py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>`, because it is the same operator underneath:

```python
# docs: skip
scored = ds.ml.predict(
    booster,
    features=feature_names,
    batch_size=100_000,
    concurrency=16,
    model_memory_gb=0.5,
)
scored.write.parquet("s3://bucket/scored/", distributed=True)
```

`batch_size` matters more here than for a GPU model. A tabular model's per-call overhead is fixed and small, so larger batches amortize it, and 100,000 rows of 50 float32 features is only 20 MB.

`threads` caps the model's own thread pool inside one worker. Left unset, Batcher sizes it to the cores the worker may use. A booster defaults to the *host* core count, so co-located workers would otherwise each grab every core and thrash.

## Fit a baseline in the engine

A linear baseline is worth having before a boosted tree, and `batcher.ml` fits one without leaving the engine. {py:class}`LinearRegression <batcher.ml.linear.LinearRegression>` and {py:class}`Ridge <batcher.ml.linear.Ridge>` build their normal equations from the feature and target moments. The fit is a single scan, prediction is a linear-combination expression, and both reproduce scikit-learn's coefficients exactly.

```python
import batcher as bt
from batcher.ml.linear import LinearRegression

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.1, 4.9, 7.0, 9.1]})
model = LinearRegression(["x"], "y").fit(ds)
print(round(model.coef_[0], 1), round(model.intercept_, 1))
# 2.0 1.0
```

`Ridge(alpha=...)` adds an L2 penalty, which stabilizes the fit when features are collinear. {py:class}`batcher.ml.sparse_linear.Lasso <batcher.ml.sparse_linear.Lasso>` and {py:class}`ElasticNet <batcher.ml.sparse_linear.ElasticNet>` go further and drive irrelevant coefficients to *exactly* zero, so they select features on a wide, correlated table. Their coordinate descent needs only the centered Gram matrix and the feature-target covariances, one scan, and the strictly convex objective means the coefficients match scikit-learn's.

The generalized linear models in `batcher.ml.glm` cover targets least squares handles badly. {py:class}`batcher.ml.glm.PoissonRegressor <batcher.ml.glm.PoissonRegressor>` fits a log-link model for a count, such as arrivals or claim frequencies, by IRLS Newton steps, keeping the predicted rate positive and matching scikit-learn's {py:class}`PoissonRegressor <batcher.ml.glm.PoissonRegressor>` across penalty strengths. {py:class}`GammaRegressor <batcher.ml.glm.GammaRegressor>` suits a positive, right-skewed amount such as a claim size or a duration. {py:class}`TweedieRegressor <batcher.ml.glm.TweedieRegressor>` is the general form of both: a `power` between 1 and 2 fits a target that is exactly zero for many rows and positive for the rest, such as an insurance pure premium.

For classification, {py:class}`RidgeClassifier <batcher.ml.linear.RidgeClassifier>` regresses on one-vs-rest targets in a closed-form single scan. {py:class}`LogisticRegression <batcher.ml.linear.LogisticRegression>` fits the probabilistic model by iteratively reweighted least squares, one scan per Newton step, and reproduces scikit-learn's unpenalized coefficients. Its `predict_proba` appends the positive-class probability, and `predict` thresholds that to a 0/1 label.

{py:class}`batcher.ml.naive_bayes.GaussianNB <batcher.ml.naive_bayes.GaussianNB>` is cheaper still. Its whole fit, a per-class prior, mean, and variance, is one {py:meth}`group_by(target) <batcher.Dataset.group_by>` aggregate, and it reproduces scikit-learn's predictions. {py:class}`MultinomialNB <batcher.ml.naive_bayes.MultinomialNB>` and {py:class}`BernoulliNB <batcher.ml.naive_bayes.BernoulliNB>` are the count-feature and binary-feature variants for text classification, fitted the same way from grouped sums.

When features are correlated within a class, the `batcher.ml.discriminant` classifiers model that covariance. {py:class}`LinearDiscriminantAnalysis <batcher.ml.discriminant.LinearDiscriminantAnalysis>` shares one covariance across classes for a linear boundary, and {py:class}`QuadraticDiscriminantAnalysis <batcher.ml.discriminant.QuadraticDiscriminantAnalysis>` gives each class its own. Both reproduce scikit-learn exactly.

## When a few rows are wrong

Squared error grows with the square of the residual, so one row off by a hundred counts as much as ten thousand rows off by one. A mistyped price or a stuck sensor tilts an ordinary fit and nothing reports it. {py:class}`HuberRegressor <batcher.ml.glm.HuberRegressor>` uses a loss that is squared near zero and linear past a threshold, which bounds a far-away row's influence:

```python
import batcher as bt
from batcher.ml import HuberRegressor, LinearRegression

readings = bt.from_pydict(
    {
        "hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "wear": [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 90.0],
    }
)

print(round(LinearRegression(["hours"], "wear").fit(readings).coef_[0], 1))
# 8.2
print(round(HuberRegressor(["hours"], "wear").fit(readings).coef_[0], 1))
# 2.0
```

Seven readings sit on a slope of about 2 and the eighth is nonsense. Least squares splits the difference at 8.2. The robust fit reports the slope the seven agree on.

The fit uses the same iteratively reweighted least squares as the GLMs, so each pass is a handful of aggregates and the whole thing distributes. `epsilon` sets where the loss turns linear, in units of the residual scale. Smaller is more robust and less efficient on clean data. The default of 1.35 keeps about 95% of least squares' efficiency when the errors really are normal, and on data with no outliers the result is what least squares returns.

The residual scale is re-estimated on every pass, because the starting residuals are already stretched by the rows being guarded against. On a degenerate input, where the retained rows fit exactly and the scale chases zero, that can hit the iteration cap. The fit warns when it stops there.

## Choose a penalty without paying for it

Choosing a ridge penalty the usual way costs a fit per candidate per fold. {py:class}`RidgeCV <batcher.ml.linear.RidgeCV>` avoids that. Ridge's normal equations come from the first and second moments of the features and target, which don't depend on the penalty, and the held-out squared error expands into the same moments. So every candidate is solved, and scored, from the same numbers.

One grouped aggregate remains: the moments per fold. Each fold's training moments are the total minus that fold's, because moments add, and everything after that is arithmetic on small matrices:

```python
import batcher as bt
from batcher.ml import RidgeCV

ds = bt.from_pydict(
    {
        "size": [750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0],
        "age": [10.0, 8.0, 12.0, 5.0, 7.0, 3.0, 9.0, 2.0],
        "price": [150.0, 162.0, 168.0, 189.0, 195.0, 214.0, 210.0, 232.0],
    }
)

model = RidgeCV(["size", "age"], "price", alphas=(0.01, 1.0, 100.0), cv=4).fit(ds)
print(model.alpha_)
# 1.0
print({a: round(s, 2) for a, s in model.scores_.items()})
# {0.01: 3.17, 1.0: 2.53, 100.0: 42.71}
print(round(model.predict(ds).to_pydict()["prediction"][0], 1))
# 150.4
```

The L1 models get the same saving, since coordinate descent also works from moments. {py:class}`LassoCV <batcher.ml.sparse_linear.LassoCV>` and {py:class}`ElasticNetCV <batcher.ml.sparse_linear.ElasticNetCV>` search a penalty path in one pass and select features as they go:

```python
import batcher as bt
from batcher.ml import LassoCV

ds = bt.from_pydict(
    {
        "tenure": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "noise": [0.3, -0.1, 0.2, -0.4, 0.1, 0.5, -0.2, 0.0],
        "spend": [12.0, 24.1, 35.9, 48.0, 60.1, 72.0, 83.9, 96.1],
    }
)

model = LassoCV(["tenure", "noise"], "spend", alphas=(0.01, 10.0), cv=4).fit(ds)
print(model.alpha_)
# 0.01
print([round(c, 2) for c in model.coef_])
# [12.0, 0.0]
```

The uninformative column comes back at exactly zero, not merely small. `ElasticNetCV` takes an `l1_ratio` to blend L1 and L2, and `LassoCV` is that class with the ratio fixed at 1.0. `scores_` holds the mean held-out squared error per candidate, and the model is refitted over all the data at `alpha_` once the search ends.

The whole search is one terminal operation, however many folds and candidates you give it. The additivity that makes this work also makes it distributable: folds are assigned by hashing each row's own values, so a row lands in the same fold however the data is partitioned. Candidates that score the same to within floating-point noise tie arbitrarily. Widen the spread of candidates rather than reading `scores_` more closely.

## More than two classes

{py:class}`LogisticRegression <batcher.ml.linear.LogisticRegression>` fits one weight vector and answers one yes-or-no question. Given a three-label target it rejects the fit and names the column.

{py:class}`OneVsRestClassifier <batcher.ml.multiclass.OneVsRestClassifier>` fits that target. It trains one binary model per class and predicts whichever scores highest. Pass the estimator as a class rather than an instance, because each sub-model needs its own target column:

```python
import batcher as bt
from batcher.ml import LogisticRegression, OneVsRestClassifier

ds = bt.from_pydict(
    {
        "weight": [0.2, 0.3, 0.4, 5.0, 5.2, 5.4, 20.0, 21.0, 22.0],
        "grade": [
            "small",
            "small",
            "small",
            "medium",
            "medium",
            "medium",
            "large",
            "large",
            "large",
        ],
    }
)

model = OneVsRestClassifier(LogisticRegression, ["weight"], "grade").fit(ds)
print(model.classes_)
# ['large', 'medium', 'small']
print(model.predict(ds).to_pydict()["prediction"])
# ['small', 'small', 'small', 'medium', 'medium', 'medium', 'large', 'large', 'large']
```

Labels can be any type. `classes_` is sorted, so the sub-model order doesn't depend on the order the scan returned labels in, and a model fitted on a cluster loads against one fitted on a laptop. Prediction stays a single pass: each sub-model's score is staged as a column and the choice is one `argmax` expression, so a hundred classes still read the data once.

Pass hyperparameters for every sub-model through `params`:

```python
model = OneVsRestClassifier(LogisticRegression, ["weight"], "grade", params={"max_iter": 50}).fit(
    ds
)
print(len(model.estimators_))
# 3
```

The base estimator must expose `predict_proba`, because ranking classes means comparing scores. An estimator without it is rejected when you construct the wrapper. {py:class}`RidgeClassifier <batcher.ml.linear.RidgeClassifier>` already decomposes a multiclass target internally and is cheaper when a closed-form fit is enough: one scan instead of one per Newton step.

A classifier's scores aren't probabilities until they're calibrated. See {doc}`/ml/inference/calibration`.

## Clustering without labels

{py:class}`batcher.ml.cluster.KMeans <batcher.ml.cluster.KMeans>` segments rows by similarity with no target. Each Lloyd iteration is one nearest-centroid assignment expression and one grouped mean, so the fit is a handful of scans and labeling is a single streaming pass. `inertia_` is the total squared distance to the centroids, the number an elbow plot uses to choose the cluster count.

```python
import batcher as bt
from batcher.ml.cluster import KMeans

ds = bt.from_pydict({"x": [0.0, 0.2, 9.8, 10.0], "y": [0.0, 0.1, 9.9, 10.1]})
km = KMeans(["x", "y"], n_clusters=2, seed=0).fit(ds)
labels = km.predict(ds).to_pydict()["cluster"]
print(labels[0] == labels[1], labels[2] == labels[3], labels[0] != labels[2])
# True True True
```

Centroids are seeded from a reproducible content-hash sample, so a fit is identical however the data is partitioned.

When clusters overlap or you want a density, {py:class}`batcher.ml.mixture.GaussianMixture <batcher.ml.mixture.GaussianMixture>` fits a blend of Gaussians by expectation-maximization. `predict` gives soft-clustering labels, `predict_proba` the membership probabilities, and `score_samples` a per-row log-likelihood you can use as an anomaly score. When the groups *are* the labels, {py:class}`batcher.ml.cluster.NearestCentroid <batcher.ml.cluster.NearestCentroid>` fits one centroid per class and labels a row by the nearest, reproducing scikit-learn's {py:class}`NearestCentroid <batcher.ml.cluster.NearestCentroid>`.

## Save a model Batcher fitted

The estimators in `batcher.ml` fit *on* the engine, so a model can be trained across a cluster. {py:func}`save_model <batcher.ml.save_model>` and {py:func}`load_model <batcher.ml.load_model>` move it to wherever it serves, and the path may be a cloud URI:

```python
import os
import tempfile

import batcher as bt
from batcher.ml import LinearRegression, load_model, save_model

train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
model = LinearRegression(["x"], "y").fit(train)

target = os.path.join(tempfile.mkdtemp(), "model.json")
save_model(model, target)

served = load_model(target)
print(served.predict(bt.from_pydict({"x": [10.0]})).to_pydict()["prediction"])
# [20.0]
```

The file is JSON, not a pickle. You can read what the model will do, a reviewer can diff it, it survives a class moving or a slot being renamed, and it is safe to load from a store you don't fully control.

```python
import json

print(sorted(json.loads(open(target).read())))
# ['class', 'params', 'state', 'version']
```

`state` holds what `fit` learned, under scikit-learn's trailing-underscore names. `params` holds the constructor arguments, read from the constructor's signature, so a parameter an estimator keeps privately, such as the `alpha` that `Ridge` stores as `_alpha`, is recorded under the name that rebuilds it. {py:func}`model_to_dict <batcher.ml.model_to_dict>` and {py:func}`model_from_dict <batcher.ml.model_from_dict>` do the same conversion without the file, for a model that travels inside a config blob, a registry row, or a message payload:

```python
from batcher.ml import model_from_dict, model_to_dict

document = model_to_dict(model)
print(document["class"], sorted(document["state"]))
# LinearRegression ['coef_', 'intercept_']
print(model_from_dict(document).coef_)
# [2.0]
```

## Fit on a reshaped target

Squared error assumes symmetric, roughly constant noise. A price, a duration, a claim amount, and a count all violate that: they're non-negative, right-skewed, and their spread grows with their level. A regression fitted directly on them spends its capacity on the long tail and under-predicts the body.

Fitting on `log1p(y)` and exponentiating back is the standard fix, and forgetting to invert at serving time is the standard mistake. The predictions then have the right shape, no error, and the wrong scale. {py:class}`TransformedTargetRegressor <batcher.ml.TransformedTargetRegressor>` wraps the pair so the inverse can't be lost:

```python
import math

import batcher as bt
from batcher.ml import LinearRegression, TransformedTargetRegressor

skewed = bt.from_pydict(
    {"x": [1.0, 2.0, 3.0, 4.0], "y": [math.expm1(v) for v in (1.0, 2.0, 3.0, 4.0)]}
)
model = TransformedTargetRegressor(LinearRegression(["x"], "y"), target="y", transform="log1p").fit(
    skewed
)
print([round(v, 3) for v in model.predict(skewed).to_pydict()["prediction"]])
# [1.718, 6.389, 19.086, 53.598]
```

The prediction comes back on the original scale, so a metric against the untransformed truth means what it says. `log1p` is the default because it is defined at zero, where a count or an amount often sits. `log` and `sqrt` are also available.

Inverting a mean in log space gives a median-like estimate on the original scale. That is usually what you want on a skewed target, but it biases the result low if you need an expectation.

## Predict from the nearest training rows

{py:class}`KNeighborsRegressor <batcher.ml.KNeighborsRegressor>` and {py:class}`KNeighborsClassifier <batcher.ml.KNeighborsClassifier>` assume nothing about the shape of the relationship. To predict a row, they find the most similar training rows and average what happened to them. That makes them the first check on whether a problem has local structure.

```python
import batcher as bt
from batcher.ml import KNeighborsClassifier

train = bt.from_pydict({"x": [0.0, 1.0, 10.0, 11.0], "label": ["low", "low", "high", "high"]})
model = KNeighborsClassifier(["x"], "label", k=2).fit(train)
print(model.predict(bt.from_pydict({"x": [0.5, 10.5]})).to_pydict()["prediction"])
# ['low', 'high']
```

A k-NN model *is* its training data. Batcher folds the reference set into the prediction as literals, the way a linear model folds in its coefficients, so scoring is one arithmetic expression over the feature columns with no join and no shuffle. It distributes unchanged.

That is also why the reference set is capped. Exact k-NN costs one distance per scored row per reference row: scoring the reference set against itself took about 0.4s at 200 rows and 4s at 1,000 on this engine. Past `max_reference` the fit fails and names the ways out. For a large corpus, use {py:func}`build_vector_index <batcher.ml.build_vector_index>`, the approximate route.

Scale the features first. Distance treats every column alike, so a column measured in millions decides every neighbour. Ties at the k-th distance all count as neighbours, so a row can have more than `k`, rather than letting arrival order break the tie.

{py:class}`KNNImputer <batcher.ml.KNNImputer>` applies the same idea to missing values. It matches a row on the columns that *are* present and fills the gap with what similar rows had:

```python
from batcher.ml import KNNImputer

homes = bt.from_pydict(
    {"size": [10.0, 11.0, 50.0, 51.0, 10.5], "price": [1.0, 1.2, 9.0, 9.4, None]}
)
print(round(KNNImputer(["size", "price"], k=2).fit_transform(homes).to_pydict()["price"][4], 3))
# 1.1
```

The column mean is about 5.15, which is the whole reason to use it. Unlike scikit-learn's imputer, a donor row must be complete across the imputed columns. The two agree wherever the neighbourhood is unambiguous.

## Requirements and limitations

Each framework is an optional extra: `pip install 'batcher-engine[xgboost]'`, `[lightgbm]`, `[catboost]`, `[onnx]`, or `[sklearn]`. `[tabular]` installs all of them.

`ds.ml.predict` feature columns must be numeric, boolean, or decimal. Encode a categorical column first, with {py:class}`OrdinalEncoder <batcher.ml.preprocessors.OrdinalEncoder>`, {py:class}`TargetEncoder <batcher.ml.preprocessors.TargetEncoder>`, or one of the encoders on {doc}`/ml/preparing/preprocessors/index`. A string column raises an error naming the column.

The estimators Batcher fits itself are stricter: a feature must be an integer, a float, or a decimal. They fit through engine aggregates, which aren't defined on a boolean, so cast a flag column first:

```python
import batcher as bt
from batcher.ml import LinearRegression

ds = bt.from_pydict(
    {"flag": [True, False, True, False], "z": [1.0, 4.0, 2.0, 9.0], "y": [0.0, 1.0, 2.0, 3.0]}
)
numeric = ds.with_columns(flag=bt.col("flag").cast("int64"))
print(len(LinearRegression(["flag", "z"], "y").fit(numeric).coef_))
# 2
```

A string, boolean, date, or all-null feature raises an error naming the column, the type, and the fix. A regressor's target must also be a number. A classifier's target is unrestricted, because a class label can be a string.

The feature-name guard only fires where the model recorded its training feature names. A booster fitted from a bare NumPy matrix records generic `f0` to `fN`, which match no real column. Fit from a DataFrame, or keep the feature list beside the model.

Under `distributed=True` a preempted worker's partition is recomputed, so scoring must be idempotent. A pure prediction is. A model wrapper that also writes to an external store is not.

## See also

- {doc}`/ml/inference/inference`: the deep-learning and HuggingFace path.
- {doc}`/ml/inference/runtimes`: ONNX Runtime, OpenVINO, and TensorRT predictors for exported models.
- {doc}`/ml/inference/calibration`: turn a classifier's scores into calibrated probabilities.
- {doc}`/ml/evaluation/evaluation`: score the predictions you just produced, per segment, in one pass.
- {doc}`/ml/evaluation/statistics-and-drift`: check that today's features still look like the training ones.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps that produce the feature columns.
- {doc}`/cookbook/ml/index`: short runnable recipes for each model family.
