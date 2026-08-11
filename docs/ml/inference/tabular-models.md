# Score a tabular model

This page describes how to run a fitted XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX model over a Batcher {py:class}`Dataset <batcher.Dataset>`.

Tabular models are the ones most production ML actually runs, and they have a different shape from a language model. The input is dozens of numeric columns rather than one text column, the model is megabytes rather than gigabytes, and the bottleneck is feeding the model rather than the model itself. {py:meth}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>` is the entry point built for that shape.

## How it works

`ds.ml.predict` builds a load-once class UDF. The model is constructed once per worker, and each Arrow batch is assembled into one dense `(rows, features)` matrix that goes to the model in a single call. Nothing crosses the boundary a row at a time, and nothing is materialized on the driver.

Three consequences are worth knowing before you use it:

The feature order is the contract. A tabular model scores by *position*, not by name, so passing the right columns in the wrong order produces confident nonsense with no error anywhere. Where the model records its own feature names, Batcher compares them and raises at plan time rather than letting the query run.

A null feature becomes NaN, which is what XGBoost and LightGBM treat as missing and what they learned a default direction for during training. Pass `missing=` when your model was trained with a different sentinel.

The output schema is resolved before the query runs. Batcher reads the model's own class count or tree count to decide how many columns the prediction produces; if the model is given as a path, it is opened once (cached) to be measured rather than assumed to be single-output.

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

The prediction is appended as an ordinary column, so everything downstream composes:

```python
high_risk = scored.filter(bt.col("prediction") == 1)
print(high_risk.count())
```

## Choose what the model computes

`method=` is uniform across every framework, so switching model libraries does not mean rewriting the call:

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

Set `as_list=True` when you would rather have a single `List<Float64>` column, which is usually what you want before writing the result out.

## Load a model from storage

A path or cloud URI works wherever a model object does. The framework is detected from the file extension, and the file is fetched once per worker:

```python
# docs: skip
scored = ds.ml.predict(
    "s3://models/churn/booster.ubj",
    features=["tenure", "monthly_charges", "total_charges"],
)
```

Pass `framework=` when the extension is ambiguous or absent.

## Explanations at batch scale

`method="contrib"` gives per-feature SHAP contributions for every row. That is the query a row-at-a-time explanation call cannot answer, and it is what turns "the model said 0.83" into "because tenure was low":

```python
# docs: skip
explained = ds.ml.predict(booster, features=feature_names, method="contrib")
top_driver = explained.select(
    *feature_names,
    driver=bt.max_horizontal(*[bt.col(f"prediction_{i}") for i in range(len(feature_names))]),
)
```

## Scale it out

The scheduling keywords are the same ones {py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>` takes, because it is the same operator underneath:

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

`batch_size` matters more here than for a GPU model: a tabular model's per-call overhead is fixed and small, so larger batches amortize it, and 100,000 rows of 50 float32 features is only 20 MB.

`threads` caps the model's own thread pool inside one worker. Leave it unset and Batcher sizes it to the cores the worker may actually use. A booster's own default is the *host* core count, so several co-located workers would otherwise each grab every core and thrash.

## A linear baseline for free

Before reaching for a boosted tree, a linear baseline is worth having, and `batcher.ml.linear` fits one without leaving the engine: {py:class}`LinearRegression <batcher.ml.linear.LinearRegression>` and {py:class}`Ridge <batcher.ml.linear.Ridge>` build their normal equations from the feature and target moments, so the whole fit is a single scan and prediction is a linear-combination expression. Both reproduce scikit-learn's coefficients exactly.

```python
import batcher as bt
from batcher.ml.linear import LinearRegression

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.1, 4.9, 7.0, 9.1]})
model = LinearRegression(["x"], "y").fit(ds)
print(round(model.coef_[0], 1), round(model.intercept_, 1))
# 2.0 1.0
```

`Ridge(alpha=...)` adds an L2 penalty that stabilizes the fit when features are collinear, which is the case where ordinary least squares is unreliable. Encode categorical columns to numbers first, as for every other model here.

Where ridge shrinks every coefficient, {py:class}`batcher.ml.sparse_linear.Lasso <batcher.ml.sparse_linear.Lasso>` and {py:class}`ElasticNet <batcher.ml.sparse_linear.ElasticNet>` drive the irrelevant ones to *exactly* zero, so the fit selects features as it trains. That makes them the tool for a wide, correlated table. Their coordinate descent needs only the centered Gram matrix and the feature-target covariances (one scan), and because the objective is strictly convex the coefficients match scikit-learn's.

For a count target such as events, arrivals, or claim frequencies, {py:class}`batcher.ml.glm.PoissonRegressor <batcher.ml.glm.PoissonRegressor>` fits a log-link generalized linear model by the same IRLS Newton steps, keeping the predicted rate positive where least squares would predict a negative count. It matches scikit-learn's {py:class}`PoissonRegressor <batcher.ml.glm.PoissonRegressor>` across penalty strengths. For a positive, right-skewed *amount* such as a claim size, a duration, or a spend, {py:class}`GammaRegressor <batcher.ml.glm.GammaRegressor>` is the matching GLM, with a variance that grows with the mean. {py:class}`TweedieRegressor <batcher.ml.glm.TweedieRegressor>` is the general form these two specialize: a `power` between 1 and 2 fits the compound distribution of a target that is exactly zero for many rows and positive for the rest, such as an insurance pure premium.

For classification, {py:class}`RidgeClassifier <batcher.ml.linear.RidgeClassifier>` casts it as regression on one-vs-rest targets (a closed-form single-scan fit), while {py:class}`LogisticRegression <batcher.ml.linear.LogisticRegression>` fits the probabilistic model by iteratively reweighted least squares, one scan per Newton step, and reproduces scikit-learn's unpenalized coefficients. `predict_proba` appends the positive-class probability, and `predict` thresholds it to a 0/1 label.

{py:class}`batcher.ml.naive_bayes.GaussianNB <batcher.ml.naive_bayes.GaussianNB>` is the even cheaper baseline. Its whole fit, a per-class prior, mean, and variance, is a single {py:meth}`group_by(target) <batcher.Dataset.group_by>` aggregate, and it reproduces scikit-learn's predictions. It is naive by assumption but a strong instant baseline in high dimensions. {py:class}`MultinomialNB <batcher.ml.naive_bayes.MultinomialNB>` and {py:class}`BernoulliNB <batcher.ml.naive_bayes.BernoulliNB>` are the count-feature and binary-feature variants that do the text-classification work, fitted the same way from grouped sums.

When the features are correlated within a class, the `batcher.ml.discriminant` classifiers model that covariance instead of assuming independence: {py:class}`LinearDiscriminantAnalysis <batcher.ml.discriminant.LinearDiscriminantAnalysis>` shares one covariance across classes for a stable linear boundary, and {py:class}`QuadraticDiscriminantAnalysis <batcher.ml.discriminant.QuadraticDiscriminantAnalysis>` gives each class its own for a quadratic one. Both reproduce scikit-learn exactly.

## When a few rows are wrong

Squared error grows with the square of the residual, so one row that is off by a hundred counts as much as ten thousand rows off by one. A single mistyped price or a stuck sensor visibly tilts an ordinary fit, and nothing reports it. {py:class}`HuberRegressor <batcher.ml.glm.HuberRegressor>` uses a loss that is squared near zero and linear past a threshold, so a far-away row keeps a bounded influence:

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

Seven readings sit on a slope of about 2 and the eighth is nonsense. Least squares splits the difference at 8.2; the robust fit reports the slope the seven agree on.

The fit is the same iteratively reweighted least squares the GLMs use, so each pass is a handful of aggregates and the whole thing distributes. `epsilon` sets where the loss turns linear, in units of the residual scale: smaller is more robust and less efficient on clean data, and the default of 1.35 keeps about 95% of least squares' efficiency when the errors really are normal. On data with no outliers it returns what least squares returns.

The residual scale is re-estimated on every pass rather than fixed from the starting fit, because those starting residuals are already stretched by the rows being guarded against. That can hit the iteration cap on a degenerate input, where the retained rows fit exactly and the scale chases zero; the fit warns when it stops on the cap rather than presenting the last iterate as an optimum.

## Choosing a penalty without paying for it

A ridge penalty has to be chosen, and the usual way costs a fit per candidate per fold. {py:class}`RidgeCV <batcher.ml.linear.RidgeCV>` does not need that. Ridge builds its normal equations from the first and second moments of the features and the target, and those moments do not depend on the penalty, so every candidate is solved from the same numbers. The held-out squared error expands into the same moments, so scoring a candidate on a fold reads no rows either.

What remains is one grouped aggregate: the moments per fold. Each fold's training moments are the total minus that fold's, because moments add, and every combination is then arithmetic on small matrices:

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

The same saving applies to the L1 models, for the same reason: coordinate descent works from the centered Gram matrix and the feature-target covariances, which are moments too. {py:class}`LassoCV <batcher.ml.sparse_linear.LassoCV>` and {py:class}`ElasticNetCV <batcher.ml.sparse_linear.ElasticNetCV>` search a penalty path in one pass, and because the penalty is L1 the search selects features as it goes:

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

The uninformative column comes back at exactly zero rather than merely small, which is what separates L1 from a ridge penalty. `ElasticNetCV` takes an `l1_ratio` to blend the two, and `LassoCV` is that class with the ratio fixed at 1.0.

`scores_` holds the mean held-out squared error per candidate, and the model is refitted over all the data at `alpha_` once the search is done.

Five folds against twenty candidates is one pass rather than a hundred, and adding candidates costs nothing extra: on 200,000 rows with five features, the search runs one terminal operation where refitting each combination runs 125. The additivity that makes this work is the same property that makes the operator distributable, so the search behaves identically on one node and across a cluster, and folds are assigned by hashing each row's own values so a row lands in the same fold however the data is partitioned.

Candidates that are statistically indistinguishable can tie. When two penalties score the same to within floating-point noise, which one wins is arbitrary, and the fix is a wider spread of candidates rather than a closer reading of `scores_`.

## More than two classes

{py:class}`LogisticRegression <batcher.ml.linear.LogisticRegression>` fits one weight vector, so it can only answer one yes-or-no question. Given a target with three labels it rejects the fit and names the column, rather than returning a model that predicts a single class for every row.

{py:class}`OneVsRestClassifier <batcher.ml.multiclass.OneVsRestClassifier>` is the way to fit that target. It trains one binary model per class, each asking whether a row belongs to that class, and predicts whichever scores highest. Pass the estimator as a class rather than an instance, because each sub-model needs its own target column:

```python
import batcher as bt
from batcher.ml import LogisticRegression, OneVsRestClassifier

ds = bt.from_pydict(
    {
        "weight": [0.2, 0.3, 0.4, 5.0, 5.2, 5.4, 20.0, 21.0, 22.0],
        "grade": ["small", "small", "small", "medium", "medium", "medium",
                  "large", "large", "large"],
    }
)

model = OneVsRestClassifier(LogisticRegression, ["weight"], "grade").fit(ds)
print(model.classes_)
# ['large', 'medium', 'small']
print(model.predict(ds).to_pydict()["prediction"])
# ['small', 'small', 'small', 'medium', 'medium', 'medium', 'large', 'large', 'large']
```

The labels can be of any type, and `classes_` is sorted so that the sub-model order does not depend on the order the scan returned the labels in. That is what lets a model fitted across a cluster be saved and loaded against one fitted on a laptop.

Prediction stays a single pass however many classes there are. Each sub-model's score is staged as a column and the choice between them is folded into one `argmax` expression, so classifying against a hundred classes reads the data once rather than a hundred times.

Pass hyperparameters for every sub-model through `params`:

```python
model = OneVsRestClassifier(
    LogisticRegression, ["weight"], "grade", params={"max_iter": 50}
).fit(ds)
print(len(model.estimators_))
# 3
```

The base estimator must expose `predict_proba`. Ranking classes means comparing their scores, which a hard 0/1 label cannot support, so an estimator without it is rejected when you construct the wrapper rather than when you predict.

{py:class}`RidgeClassifier <batcher.ml.linear.RidgeClassifier>` already does this decomposition internally and takes a multiclass target directly. It is the cheaper option when a closed-form fit is enough, since it needs one scan rather than one per Newton step.

A fitted classifier's scores are not probabilities until they are calibrated; see
{doc}`/ml/inference/calibration` for when that matters and how to fix it.

## Clustering without labels

Not every tabular job has a target. {py:class}`batcher.ml.cluster.KMeans <batcher.ml.cluster.KMeans>` segments rows by similarity, learning its centroids in the engine: each Lloyd iteration is one nearest-centroid assignment expression and one grouped mean, so the fit is a handful of scans and labeling any dataset is a single streaming pass. The `inertia_` it learns is the total squared distance to the centroids, which is the number an elbow plot uses to choose the cluster count.

```python
import batcher as bt
from batcher.ml.cluster import KMeans

ds = bt.from_pydict({"x": [0.0, 0.2, 9.8, 10.0], "y": [0.0, 0.1, 9.9, 10.1]})
km = KMeans(["x", "y"], n_clusters=2, seed=0).fit(ds)
labels = km.predict(ds).to_pydict()["cluster"]
print(labels[0] == labels[1], labels[2] == labels[3], labels[0] != labels[2])
# True True True
```

The centroids are seeded from a reproducible content-hash sample, so a fit is identical however the data is partitioned. Encode categorical columns to numbers first, exactly as for the predictors above.

When clusters overlap or the goal is a density rather than a partition, {py:class}`batcher.ml.mixture.GaussianMixture <batcher.ml.mixture.GaussianMixture>` models the data as a blend of Gaussians fitted by expectation-maximization: `predict` gives soft-clustering labels, `predict_proba` the membership probabilities, and `score_samples` a per-row log-likelihood that turns the fitted model into an anomaly detector.

When the groups *are* the labels, {py:class}`batcher.ml.cluster.NearestCentroid <batcher.ml.cluster.NearestCentroid>` is the supervised counterpart: it fits one centroid per class and labels a row by the nearest, reproducing scikit-learn's {py:class}`NearestCentroid <batcher.ml.cluster.NearestCentroid>`.

## Save a model Batcher fitted

The estimators in `batcher.ml` fit *on* the engine, so a model can be trained across a
cluster. {py:func}`save_model <batcher.ml.save_model>` and
{py:func}`load_model <batcher.ml.load_model>` are what move it afterwards — without them the
only route from a fitted model to a prediction is to fit again, which is not a serving
story.

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

The path may be a cloud URI, because a fitted model belongs next to the data it scores
rather than on the machine that fitted it.

What is written is JSON, not a pickle, and for the same reasons the preprocessors use JSON:
you can read what the model will do, a reviewer can diff it, it survives a class moving or a
slot being renamed, and it is safe to load from a store you do not fully control.

```python
import json

print(sorted(json.loads(open(target).read())))
# ['class', 'params', 'state', 'version']
```

`state` holds what `fit` learned, under scikit-learn's trailing-underscore names, and
{py:func}`model_to_dict <batcher.ml.model_to_dict>` and
{py:func}`model_from_dict <batcher.ml.model_from_dict>` are the same conversion without the
file, for when the model travels inside something else — a config blob, a registry row, a
message payload:

```python
from batcher.ml import model_from_dict, model_to_dict

document = model_to_dict(model)
print(document["class"], sorted(document["state"]))
# LinearRegression ['coef_', 'intercept_']
print(model_from_dict(document).coef_)
# [2.0]
```

`params` holds the constructor arguments — read from the constructor's own signature, so a
parameter an estimator keeps privately (`Ridge` takes `alpha` and stores `_alpha`) is still
recorded under the name that rebuilds it.

## Fitting on a reshaped target

Squared error assumes the target's noise is symmetric and roughly constant. A price, a
duration, a claim amount and a count all violate that: they are non-negative, right-skewed,
and their spread grows with their level, so a regression fitted directly on them spends its
capacity on the long tail and under-predicts the body.

Fitting on `log1p(y)` and exponentiating back is the standard fix, and the third step —
remembering to invert at serving time — is the one that gets forgotten. Predictions are then
wrong by a factor of *e*, with the right shape and no error.
{py:class}`TransformedTargetRegressor <batcher.ml.TransformedTargetRegressor>` wraps the pair
so the inverse cannot be lost:

```python
import math

import batcher as bt
from batcher.ml import LinearRegression, TransformedTargetRegressor

skewed = bt.from_pydict(
    {"x": [1.0, 2.0, 3.0, 4.0], "y": [math.expm1(v) for v in (1.0, 2.0, 3.0, 4.0)]}
)
model = TransformedTargetRegressor(
    LinearRegression(["x"], "y"), target="y", transform="log1p"
).fit(skewed)
print([round(v, 3) for v in model.predict(skewed).to_pydict()["prediction"]])
# [1.718, 6.389, 19.086, 53.598]
```

The prediction comes back on the original scale, so a metric computed against the
untransformed truth means what it says — comparing a model fitted on `log1p(y)` against one
fitted on `y` is otherwise comparing two different quantities and calling the smaller number
better.

`log1p` is the default because, unlike a bare `log`, it is defined at zero, which is where a
count or an amount most often sits. `log` and `sqrt` are also available.

One property worth knowing: inverting a mean in log space gives a median-like estimate on
the original scale, not a mean. That is the accepted behaviour of the technique and usually
what you want on a skewed target, but it biases the result low if you need an expectation.

## Predicting from the nearest training rows

{py:class}`KNeighborsRegressor <batcher.ml.KNeighborsRegressor>` and
{py:class}`KNeighborsClassifier <batcher.ml.KNeighborsClassifier>` assume nothing about the
shape of the relationship: to predict a row, they find the training rows most like it and
average what happened to them. That makes them the natural first check on whether a problem
has local structure, and the right tool when a boundary is genuinely irregular.

```python
import batcher as bt
from batcher.ml import KNeighborsClassifier

train = bt.from_pydict(
    {"x": [0.0, 1.0, 10.0, 11.0], "label": ["low", "low", "high", "high"]}
)
model = KNeighborsClassifier(["x"], "label", k=2).fit(train)
print(model.predict(bt.from_pydict({"x": [0.5, 10.5]})).to_pydict()["prediction"])
# ['low', 'high']
```

A k-NN model has no parameters — it *is* its training data — so `fit` keeps a reference set
and `predict` measures against it. Batcher folds that reference set into the prediction as
literals, exactly the way a fitted linear model folds in its coefficients, so scoring is one
arithmetic expression over the feature columns with no join and no shuffle.

That is what makes it distribute unchanged, and it is also why the reference set is capped.
Exact k-NN costs one distance per scored row per reference row, and nothing removes that:
measured on this engine, scoring the reference set against itself takes about 0.4s at 200
rows and 4s at 1,000. Past `max_reference` the fit fails and names the ways out rather than
building a query nobody wants to wait for.

Two habits matter more here than for most models:

- **Scale the features first.** Distance treats every column alike, so a column measured in
  millions decides every neighbour and one measured in fractions is ignored.
- **Reach for an index when the corpus is large.**
  {py:func}`build_vector_index <batcher.ml.build_vector_index>` is the approximate route;
  a broadcast reference set is not.

Ties at the k-th distance all count as neighbours, so a row can have more than `k` of them.
The alternative would be to break the tie by reference-set order, which makes a prediction
depend on the order rows happened to arrive in.

{py:class}`KNNImputer <batcher.ml.KNNImputer>` applies the same idea to missing values: it
matches a row on the columns that *are* present and fills the gap with what similar rows
had there.

```python
from batcher.ml import KNNImputer

homes = bt.from_pydict(
    {"size": [10.0, 11.0, 50.0, 51.0, 10.5], "price": [1.0, 1.2, 9.0, 9.4, None]}
)
print(round(KNNImputer(["size", "price"], k=2).fit_transform(homes).to_pydict()["price"][4], 3))
# 1.1
```

The column mean there is about 5.15, so the gap between the two is the whole reason to use
it. Unlike scikit-learn's, a donor row must be complete across the imputed columns; the two
agree wherever the neighbourhood is unambiguous.

## Requirements and limitations

Each framework is an optional extra: `pip install 'batcher-engine[xgboost]'`, `[lightgbm]`, `[catboost]`, `[onnx]`, or `[sklearn]`. `[tabular]` installs all of them.

Feature columns must be numeric, boolean, or decimal. Encode a categorical column first, with {py:class}`OrdinalEncoder <batcher.ml.preprocessors.OrdinalEncoder>`, {py:class}`TargetEncoder <batcher.ml.preprocessors.TargetEncoder>`, or one of the cardinality-tolerant encoders on {doc}`/ml/preparing/preprocessors/index`. A string column raises an error naming the column rather than failing deep inside the model.

The estimators Batcher fits itself are stricter than `ds.ml.predict` on one point: a feature column must be an integer, a float, or a decimal. They fit through engine aggregates, which are not defined on a boolean, so a flag column has to be cast before it can be used as a feature:

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

A string, boolean, date, or all-null feature raises an error naming the column, the type, and the fix, rather than surfacing as an aggregate or cast failure from inside the engine. The same applies to the target of a regressor, which must also be a number. A classifier's target is unrestricted, because a class label is legitimately a string.

The feature-name guard only fires where the model recorded its training feature names. A booster fitted from a bare NumPy matrix records generic `f0…fN`, which match no real column, so nothing can be checked. Fit from a DataFrame, or keep the feature list beside the model.

Under `distributed=True` a preempted worker's partition is recomputed, so scoring must be idempotent. A pure prediction is; a `fn` that also writes to an external store is not.

## See also

- {doc}`/ml/evaluation/evaluation`: score the predictions you just produced, per segment, in one pass.
- {doc}`/ml/preparing/preprocessors/index`: the fit and transform steps that produce the feature columns.
- {doc}`/ml/evaluation/statistics-and-drift`: check that today's features still look like the training ones.
- {doc}`/ml/inference/inference`: the deep-learning and HuggingFace path.
- {doc}`/cookbook/ml/index`: short runnable recipes for each model family.
