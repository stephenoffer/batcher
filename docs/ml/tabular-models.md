# Score a tabular model

This page describes how to run a fitted XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX model over a Batcher `Dataset`.

Tabular models are the ones most production ML actually runs, and they have a different shape from a language model: the input is dozens of numeric columns rather than one text column, the model is megabytes rather than gigabytes, and the bottleneck is feeding the model rather than the model itself. `ds.ml.predict` is the entry point built for that shape.

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
| `"predict"` | The model's natural output — a label, or a value for a regressor. | all |
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

The scheduling keywords are the same ones `ds.ml.infer` takes, because it is the same operator underneath:

```python
# docs: skip
scored = ds.ml.predict(
    booster,
    features=feature_names,
    batch_size=100_000,
    concurrency=16,
    model_memory_gb=0.5,
)
scored.write_parquet("s3://bucket/scored/", distributed=True)
```

`batch_size` matters more here than for a GPU model: a tabular model's per-call overhead is fixed and small, so larger batches amortize it, and 100,000 rows of 50 float32 features is only 20 MB.

`threads` caps the model's own thread pool inside one worker. Leave it unset and Batcher sizes it to the cores the worker may actually use — a booster's own default is the *host* core count, so several co-located workers otherwise each grab every core and thrash.

## A linear baseline for free

Before reaching for a boosted tree, a linear baseline is worth having, and `batcher.ml.linear` fits one without leaving the engine: `LinearRegression` and `Ridge` build their normal equations from the feature and target moments, so the whole fit is a single scan and prediction is a linear-combination expression. Both reproduce scikit-learn's coefficients exactly.

```python
import batcher as bt
from batcher.ml.linear import LinearRegression

ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [3.1, 4.9, 7.0, 9.1]})
model = LinearRegression(["x"], "y").fit(ds)
print(round(model.coef_[0], 1), round(model.intercept_, 1))
# 2.0 1.0
```

`Ridge(alpha=...)` adds an L2 penalty that stabilizes the fit when features are collinear — the case where ordinary least squares is unreliable. Encode categorical columns to numbers first, as for every other model here.

Where ridge shrinks every coefficient, `batcher.ml.sparse_linear.Lasso` and `ElasticNet` drive the irrelevant ones to *exactly* zero, so the fit selects features as it trains — the tool for a wide, correlated table. Their coordinate descent needs only the centered Gram matrix and the feature-target covariances (one scan), and because the objective is strictly convex the coefficients match scikit-learn's.

For a count target — events, arrivals, claim frequencies — `batcher.ml.glm.PoissonRegressor` fits a log-link generalized linear model by the same IRLS Newton steps, keeping the predicted rate positive where least squares would predict a negative count. It matches scikit-learn's `PoissonRegressor` across penalty strengths. For a positive, right-skewed *amount* — a claim size, a duration, a spend — `GammaRegressor` is the matching GLM, with a variance that grows with the mean. `TweedieRegressor` is the general form these two specialize: a `power` between 1 and 2 fits the compound distribution of a target that is exactly zero for many rows and positive for the rest, such as an insurance pure premium.

For classification, `RidgeClassifier` casts it as regression on one-vs-rest targets (a closed-form single-scan fit), while `LogisticRegression` fits the probabilistic model by iteratively reweighted least squares — each Newton step is one scan — and reproduces scikit-learn's unpenalized coefficients. `predict_proba` appends the positive-class probability; `predict` thresholds it to a 0/1 label.

`batcher.ml.naive_bayes.GaussianNB` is the even cheaper baseline: its whole fit — a per-class prior, mean, and variance — is a single `group_by(target)` aggregate, and it reproduces scikit-learn's predictions. Naive by assumption, but a strong instant baseline in high dimensions. `MultinomialNB` and `BernoulliNB` are the count-feature and binary-feature variants (the text-classification workhorses), fitted the same way from grouped sums.

When the features are correlated within a class, the `batcher.ml.discriminant` classifiers model that covariance instead of assuming independence: `LinearDiscriminantAnalysis` shares one covariance across classes for a stable linear boundary, and `QuadraticDiscriminantAnalysis` gives each class its own for a quadratic one. Both reproduce scikit-learn exactly.

## Clustering without labels

Not every tabular job has a target. `batcher.ml.cluster.KMeans` segments rows by similarity, learning its centroids in the engine: each Lloyd iteration is one nearest-centroid assignment expression and one grouped mean, so the fit is a handful of scans and labeling any dataset is a single streaming pass. The `inertia_` it learns is the total squared distance to the centroids, which is the number an elbow plot uses to choose the cluster count.

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

When clusters overlap or the goal is a density rather than a partition, `batcher.ml.mixture.GaussianMixture` models the data as a blend of Gaussians fitted by expectation-maximization: `predict` gives soft-clustering labels, `predict_proba` the membership probabilities, and `score_samples` a per-row log-likelihood that turns the fitted model into an anomaly detector.

When the groups *are* the labels, `batcher.ml.cluster.NearestCentroid` is the supervised counterpart: it fits one centroid per class and labels a row by the nearest, reproducing scikit-learn's `NearestCentroid`.

## Requirements and limitations

Each framework is an optional extra: `pip install 'batcher-engine[xgboost]'`, `[lightgbm]`, `[catboost]`, `[onnx]`, or `[sklearn]`. `[tabular]` installs all of them.

Feature columns must be numeric, boolean, or decimal. Encode a categorical column first, with `OrdinalEncoder`, `TargetEncoder`, or one of the cardinality-tolerant encoders on {doc}`preprocessors`. A string column raises an error naming the column rather than failing deep inside the model.

The feature-name guard only fires where the model recorded its training feature names. A booster fitted from a bare NumPy matrix records generic `f0…fN`, which match no real column, so nothing can be checked. Fit from a DataFrame, or keep the feature list beside the model.

Under `distributed=True` a preempted worker's partition is recomputed, so scoring must be idempotent. A pure prediction is; a `fn` that also writes to an external store is not.

## See also

- {doc}`evaluation` — score the predictions you just produced, per segment, in one pass.
- {doc}`preprocessors` — the fit/transform steps that produce the feature columns.
- {doc}`statistics-and-drift` — check that today's features still look like the training ones.
- {doc}`inference` — the deep-learning and HuggingFace path.
