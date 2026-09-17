# MLflow

This page describes how to score a model straight out of an MLflow registry.

A team that trains with MLflow refers to a model the way MLflow does: `models:/churn/3`, `models:/churn@champion`, or `runs:/<run-id>/model`. That reference survives retraining, carries a version or an alias, and is what the rest of the platform is already configured with. {py:meth}`ds.ml.predict <batcher.api.dataset.ml.DatasetML.predict>` takes it directly, with no download step and no framework-specific loader in your pipeline.

## Score by URI

Pass the reference where you would pass a model object or a file path:

```python
# docs: skip
import batcher as bt

scored = bt.read.parquet("s3://data/customers/*.parquet").ml.predict(
    model="models:/churn/3",
    features=["tenure", "monthly_charges", "contract_type"],
    output_column="churn_score",
)
```

Aliases work the same way, which is usually what you want in production so a promotion does
not mean editing a pipeline:

```python
# docs: skip
ds.ml.predict(model="models:/churn@champion", features=[...], output_column="score")
```

## Where the model is loaded

The model loads once per worker, on the worker that scores with it, against that machine's own MLflow configuration. Only the URI travels in the plan.

That matters on both ends. A distributed run resolves the reference on each worker, so
a worker authenticates to the tracking server as itself rather than inheriting a session from
the driver. The artifact is never copied through the driver either, so a large model costs one download per worker rather than a download plus a fan-out.

Set `MLFLOW_TRACKING_URI` in the worker environment, or configure MLflow in your host
process before the query runs.

## Feature names and output width

Batcher reads the model's logged signature, when it has one.

Input column names are checked against the `features=` you pass, so a pipeline that feeds columns in a different order from training fails rather than returning wrong scores. Log with an `input_example` so a signature is recorded.

The output width comes from the same signature. A model logged from a DataFrame or an array
records a tensor output of shape `(-1,)`, which is one value per row, so `output_column=`
is enough. A model logged without a signature has no recorded width, and Batcher asks for
`output_columns=[...]` or `as_list=True` rather than guessing a width the plan would then
fail on at execution.

## Requirements and limitations

Loading goes through the `pyfunc` flavor, which every logged model has, so one path covers
scikit-learn, XGBoost, LightGBM, PyTorch, and a custom `PythonModel` alike.

`pyfunc` exposes only `predict`. `method="predict_proba"` and
`method="contrib"` are framework-specific entry points that `pyfunc` does not offer, so
they are refused rather than approximated. To use one, name the framework and point at the
artifact directly:

```python
# docs: skip
ds.ml.predict(
    model="s3://models/churn.json",
    framework="xgboost",
    method="predict_proba",
    features=[...],
    output_columns=["p_stay", "p_churn"],
)
```

Batcher doesn't log to MLflow. It scores models the registry holds, and producing runs and registering models stays with your training code.

## See also

- {doc}`/integrations/compute/pytorch`: scoring a PyTorch model directly.
- {doc}`/integrations/compute/huggingface`: scoring a Hub model by id.
- {doc}`/ml/index`: the wider batch-inference surface.
