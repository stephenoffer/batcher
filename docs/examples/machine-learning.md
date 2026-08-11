# Machine learning

This page covers the scripts that build features, fit models, evaluate them, and run
inference, plus the retrieval and metric scripts that sit alongside them.

## Fit and transform are separate for a reason

Every preprocessor follows the same split: the statistics come from the training set and are
then applied to validation and production data. Fitting on everything is the classic leak,
and the API makes the correct thing the easy thing.

```python
import batcher as bt
from batcher import col, ml

train = bt.from_pydict({"x": [0.0, 5.0, 10.0, 5.0]})
holdout = bt.from_pydict({"x": [2.5, 7.5]})

scaler = ml.StandardScaler("x").fit(train)

# The training half centres exactly; the holdout does not, which is the proof that the
# fitted statistics were reused rather than recomputed.
assert abs(scaler.transform(train).to_pydict()["x"][0] + 1.0) < 0.6
assert scaler.transform(holdout).count() == 2
```

A `Chain` fits its stages in order and applies them as a unit, which is the only way to be
sure the same transformations with the same fitted statistics reach production.
`examples/ml/pipeline_serving_parity.py` asserts that a single row through the chain matches
what the batch produced for that row.

## Evaluation

On an imbalanced problem a model that always predicts the majority class scores well on
accuracy and finds nothing. Precision and recall are what separate the two, and they need the
confusion counts rather than a single number.

```python
import batcher as bt
from batcher import col

scored = bt.from_pydict(
    {
        "actual": [True, True, False, False, False],
        "predicted": [True, False, False, False, False],
    }
)

counts = scored.agg(
    tp=bt.count_if(col("actual") & col("predicted")),
    fn=bt.count_if(col("actual") & ~col("predicted")),
    tn=bt.count_if(~col("actual") & ~col("predicted")),
).to_pydict()

recall = counts["tp"][0] / (counts["tp"][0] + counts["fn"][0])
assert recall == 0.5
```

## Inference

`map_batches` hands your function a whole Arrow batch, never a row, which is what keeps a
Python model call from costing a Python function call per row. Using a class rather than a
closure lets an expensive model load once per worker rather than once per batch.

One consequence to plan for: a Python callback's output schema is not known until it runs, so
the new column exists in the result but not in `Dataset.columns`. Materialize before you
project it.

## Every script on this page

The table below lists the ML and metric scripts in path order.

<!-- library-table: ml,metrics -->
| Script | Shows |
| --- | --- |
| `examples/ml/batch_inference.py` | Batch inference: a model over every row, without a Python loop |
| `examples/ml/batch_inference_udf.py` | Batch inference with a plain Python callable |
| `examples/ml/binning_features.py` | Turning a continuous feature into bins, three ways |
| `examples/ml/chunked_document_index.py` | Building a chunk-level index from document-level text |
| `examples/ml/class_balance.py` | Measuring and correcting class imbalance |
| `examples/ml/classification_pipeline.py` | Classifying order priority from order features |
| `examples/ml/classifiers.py` | Classifiers that fit in the engine: naive Bayes, discriminant analysis, and baselines |
| `examples/ml/clustering_and_decomposition.py` | Unsupervised models: KMeans, Gaussian mixtures, PCA, and truncated SVD |
| `examples/ml/clustering_customers.py` | K-means over real customer features |
| `examples/ml/cross_validation.py` | K-fold cross-validation on real data |
| `examples/ml/dimensionality_reduction.py` | PCA and truncated SVD on real numeric features |
| `examples/ml/embedding_hygiene.py` | Checking an embedding column before you index it |
| `examples/ml/encoding_categories.py` | Turning categories into numbers, four ways, and when each is wrong |
| `examples/ml/end_to_end_model_lifecycle.py` | The classical-ML lifecycle over real data, end to end |
| `examples/ml/evaluation_metrics.py` | Scoring a model: the metrics, and why accuracy alone is a trap |
| `examples/ml/feature_construction.py` | Building new features: interactions, ratios, calendar parts, lags, and rolling windows |
| `examples/ml/feature_construction_on_tpch.py` | Building model features from a real table |
| `examples/ml/feature_pipeline_chain.py` | Composing preprocessors into one fitted Chain |
| `examples/ml/glm_regressors.py` | Generalized linear models for counts, costs, and mixed zero-and-positive targets |
| `examples/ml/hybrid_search.py` | Combining a lexical and a vector ranking |
| `examples/ml/imbalance_and_sampling.py` | Class imbalance: measure it, then resample or reweight |
| `examples/ml/imputation_and_missing.py` | Filling missing values, and recording that they were missing |
| `examples/ml/inference_batching.py` | Batching an expensive per-call model so the overhead amortizes |
| `examples/ml/linear_models.py` | Regularized linear regression: Ridge, Lasso, and ElasticNet |
| `examples/ml/model_persistence.py` | Saving a fitted model and loading it back |
| `examples/ml/model_selection.py` | Cross-validation, learning curves, and feature importance -- all in the engine |
| `examples/ml/outlier_detection.py` | Finding outliers: per-column rules and a multivariate distance |
| `examples/ml/outlier_detection_on_tpch.py` | Finding outliers, and deciding what to do about them |
| `examples/ml/pipeline_serving_parity.py` | Training-serving parity: the same transformations over one row and over millions |
| `examples/ml/preprocessing_binning.py` | Discretizing, clipping, and reshaping the distribution of a numeric column |
| `examples/ml/preprocessing_chain.py` | Chaining preprocessors into one fitted pipeline |
| `examples/ml/preprocessing_encoding.py` | Turning categories into numbers, and picking the encoder by cardinality |
| `examples/ml/preprocessing_imputation.py` | Filling missing values, and keeping the fact that they were missing |
| `examples/ml/preprocessing_scaling.py` | Scaling numeric features, and why the choice of scaler matters |
| `examples/ml/regression_on_real_data.py` | Fitting a regression on real TPC-H lineitems, end to end |
| `examples/ml/reranking.py` | Reranking a candidate set: cheap retrieval, then an expensive score |
| `examples/ml/scaling_comparison.py` | Four scalers on a skewed real column, and what each does to the outliers |
| `examples/ml/similarity_join.py` | Joining two sets of vectors by similarity rather than by key |
| `examples/ml/splits_and_leakage.py` | Splitting data without leaking: random, stratified, grouped, and by time |
| `examples/ml/text_features.py` | Turning raw text into model-ready features without a model |
| `examples/ml/text_features_on_tpch.py` | Turning a text column into numeric features without a model |
| `examples/ml/vector_search.py` | Vector search over an embedding column, in the engine |
| `examples/ml/vector_search_over_real_text.py` | Nearest-neighbour search over an embedding column |
| `examples/metrics/agreement.py` | Agreement metrics: how well a prediction tracks the truth, not just how close |
| `examples/metrics/classification.py` | Classification metrics computed as aggregates over a predictions table |
| `examples/metrics/diagnostic.py` | Diagnostic metrics: the epidemiology-style view of a binary classifier |
| `examples/metrics/embeddings.py` | Corpus-level embedding metrics: monitoring a vector column in aggregate |
| `examples/metrics/probabilistic_losses.py` | Losses that score a probability or a margin rather than a hard label |
| `examples/metrics/regression_errors.py` | Regression error metrics: absolute, squared, percentage, and robust |
| `examples/metrics/text_diversity.py` | Degeneracy detection: repetition, truncation, refusal, and empty output |
| `examples/metrics/text_formatting.py` | Did the model obey the output format you asked for? |
| `examples/metrics/text_length.py` | Length and readability distribution over a text column |
| `examples/metrics/text_overlap.py` | Comparing a generated answer against a reference, without a model |
| `examples/metrics/text_pii_safety.py` | PII leak rates over a text column |
| `examples/metrics/text_quality.py` | Corpus hygiene rates: what fraction of a text column looks broken |
| `examples/metrics/text_retrieval.py` | RAG groundedness: is the answer actually supported by the retrieved context? |
| `examples/metrics/text_tone_and_script.py` | Tone and writing-system rates: style drift and language mix |
<!-- /library-table -->
