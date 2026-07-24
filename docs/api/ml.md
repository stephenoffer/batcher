# The ML accessor

This page covers the `.ml` accessor on a `Dataset` and the `batcher.ml` package behind it. For the relational surface these compose with, see [Dataset](dataset.md).

ML work attaches to a `Dataset` through the `.ml` accessor:

| Method | Use |
| --- | --- |
| `ds.ml.map_batches(fn, ...)` | Apply an arbitrary function to each Arrow batch. |
| `ds.ml.infer(model, ...)` | Run batch inference from a model id plus `column`, or from a model callable. |
| `ds.ml.embed(model, ...)` | Generate embeddings from a model id plus `column`, or from a model callable. |
| `ds.ml.generate(engine, ...)` | Offline LLM text generation, appending the response column. |
| `ds.ml.download(url_col, ...)` | Fetch bytes at each URL/path into a column. |
| `ds.ml.upload(data_col, dir, ...)` | Write a bytes column out to object storage. |
| `ds.ml.iter_torch_batches(...)` | Stream the dataset to PyTorch as tensor batches. |
| `ds.ml.stream_loader(...)` | A distributed-training `IterableDataset` for one rank. |
| `ds.ml.train_test_split(test_size, seed=0)` | Disjoint, reproducible train/test `Dataset`s. |
| `ds.ml.random_split(fractions, seed=0)` | The n-way generalization (train/val/test). |
| `ds.ml.near_duplicates(column, threshold=0.8)` | MinHash + LSH near-duplicate pairs. |
| `ds.ml.drop_near_duplicates(column, threshold=0.8)` | Fuzzy dedup, keeping one per cluster. |
| `ds.ml.nearest_neighbors(query, column="embedding", k=10, metric="cosine")` | Exact brute-force top-`k` retrieval against a query vector. |
| `ds.ml.similarity_to(query, column="embedding", metric="cosine")` | Score every row against a query vector (no top-`k` cut). |
| `ds.ml.normalize_embeddings(column, output_column=None)` | Unit-normalize an embedding column (L2 = 1). |

These operate on whole `pyarrow.RecordBatch` objects, never on individual rows. They're lazy, as every other transformation is, and return a new `Dataset`. The loaders are the exception, returning a torch iterator.

## Whole-batch semantics

A function passed to `map_batches` takes one `pyarrow.RecordBatch` and returns one
`pyarrow.RecordBatch`. Because it sees the whole batch, it can use vectorized
Arrow compute rather than per-row Python loops.

```python
import batcher as bt
import pyarrow.compute as pc

ds = bt.from_pydict({"x": [1, 2, 3, 4], "y": [10, 20, 30, 40]})


def add_sum(batch):
    total = pc.add(batch.column("x"), batch.column("y"))
    return batch.append_column("sum", total)


print(ds.ml.map_batches(add_sum).to_pydict())
# {'x': [1, 2, 3, 4], 'y': [10, 20, 30, 40], 'sum': [11, 22, 33, 44]}
```

## Class-based functions load once per worker

A plain function is re-imported on each worker. A class is instantiated once per
worker and then called per batch, so any expensive setup (loading a model,
opening a tokenizer) happens once and is reused across all batches that worker
processes. The class implements `__call__(self, batch) -> batch`.

```python
import pyarrow as pa


class Scale:
    def __init__(self, factor):
        self.factor = pa.scalar(factor)

    def __call__(self, batch):
        scaled = pc.multiply(batch.column("x"), self.factor)
        return batch.set_column(0, "x", scaled)


print(ds.ml.map_batches(Scale(10)).to_pydict())
# {'x': [10, 20, 30, 40], 'y': [10, 20, 30, 40]}
```

For a real model, the constructor loads the weights and `__call__` runs the
forward pass. That needs a GPU and a model, so it's shown but not run here.

```python
# docs: skip
class Classifier:
    def __init__(self):
        import torch

        self.model = torch.load("model.pt").cuda().eval()

    def __call__(self, batch):
        import torch

        x = torch.tensor(batch.column("features").to_pylist()).cuda()
        with torch.no_grad():
            preds = self.model(x).argmax(dim=1).cpu().tolist()
        return batch.append_column("prediction", pa.array(preds))


labelled = ds.ml.map_batches(Classifier(), num_gpus=1, concurrency=4)
```

## Common arguments

`map_batches`, `infer`, and `embed` share these keywords:

| Argument | Meaning |
| --- | --- |
| `batch_size` | Rows per batch handed to `fn`. Defaults to the engine morsel size. |
| `output_columns` | Names of the columns the function produces, when they differ from the input. |
| `batch_format` | What `fn` receives/returns: `"pyarrow"` (default), `"numpy"`, `"pandas"`, or `"torch"`. |
| `num_gpus` | GPUs to reserve per worker (a fraction packs several workers onto one GPU). |
| `concurrency` | Actor-pool size: an `int` for a fixed pool, or a `(min, max)` tuple to autoscale to the workload. |
| `accelerator_type` | Pin GPU actors to a device model (a `ray.util.accelerators` name such as `"NVIDIA_A100"`). |
| `model_memory_gb` | The model's GB footprint. Budgets host memory per worker to protect against OOM, and VRAM-packs small models onto a shared GPU. |
| `num_workers` | Number of workers (`map_batches`). |

`num_gpus` and `concurrency` together describe a GPU actor pool: each actor holds
`num_gpus` of a device, and `concurrency` actors run in parallel. `batch_format` converts only around the call, and the engine boundary stays Arrow. See [GPU scheduling](../ml/gpu.md).

## infer and embed

`ds.ml.infer(model, ...)` and `ds.ml.embed(model, ...)` are the inference-shaped calls. The quickest form is a **model identifier** plus the `column` to run on. The model loads once per worker and the result is appended, as a prediction for `infer` and a vector for `embed`. `infer` resolves a HuggingFace `transformers` pipeline, and `embed` resolves a `sentence-transformers` model.

```python
# docs: skip
scored = ds.ml.infer("distilbert-base-uncased-finetuned-sst-2-english", column="text")
vectors = ds.ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="text")
```

For full control over a custom model, a non-text modality, or your own batching, pass a callable or a class that loads weights once per worker, and declare the result schema with `output_columns`. Both forms take `batch_size`, `num_gpus`, and `concurrency`. Real models need GPUs, so these aren't run here.

```python
# docs: skip
scored = ds.ml.infer(Classifier(), output_columns=[...], batch_size=512, num_gpus=1, concurrency=4)
vectors = ds.ml.embed(Embedder(), output_columns=[...], batch_size=256, num_gpus=1, concurrency=2)
```

See [Inference](../ml/inference.md) for the inference workflow and
[Streaming](../ml/streaming.md) for feeding training loops.

## What lives outside the accessor

Operators that aren't `Dataset` methods live in `batcher.ml`: the standalone `embed` and `llm_generate` functions, the [preprocessors](../ml/preprocessors.md), the [serving adapters](../ml/serving.md), [vector search](../ml/multimodal.md), the `Chain` preprocessor pipeline, the `ResumableSampler` checkpointable per-rank index stream, and the [LLM engines](../ml/llm.md).

A *callable* model passed to `map_batches` or `infer` receives the whole batch and picks its own columns, so there's no `input_columns=` keyword. The model-identifier form of `infer` and `embed` takes the `column` to run on instead.

## Preprocessors

`batcher.ml.preprocessors` holds the fit/transform feature-engineering estimators.
Each one `fit`s over a `Dataset` to learn its statistics, then `transform`s any `Dataset` with them. `Chain` composes several into one pipeline. See the
[preprocessors guide](../ml/preprocessors.md) for how they fit into a training
workflow.

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autoclass:: Preprocessor
   :members:

.. autoclass:: Chain
   :members:
```

### Scalers and normalizers

These rescale numeric columns:

```{eval-rst}
.. autoclass:: StandardScaler
   :members:

.. autoclass:: MinMaxScaler
   :members:

.. autoclass:: MaxAbsScaler
   :members:

.. autoclass:: RobustScaler
   :members:

.. autoclass:: Normalizer
   :members:
```

### Distribution shaping

These reshape a column's *distribution* rather than only its scale. Reach for them when a
feature is heavily skewed or long-tailed and a linear rescale would leave it that way:

```{eval-rst}
.. autoclass:: PowerTransformer
   :members:

.. autoclass:: BoxCoxTransformer
   :members:

.. autoclass:: QuantileTransformer
   :members:

.. autoclass:: RankTransformer
   :members:

.. autoclass:: LogTransformer
   :members:

.. autoclass:: PCA
   :members:

.. autoclass:: TruncatedSVD
   :members:
```

### Encoders

These turn categorical columns into numeric ones:

```{eval-rst}
.. autoclass:: OneHotEncoder
   :members:

.. autoclass:: MultiHotEncoder
   :members:

.. autoclass:: LabelBinarizer
   :members:

.. autoclass:: MultiLabelBinarizer
   :members:

.. autoclass:: LabelEncoder
   :members:

.. autoclass:: OrdinalEncoder
   :members:

.. autoclass:: BinaryEncoder
   :members:

.. autoclass:: TargetEncoder
   :members:

.. autoclass:: FrequencyEncoder
   :members:

.. autoclass:: HashingEncoder
   :members:

.. autoclass:: RareCategoryEncoder
   :members:

.. autoclass:: WOEEncoder
   :members:
```

### Binning, imputation, text, and assembly

The rest of the estimators cover discretization, missing values, text splitting, and feature assembly:

```{eval-rst}
.. autoclass:: KBinsDiscretizer
   :members:

.. autoclass:: SimpleImputer
   :members:

.. autoclass:: Tokenizer
   :members:

.. autoclass:: Concatenator
   :members:

.. autoclass:: PolynomialFeatures
   :members:

.. autoclass:: Clipper
   :members:

.. autoclass:: MissingIndicator
   :members:

.. autoclass:: Binarizer
   :members:

.. autoclass:: VarianceThreshold
   :members:

.. autoclass:: ColumnSelector
   :members:

.. autoclass:: ColumnDropper
   :members:
```

### Derived and grouped features

These build new columns out of existing ones: products and ratios that a linear model
cannot learn on its own, and group-relative statistics that let a row see its cohort:

```{eval-rst}
.. autoclass:: InteractionFeatures
   :members:

.. autoclass:: RatioFeatures
   :members:

.. autoclass:: GroupStatEncoder
   :members:

.. autoclass:: GroupImputer
   :members:
```

### Timestamp features

A raw timestamp is the least useful column in a feature table. These turn it into parts a
model can learn from — integer parts for a tree, circular coordinates for anything that
measures distance:

```{eval-rst}
.. autoclass:: DateTimeFeaturizer
   :members:

.. autoclass:: CyclicalEncoder
   :members:
```

### Lag and rolling features

History as columns, for a forecasting model. Both exclude the current row by construction,
because a rolling window that includes it puts the target's own value inside its own
feature — the most common leak in a forecasting pipeline, and one that raises nothing:

```{eval-rst}
.. autoclass:: LagFeaturizer
   :members:

.. autoclass:: RollingFeaturizer
   :members:
```

### Text surface features

Cheap, interpretable text signals — length, word count, character mix — that need no model
and often carry most of the signal a gradient-boosted model splits on:

```{eval-rst}
.. autoclass:: TextStatFeaturizer
   :members:
```

## `batcher.ml` reference

The `.ml` accessor above covers the common path. Underneath it, `batcher.ml` exports the same machinery as plain functions over Arrow batch iterators. Reach for those when you're driving the pipeline yourself, such as from a custom training loop or a serving process, rather than executing a `Dataset`.

```python
import batcher.ml as ml
```

### LLM inference

An *engine* is any callable from a list of prompts to a list of completions. That is the
whole contract, which is why a local vLLM engine, a remote OpenAI-compatible endpoint, and
a hosted Claude model are interchangeable: swap {py:obj}`vllm_engine <batcher.ml.vllm_engine>`
for {py:obj}`http_engine <batcher.ml.http_engine>` or
{py:obj}`anthropic_engine <batcher.ml.anthropic_engine>` and nothing else changes.

```{eval-rst}
.. currentmodule:: batcher.ml

.. autosummary::
   :toctree: generated
   :nosignatures:

   vllm_engine
   http_engine
   anthropic_engine
   llm_generate
   llm_udf
   json_schema

.. autodata:: Engine

.. autodata:: EngineFactory
```

### Model serving

Call a model that lives in another process or on another host. Each client turns a
served endpoint into a UDF you can drop into a pipeline.

```{eval-rst}
.. autoclass:: ServingClient
   :members:

.. autosummary::
   :toctree: generated
   :nosignatures:

   serving_udf
   serve_deployment
   triton_client
   torchserve_client
   http_client
```

### Inference pools and pipelines

{py:obj}`InferencePool <batcher.ml.InferencePool>` keeps model-loading off the hot path:
workers load once and are reused across batches. {py:obj}`run_pipeline <batcher.ml.run_pipeline>`
chains {py:obj}`Stage <batcher.ml.Stage>`s with credit-based backpressure, which is what
overlaps a CPU decode with the GPU forward of the previous batch instead of running them
in lockstep.

When the embedding model runs behind a service rather than in the worker,
{py:obj}`openai_embedding_encoder <batcher.ml.openai_embedding_encoder>` (any
OpenAI-compatible `/embeddings` endpoint) and {py:obj}`tei_encoder <batcher.ml.tei_encoder>`
(a HuggingFace Text-Embeddings-Inference server) are load-once encoders that drop into
`ds.ml.embed(...)` in place of a local model id.

```{eval-rst}
.. autoclass:: InferencePool
   :members:

.. autoclass:: Stage
   :members:

.. autosummary::
   :toctree: generated
   :nosignatures:

   run_pipeline
   embed
   openai_embedding_encoder
   tei_encoder

.. autodata:: Worker

.. autodata:: WorkerFactory
```

### Training loaders

Stream a dataset into a training loop as tensors, without materializing it.
{py:obj}`streaming_split <batcher.ml.streaming_split>` gives each DDP rank a disjoint
shard.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   iter_torch_batches
   to_torch_iterable
   to_tf_dataset
   to_numpy_batches
   stream_loader
   shard_stream_loader
   streaming_split
```

### Feature contract

A trained model is only valid against the exact columns, order, and dtypes it saw during
training. `FeatureSpec` pins that contract so scoring can be checked against it rather than
failing silently on a reordered or retyped frame.

```{eval-rst}
.. autoclass:: FeatureSpec
   :members:
```

### Sampling and resumption

These give deterministic, resumable epoch ordering. A training run that dies at step 40,000 restarts at step 40,000 seeing the same samples in the same order, rather than silently re-showing data it already trained on.

```{eval-rst}
.. autoclass:: ResumableSampler
   :members:
   :special-members: __len__, __iter__

.. autosummary::
   :toctree: generated
   :nosignatures:

   epoch_order
   epoch_permutation
   rank_index_batches
   usable_length
   pack_sequences
```

### Vector search

Build an index over an embedding column and query it:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   build_vector_index
   vector_search

.. autodata:: EncoderFactory
```

## Persistence

A fitted preprocessor's state has to outlive the process that fitted it, or a serving
request is standardized with its own mean instead of the training set's. These read and
write that state as plain JSON — reviewable, diffable, portable, and safe to load from a
store you do not fully control, which a pickle is none of.

```{eval-rst}
.. currentmodule:: batcher.ml.preprocessors

.. autofunction:: save
.. autofunction:: load
.. autofunction:: to_dict
.. autofunction:: from_dict
```

## Tabular models

`batcher.ml.tabular` is the classical-ML inference plane behind
{meth}`~batcher.Dataset.ml.predict`: it assembles Arrow columns into the dense matrix an
XGBoost, LightGBM, CatBoost, scikit-learn, or ONNX model expects, and wraps the model as a
load-once class UDF. See the [tabular models guide](../ml/tabular-models.md).

```{eval-rst}
.. currentmodule:: batcher.ml.tabular

.. autofunction:: tabular_predictor
.. autofunction:: predicted_column_names
.. autofunction:: feature_matrix
.. autofunction:: prediction_columns
.. autofunction:: resolve_features
.. autofunction:: detect_framework
.. autofunction:: get_adapter
```

## Linear models

`batcher.ml.linear` fits ordinary and ridge regression inside the engine — the normal equations
are built from the feature/target moments, so the whole fit is a single scan and only the small
solve runs on the driver. Both reproduce scikit-learn's coefficients exactly.

```{eval-rst}
.. currentmodule:: batcher.ml.linear

.. autoclass:: LinearRegression
   :members:

.. autoclass:: Ridge
   :members:

.. autoclass:: RidgeClassifier
   :members:

.. autoclass:: LogisticRegression
   :members:
```

`batcher.ml.dummy` holds the baseline predictors a real model must beat.

```{eval-rst}
.. currentmodule:: batcher.ml.dummy

.. autoclass:: DummyRegressor
   :members:

.. autoclass:: DummyClassifier
   :members:
```

`batcher.ml.sparse_linear` adds the L1-regularized linear models that select features by zeroing coefficients.

```{eval-rst}
.. currentmodule:: batcher.ml.sparse_linear

.. autoclass:: Lasso
   :members:

.. autoclass:: ElasticNet
   :members:
```

`batcher.ml.glm` fits the Tweedie family of generalized linear models — the general form and its Poisson and gamma special cases — by the same one-scan IRLS steps.

```{eval-rst}
.. currentmodule:: batcher.ml.glm

.. autoclass:: TweedieRegressor
   :members:

.. autoclass:: PoissonRegressor
   :members:

.. autoclass:: GammaRegressor
   :members:
```

`batcher.ml.naive_bayes` adds the probabilistic baseline whose entire fit is one grouped
aggregate.

```{eval-rst}
.. currentmodule:: batcher.ml.naive_bayes

.. autoclass:: GaussianNB
   :members:

.. autoclass:: MultinomialNB
   :members:

.. autoclass:: BernoulliNB
   :members:
```

`batcher.ml.discriminant` adds the Gaussian classifiers: `LinearDiscriminantAnalysis` shares one
covariance across classes (linear boundaries), and `QuadraticDiscriminantAnalysis` gives each class
its own (quadratic boundaries).

```{eval-rst}
.. currentmodule:: batcher.ml.discriminant

.. autoclass:: LinearDiscriminantAnalysis
   :members:

.. autoclass:: QuadraticDiscriminantAnalysis
   :members:
```

## Evaluation

`batcher.ml.metrics` holds the metrics that need a global ordering or return a table. The
single-pass metric *expressions* — `bt.rmse`, `bt.f1_score`, and the rest — are in the
[expression reference](expressions.md) instead, because they are ordinary aggregates. See
the [evaluation guide](../ml/evaluation.md).

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autofunction:: evaluate
.. autofunction:: roc_auc
.. autofunction:: average_precision
.. autofunction:: ks_statistic
.. autofunction:: gini_coefficient
.. autofunction:: confusion_matrix
.. autofunction:: threshold_sweep
.. autofunction:: lift_table
.. autofunction:: calibration_curve
.. autofunction:: classification_report
.. autofunction:: multiclass_averages
.. autofunction:: residual_summary
.. autofunction:: prediction_interval_coverage
.. autofunction:: top_k_accuracy
.. autofunction:: expected_calibration_error
.. autofunction:: maximum_calibration_error
.. autofunction:: brier_skill_score

.. autodata:: METRIC_SETS
```

A recommender is scored differently: what matters is the order within *one* query, averaged
over queries. These compute the metric per group and then average, never pooling rows across
groups — which silently rewards a model that ranks one heavy user well and everyone else
badly.

```{eval-rst}
.. autofunction:: precision_at_k
.. autofunction:: recall_at_k
.. autofunction:: hit_rate_at_k
.. autofunction:: mean_reciprocal_rank
.. autofunction:: map_at_k
.. autofunction:: ndcg_at_k
```

Picking a cutoff is the step between a good AUC and a deployed model, and 0.5 is almost never
the right one:

```{eval-rst}
.. autofunction:: best_threshold
.. autofunction:: best_cost_threshold
.. autofunction:: expected_cost_curve
.. autofunction:: compare_models
```

## Outlier detection

`batcher.ml.outliers` finds the rows that do not belong to the same process as the rest, by
the three standard rules (IQR, z-score, MAD), each a per-column bound learned in one aggregate.

```{eval-rst}
.. currentmodule:: batcher.ml.outliers

.. autofunction:: outlier_bounds
.. autofunction:: flag_outliers
.. autofunction:: count_outliers
.. autofunction:: mahalanobis_distance
.. autoclass:: OutlierClipper
   :members:
```

## Clustering

`batcher.ml.cluster` holds the unsupervised clusterers. K-means maps each Lloyd iteration onto
one assignment expression and one grouped mean, so the fit is a handful of scans and labeling is
a single streaming pass.

```{eval-rst}
.. currentmodule:: batcher.ml.cluster

.. autoclass:: KMeans
   :members:

.. autoclass:: NearestCentroid
   :members:
```

`batcher.ml.mixture` fits a Gaussian mixture — soft clustering and density estimation by EM.

```{eval-rst}
.. currentmodule:: batcher.ml.mixture

.. autoclass:: GaussianMixture
   :members:
```

Score a clustering against a reference labeling with `batcher.ml.metrics`, each computed from one
`group_by` contingency table.

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autofunction:: adjusted_rand_score
.. autofunction:: rand_score
.. autofunction:: normalized_mutual_info_score
.. autofunction:: adjusted_mutual_info_score
.. autofunction:: mutual_info_score
.. autofunction:: homogeneity_score
.. autofunction:: completeness_score
.. autofunction:: v_measure_score
.. autofunction:: fowlkes_mallows_score
.. autofunction:: contingency_matrix
.. autofunction:: pair_confusion_matrix
.. autofunction:: calinski_harabasz_score
.. autofunction:: davies_bouldin_score
```

## Feature selection

`batcher.ml.selection` decides which columns are worth keeping without fitting a model,
which is both cheaper and less circular than reading a model's own importances.

```{eval-rst}
.. currentmodule:: batcher.ml.selection

.. autofunction:: feature_report
.. autofunction:: feature_profile
.. autofunction:: constant_columns
.. autofunction:: correlated_columns
```

`batcher.ml.feature_scores` ranks each feature against the target with a univariate score,
the filter half of scikit-learn's `SelectKBest`.

```{eval-rst}
.. currentmodule:: batcher.ml.feature_scores

.. autofunction:: f_classif_scores
.. autofunction:: f_regression_scores
.. autofunction:: chi2_scores
.. autofunction:: mutual_info_scores
.. autofunction:: select_k_best
```

`batcher.ml.timeseries` diagnoses serial structure — the autocorrelation function and the standard tests of whether a series or a model's residuals still carry it.

```{eval-rst}
.. currentmodule:: batcher.ml.timeseries

.. autofunction:: autocorrelation
.. autofunction:: autocorrelations
.. autofunction:: partial_autocorrelation
.. autofunction:: partial_autocorrelations
.. autofunction:: ljung_box
.. autofunction:: durbin_watson
.. autofunction:: mean_absolute_scaled_error
```

## Statistics and drift

`batcher.ml.stats` covers the statistics that need more than one aggregate, and the
reference-versus-current comparisons a deployed model needs. See the
[statistics and drift guide](../ml/statistics-and-drift.md).

```{eval-rst}
.. currentmodule:: batcher.ml.stats

.. autofunction:: spearman_corr
.. autofunction:: entropy
.. autofunction:: gini_impurity
.. autofunction:: herfindahl_index
.. autofunction:: mode_share
.. autofunction:: chi_square
.. autofunction:: cramers_v
.. autofunction:: mutual_information
.. autofunction:: anova_f
.. autofunction:: theils_u
.. autofunction:: eta_squared
.. autofunction:: epsilon_squared
.. autofunction:: omega_squared
.. autofunction:: cohens_f
.. autofunction:: correlation_matrix
.. autofunction:: covariance_matrix
.. autofunction:: partial_correlation
.. autofunction:: variance_inflation_factor
.. autofunction:: trimmed_mean
.. autofunction:: winsorized_mean
.. autofunction:: median_abs_deviation
.. autofunction:: outlier_mask
.. autofunction:: population_stability_index
.. autofunction:: kl_divergence
.. autofunction:: js_divergence
.. autofunction:: categorical_drift
.. autofunction:: woe_table
.. autofunction:: information_value
.. autofunction:: drift_report
.. autoclass:: TestResult
.. autofunction:: t_test_1samp
.. autofunction:: t_test_ind
.. autofunction:: anova_test
.. autofunction:: chi_square_test
.. autofunction:: normality_test
.. autofunction:: pearson_test
.. autofunction:: spearman_test
.. autofunction:: proportion_ztest
.. autofunction:: binomial_test
.. autofunction:: mcnemar_test
.. autofunction:: bartlett_test
.. autofunction:: levene_test
.. autofunction:: mann_whitney_u
.. autofunction:: wilcoxon_signed_rank
.. autofunction:: kruskal_wallis
.. autofunction:: friedman_test
.. autofunction:: cliffs_delta
.. autofunction:: common_language_effect_size
```

## Fairness

`batcher.ml.metrics` includes the fairness metrics — grouped comparisons that reveal a model
treating a protected group differently, each a single grouped aggregate.

```{eval-rst}
.. currentmodule:: batcher.ml.metrics

.. autofunction:: demographic_parity_difference
.. autofunction:: disparate_impact_ratio
.. autofunction:: equal_opportunity_difference
.. autofunction:: equalized_odds_difference
.. autofunction:: predictive_parity_difference
.. autofunction:: group_fairness_report
.. autofunction:: d2_tweedie_score
.. autofunction:: d2_absolute_error_score
.. autofunction:: d2_pinball_score
```

## Resampling for imbalanced learning

`batcher.ml.sampling` reshapes the class balance as a relational operation — an exact
content-hashed filter or concatenation, never a driver-side shuffle.

```{eval-rst}
.. currentmodule:: batcher.ml.sampling

.. autofunction:: class_counts
.. autofunction:: class_weights
.. autofunction:: sample_weights
.. autofunction:: undersample
.. autofunction:: oversample
.. autofunction:: balanced_sample
.. autofunction:: stratified_sample
```

## Cross-validated scoring

`batcher.ml.model_selection` ties the fold splitter, a fitted model, and a metric into one
loop — each fold's data runs through the engine rather than a driver-held array.

```{eval-rst}
.. currentmodule:: batcher.ml.model_selection

.. autofunction:: cross_val_score
.. autofunction:: cross_val_predict
.. autofunction:: learning_curve
.. autofunction:: validation_curve
```

## Cross-validation splits

`batcher.ml.splitting` builds folds as content-hash filters rather than a materialized
shuffle, so a fold is an ordinary row-wise predicate and the assignment is identical
however the data is partitioned.

```{eval-rst}
.. currentmodule:: batcher.ml.splitting

.. autofunction:: kfold
.. autofunction:: stratified_kfold
.. autofunction:: group_kfold
.. autofunction:: time_series_split
.. autofunction:: fold_column
```

## Model interpretation

`batcher.ml.interpret` explains a model over the whole dataset rather than a driver-sized
sample, because both techniques re-score through the engine.

```{eval-rst}
.. currentmodule:: batcher.ml.interpret

.. autofunction:: permutation_importance
.. autofunction:: partial_dependence
```

## Next steps

- [Inference](../ml/inference.md): batch prediction and embeddings.
- [Preprocessors](../ml/preprocessors.md): fit/transform feature engineering.
- [Multimodal](../ml/multimodal.md): download, decode, tensors, vector search.
- [Serving](../ml/serving.md) and [LLM inference](../ml/llm.md).
- [PyTorch](../ml/pytorch.md) and [streaming](../ml/streaming.md) training loaders.
- [GPU scheduling](../ml/gpu.md): how `num_gpus` and `concurrency` map to actors.
- [Tabular models](../ml/tabular-models.md): scoring XGBoost, LightGBM, and scikit-learn.
- [Evaluation](../ml/evaluation.md): metrics, per-segment scoring, diagnostic tables.
- [Statistics and drift](../ml/statistics-and-drift.md): feature screening and monitoring.
