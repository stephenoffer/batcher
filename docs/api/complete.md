# Complete reference

This page lists every public name in `batcher`, rendered from the source docstrings. It's the exhaustive backstop behind the [quick reference](reference.md) and the example-first [area pages](index.md). Each top-level function links to its own page.

## Construction and I/O

Build a `Dataset` from in-memory data, another framework, or a storage source, and
register SQL functions or sessions.

```{eval-rst}
.. currentmodule:: batcher

.. autosummary::
   :toctree: generated
   :nosignatures:

   from_pydict
   from_dict
   from_pylist
   from_dicts
   from_records
   from_items
   from_iter
   from_arrow
   from_batches
   from_numpy
   from_pandas
   from_polars
   from_duckdb
   from_spark
   from_dask
   from_huggingface
   from_torch
   from_tf
   from_ray_dataset
   from_any
   read
   read_table
   read_csv
   read_parquet
   read_json
   read_ndjson
   read_ipc
   read_orc
   read_avro
   read_excel
   read_delta
   read_iceberg
   read_database
   read_memory
   sql
   streams
   await_any_termination
   register_function
   udf
   compact
   vacuum
   engine_version
   versions
   show_versions
   start_ui
   stop_ui
   ui_url
```

## Expressions and columns

Reference and derive columns, build literals, and branch.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   col
   lit
   when
   coalesce
   nullif
   iff
   element
   struct
   named_struct
   array
```

## Column selectors

A *selector* stands for every column matching a predicate. Pass one anywhere a column is expected, such as `ds.select(bt.numeric())` or `ds.with_columns(bt.floating().round(2))`, and it expands against the input schema. See the [transformations guide](../user-guide/transformations.md) for how they compose.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   all
   numeric
   integer
   floating
   string
   boolean
   temporal
   by_dtype
   matches
   starts_with
   ends_with
   contains
   exclude
```

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.selectors.Selector
   :members:

.. autoclass:: batcher.plan.expr_ir.selectors.core._SelectorNameNamespace
   :members:
   :member-order: bysource
```

## Scalar functions

Row-wise math, string, and date/time helpers usable anywhere an expression is.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   greatest
   least
   atan2
   arctan2
   hypot
   gcd
   lcm
   log
   nanvl
   width_bucket
   concat
   concat_str
   concat_ws
   format_string
   mask
   hmac_sha256
   aes_encrypt
   aes_decrypt
   current_date
   current_timestamp
   date_add
   date_sub
   date_part
   range
   date_range
   sequence
   hash_rows
```

## Text extraction and LLM output parsing

Pull structured fragments out of a model's generated text as a vectorized expression: a JSON
blob, a fenced code block, an XML-style tag, a reasoning trace, or a multiple-choice letter.
Each returns an empty string (or null, for the numeric one) where the fragment is absent, so a
malformed row degrades instead of raising.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   extract_json
   extract_json_array
   extract_code_block
   extract_first_number
   extract_tag
   extract_reasoning
   strip_reasoning
   extract_after
   extract_between
   is_refusal
   extract_choice
```

## Prompt construction

Assemble an LLM prompt from row columns in the data plane: interpolate columns into a template,
wrap a field in tags, or trim a column to fit a context budget.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   render_template
   wrap_tag
   truncate_to_token_budget
```

## Horizontal functions

These reduce *across* the listed expressions within one row, rather than down a column. The vertical counterparts are the aggregates below.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   min_horizontal
   max_horizontal
   sum_horizontal
   mean_horizontal
   count_horizontal
   product_horizontal
   all_horizontal
   any_horizontal
   reduce_horizontal
   fold_horizontal
```

## Aggregate and window functions

Use these in `group_by(...).agg(...)` or `.over(...)` window frames. The ranking and
value functions are window-only: bind them with `.over(partition_by=…, order_by=…)`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   count
   sum
   mean
   min
   max
   median
   std
   var
   n_unique
   product
   mode
   skewness
   kurtosis
   bool_and
   bool_or
   bit_and
   bit_or
   bit_xor
   array_agg
   quantile
   approx_quantile
   approx_median
   approx_n_unique
   histogram
   count_if
   corr
   covar_pop
   covar_samp
   regr_slope
   regr_intercept
   regr_r2
   regr_count
   regr_avgx
   regr_avgy
   regr_sxx
   regr_syy
   regr_sxy
   var_pop
   stddev_pop
   geometric_mean
   harmonic_mean
   rms
   cv
   sem
   midrange
   weighted_mean
   weighted_var
   weighted_std
   weighted_covariance
   weighted_correlation
   q1
   q3
   iqr
   value_range
   null_rate
   non_null_rate
   nunique_ratio
   first
   last
   arg_min
   arg_max
   row_number
   rank
   dense_rank
   percent_rank
   cume_dist
   ntile
   lag
   lead
   first_value
   last_value
   nth_value
```

## Model metrics

Scoring aggregates over a column of labels and a column of predictions. They are ordinary
aggregates, so they compose with `group_by(...).agg(...)` to score per segment in one pass.
See the [model evaluation guide](../ml/evaluation.md).

The classification metrics take a label column and a predicted-label column, plus
`positive=` to say which label value counts as positive:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   accuracy
   balanced_accuracy
   precision
   recall
   specificity
   f1_score
   fbeta_score
   cohen_kappa
   matthews_corrcoef
   negative_predictive_value
   false_negative_rate
   false_positive_rate
   prevalence
   true_positives
   true_negatives
   false_positives
   false_negatives
   hamming_loss
   jaccard_score
   false_discovery_rate
   false_omission_rate
   positive_likelihood_ratio
   negative_likelihood_ratio
   diagnostic_odds_ratio
   informedness
   markedness
   fowlkes_mallows_index
   geometric_mean_score
   prevalence_threshold
```

The regression metrics take a truth column and a prediction column:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   mae
   mse
   rmse
   normalized_rmse
   medae
   msle
   rmsle
   mape
   smape
   wape
   r2
   explained_variance
   max_error
   mean_bias
   mean_percentage_error
   huber_loss
   pinball_loss
   poisson_deviance
   gamma_deviance
   tweedie_deviance
   concordance_correlation
   nash_sutcliffe_efficiency
   kling_gupta_efficiency
```

The probabilistic metrics score a predicted *probability* rather than a hard label:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   brier_score
   log_loss
   hinge_loss
   squared_hinge_loss
```

The generation metrics score a model's generated text against a reference string. The token-set
metrics split on whitespace; the character n-gram metrics (chrF-style) need no word boundaries, so
they score languages without spaces such as Chinese or Japanese:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   exact_match
   normalized_exact_match
   token_set_precision
   token_set_recall
   token_set_f1
   token_set_jaccard
   length_ratio
   char_ngram_precision
   char_ngram_recall
   char_ngram_f1
   char_ngram_jaccard
```

The embedding metrics score fixed-width vector columns for retrieval quality and drift:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   mean_cosine_similarity
   mean_euclidean_distance
   mean_dot_product
   mean_embedding_norm
   unit_norm_rate
   zero_vector_rate
   mean_cosine_distance
   mean_manhattan_distance
   mean_angular_distance
   mean_hamming_distance
```

The generation-quality metrics score an output column on its own, with no reference: diversity,
verbosity, and the empty, refusal, and truncation rates a team watches on a dashboard.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   distinct_token_ratio
   mean_output_tokens
   empty_generation_rate
   refusal_rate
   truncation_rate
```

The text-quality monitors watch a generated column for what it should not do at scale: shout,
spray punctuation, drift into non-ASCII, emit a URL, leak a code block, or run too long or short.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   all_caps_rate
   repeated_punctuation_rate
   non_ascii_rate
   url_rate
   code_block_rate
   long_output_rate
   short_output_rate
   mean_sentence_count
   mean_word_length
```

The token and cost aggregates size an LLM run before it is paid for: the total token bill, the
fraction of rows that overflow a context window, and the token-length tail that sizes the window.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   total_token_estimate
   token_budget_exceed_rate
   token_estimate_quantile
```

The structured-output compliance metrics measure whether a model returned the shape it was asked
for: valid JSON, an extractable JSON object, or an answer inside a named tag.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   valid_json_rate
   json_present_rate
   tagged_answer_rate
   numeric_answer_rate
   choice_answer_rate
   boxed_answer_rate
```

The RAG grounding metrics compare a generated answer column against its retrieved context column,
measuring how much of the answer the context supports and how much is unsupported.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   answer_groundedness
   context_utilization
   unsupported_token_rate
   fully_grounded_rate
   citation_rate
```

The readability metrics score how complex a generated column reads, for matching a target reading
level.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   automated_readability_index
   mean_words_per_sentence
   mean_chars_per_word
   long_word_rate
   mean_paragraph_count
```

The repetition metrics detect degeneration, where a model loops or repeats n-grams instead of
producing new content.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   distinct_char_ngram_ratio
   char_repetition_rate
   repeated_line_rate
   compression_ratio_proxy
```

The PII and safety monitors flag a generated column that leaks a contact detail, a structured
identifier, or a blocklisted term.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   email_rate
   phone_rate
   pii_rate
   ssn_like_rate
   credit_card_like_rate
   contains_any_rate
```

The formatting metrics check whether generated text used the Markdown elements a task asked for.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   heading_rate
   bullet_list_rate
   numbered_list_rate
   markdown_link_rate
   table_rate
   code_block_present_rate
```

The tone metrics track the register of generated text, catching a model that hedges, over-excites,
or deflects a question.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   question_rate
   exclamation_rate
   hedge_rate
   first_person_rate
   politeness_rate
   contains_phrase_rate
```

The script metrics measure the character-set composition of generated text, for catching language
drift or emoji spam.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   cjk_rate
   cyrillic_rate
   arabic_rate
   emoji_rate
   latin_only_rate
```

## Statistical analysis

Descriptive and inferential statistics as aggregates. See the
[model evaluation guide](../ml/evaluation.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   trimean
   midhinge
   interdecile_range
   quartile_dispersion
   decile_ratio
   robust_cv
   bowley_skew
   moors_kurtosis
   pearson_mode_skew
   jarque_bera
   correlation_ratio
   point_biserial
   signal_ratio
   group_mean
   cohens_d
   hedges_g
   welch_t_statistic
   welch_df
   mean_ci_half_width
   proportion_ci_half_width
   proportion_z_statistic
   index_of_dispersion
   signal_to_noise
   studentized_range
   relative_range
   geometric_std
   cut
```

## Configuration functions

Read and override the engine tunables. See the [configuration guide](configuration.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   set_config
   config_context
```

```{eval-rst}
.. autofunction:: batcher.config.active_config
```

### Options by name

Address any tunable by its dotted path, in the style of `pandas.set_option` and
`spark.conf.set`. See the [configuration guide](../configuration/index.md).

```{eval-rst}
.. autofunction:: batcher.config.get_option
.. autofunction:: batcher.config.set_option
.. autofunction:: batcher.config.reset_option
.. autofunction:: batcher.config.option_context
.. autofunction:: batcher.config.option_names
.. autofunction:: batcher.config.describe_options
```

### Serialization

```{eval-rst}
.. autofunction:: batcher.config.config_to_dict
.. autofunction:: batcher.config.env_var_names
```

### Logging and verbosity

One-line switches over `ObservabilityConfig`. See
[observability](../user-guide/observability.md).

```{eval-rst}
.. autofunction:: batcher.config.set_log_level
.. autofunction:: batcher.config.enable_logging
.. autofunction:: batcher.config.disable_logging
.. autofunction:: batcher.config.set_verbosity
.. autofunction:: batcher.config.set_progress
.. autofunction:: batcher.config.get_logger
```

### Metrics export

Process-wide counters as plain data, for Prometheus, OpenTelemetry, or a log line.

```{eval-rst}
.. autofunction:: batcher.observe.metrics_snapshot
.. autofunction:: batcher.observe.prometheus_text
.. autofunction:: batcher.observe.start_metrics
.. autofunction:: batcher.observe.reset_metrics
```

## Dataset

```{eval-rst}
.. autoclass:: batcher.Dataset
   :members:
   :member-order: groupwise
   :special-members: __getitem__, __len__, __iter__, __contains__, __add__, __or__, __and__, __sub__, __arrow_c_stream__
```

## GroupBy

```{eval-rst}
.. autoclass:: batcher.GroupBy
   :members:
   :member-order: groupwise
```

## Governance

See the [governance guide](../user-guide/governance.md) for how these fit together.

```{eval-rst}
.. autoclass:: batcher.SecurityCatalog
   :members:

.. autoclass:: batcher.Principal
   :members:

.. autoclass:: batcher.GovernanceEvent
   :members:

.. autofunction:: batcher.security
```

Column-level lineage is reached from the dataset itself: :meth:`batcher.Dataset.lineage`.

## Expressions

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.core.Expr
   :members:
   :member-order: groupwise

.. autoclass:: batcher.AggExpr
   :members:
   :member-order: groupwise
```

### Expression accessors

These are the typed namespaces reached as `col("x").str`, `.dt`, `.list`, `.struct`, `.json`, and `.map`. Multimodal columns add `.image`, `.audio`, and `.video`.

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.namespaces.strings._StrNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.temporal._DtNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.collections._ListNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.collections._StructNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.collections._JsonNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.namespaces.collections._MapNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.image._ImageNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.audio._AudioNamespace
   :members:
   :member-order: bysource

.. autoclass:: batcher.plan.expr_ir.video._VideoNamespace
   :members:
   :member-order: bysource
```

## Reading and writing

`bt.read` is the reader namespace; `ds.write` is the writer namespace.

```{eval-rst}
.. autoclass:: batcher.api.io_namespace.reader.Reader
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.io_namespace.writer.Writer
   :members:
   :member-order: bysource
```

## Dataset accessors

The `ds.ml`, `ds.dq`, and `ds.scd` namespaces for machine learning, data quality,
and slowly-changing-dimension workflows.

```{eval-rst}
.. autoclass:: batcher.api.dataset.ml.DatasetML
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.dq.DatasetDQ
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.dq.ValidationReport
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.scd.DatasetSCD
   :members:
   :member-order: bysource
```

## Metadata shortcuts

The `ds.meta` namespace and the accessors it hands out read answers from footers, manifests, and catalogs instead of from the data. See the [metadata shortcuts guide](../user-guide/metadata-shortcuts.md).

```{eval-rst}
.. autoclass:: batcher.api.dataset.meta.frame.DatasetMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.column.ColumnMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.checks.ColumnChecks
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.schema.SchemaMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.nulls.NullsMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.approx.ApproxMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.storage.StorageMeta
   :members:
   :member-order: bysource

.. autoclass:: batcher.api.dataset.meta.pair.PairMeta
   :members:
   :member-order: bysource
```

## SQL sessions

```{eval-rst}
.. autoclass:: batcher.Session
   :members:
   :member-order: groupwise
```

## Streaming

```{eval-rst}
.. autoclass:: batcher.Trigger
   :members:

.. autoclass:: batcher.OutputMode
   :members:
```

## Configuration classes

The tunables, grouped by subsystem. See the [configuration guide](configuration.md)
for what each one does and when to change it.

```{eval-rst}
.. autoclass:: batcher.Config
   :members:

.. autoclass:: batcher.ExecutionConfig
   :members:

.. autoclass:: batcher.MemoryConfig
   :members:

.. autoclass:: batcher.FlowControlConfig
   :members:

.. autoclass:: batcher.OptimizerConfig
   :members:

.. autoclass:: batcher.config.config.CardinalityConfig
   :members:

.. autoclass:: batcher.config.config.CostWeights
   :members:

.. autoclass:: batcher.config.config.CostCoefficients
   :members:

.. autoclass:: batcher.config.config.DistributedConfig
   :members:

.. autoclass:: batcher.PIDConfig
   :members:

.. autoclass:: batcher.MetadataConfig
   :members:

.. autoclass:: batcher.config.config.ObservabilityConfig
   :members:

.. autoclass:: batcher.config.config.ShuffleTlsConfig
   :members:
```
