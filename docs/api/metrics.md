# Metrics reference

Scoring and statistical aggregates, from classification and regression metrics to
descriptive and inferential statistics. Every one is an ordinary aggregate, so it
composes with `group_by(...).agg(...)` and scores per segment in one pass.

The metrics that need a global ordering or return a table live in `batcher.ml.metrics`
instead, on {doc}`ml-models`.

```{eval-rst}
.. currentmodule:: batcher
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

## See also

:::{seealso}
- {doc}`../ml/evaluation`: the guide to scoring a model with these.
- {doc}`ml-models`: the table-returning metrics and the in-engine estimators.
- {doc}`ml-statistics`: drift comparisons and cross-validated scoring.
:::
