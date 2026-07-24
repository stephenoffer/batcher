# Functions reference

Every top-level function you call on or around an expression: the row-wise scalar
helpers, the text-extraction and prompt-building functions for LLM pipelines, the
horizontal reducers that work across columns within a row, and the aggregates and window
functions that reduce down a column.

The `Expr` *methods* these compose with are in {doc}`expressions`. The metric aggregates
have their own page, {doc}`metrics`.

```{eval-rst}
.. currentmodule:: batcher
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
   great_circle_distance
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
   make_date
   make_timestamp
   from_epoch
   from_unix_date
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

## See also

:::{seealso}
- {doc}`expressions`: the `Expr` methods and accessor namespaces.
- {doc}`metrics`: the scoring and statistical aggregates.
- {doc}`../user-guide/expressions`: the same language taught rather than tabulated.
:::
