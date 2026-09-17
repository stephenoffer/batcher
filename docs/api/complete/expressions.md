# Expressions and selectors

The column language: the functions that build an expression, the selectors that stand for
a set of columns, and the typed accessor namespaces an expression carries.

## Expression constructors

The free functions that produce an {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`. A
column reference, a literal, a conditional, and the constructors for the composite types.

```{eval-rst}
.. currentmodule:: batcher

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
   map_from_arrays
   array
```

## Column selectors

A *selector* stands for every column matching a predicate. Pass one anywhere a column is expected, such as {py:meth}`ds.select(bt.numeric()) <batcher.Dataset.select>` or {py:meth}`ds.with_columns(bt.floating().round(2)) <batcher.Dataset.with_columns>`, and it expands against the input schema. See the {doc}`transformations guide </user-guide/transform/rows/transformations>` for how they compose.

```{eval-rst}
.. currentmodule:: batcher

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

`Selector` subclasses `Expr`, and everything else follows from that. The scalar algebra
composes onto a selector exactly as it composes onto a column, and applies to every column
the selector matched, so `bt.floating().round(2)` is one expression that rounds however
many floating columns the input turns out to have. Two selectors combine with `|`, `&` and `-`, and `~` takes the complement. The
`.name` namespace renames the outputs.

```{eval-rst}
.. autoclass:: batcher.plan.expr_ir.selectors.Selector
   :members:

.. autoclass:: batcher.plan.expr_ir.selectors.core._SelectorNameNamespace
   :members:
   :member-order: bysource
```

## Expr

`Expr` is the fluent builder every column operation returns. Nothing here evaluates. An expression is a tree the plan carries into Rust. Its methods are grouped below by the task you look them up by, and each one links to its own page.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.core

.. autoclass:: batcher.plan.expr_ir.core.Expr
   :no-members:
```

### Naming, casting, and membership

Name an output, change its type, or test a value against a set, a range, or a recode map.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.alias
   Expr.cast
   Expr.pipe
   Expr.try_cast
   Expr.is_in
   Expr.between
   Expr.replace
```

### Null and NaN handling

Test for, replace, or fill across missing values, keeping SQL null and IEEE NaN apart.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.is_null
   Expr.is_not_null
   Expr.fill_null
   Expr.eq_missing
   Expr.is_nan
   Expr.is_not_nan
   Expr.fill_nan
   Expr.is_finite
   Expr.is_infinite
   Expr.forward_fill
   Expr.backward_fill
   Expr.interpolate
```

### Rounding, sign, and clamping

Round a numeric value, take its magnitude or sign, clamp it into bounds, or test its sign and parity.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.round
   Expr.floor
   Expr.ceil
   Expr.trunc
   Expr.even
   Expr.abs
   Expr.abs_diff
   Expr.sign
   Expr.clip
   Expr.is_zero
   Expr.is_positive
   Expr.is_negative
   Expr.is_even
   Expr.is_odd
```

### Powers, logarithms, and special functions

Roots, powers, exponentials and logarithms, the null-safe division, and the factorial and gamma functions.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.sqrt
   Expr.cbrt
   Expr.square
   Expr.exp
   Expr.expm1
   Expr.ln
   Expr.log10
   Expr.log2
   Expr.log1p
   Expr.safe_divide
   Expr.factorial
   Expr.gamma
   Expr.lgamma
```

### Trigonometry

Trigonometric and hyperbolic functions of an angle in radians, the degree conversions, and the Polars/NumPy `arc*` spellings of the inverses.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.sin
   Expr.cos
   Expr.tan
   Expr.cot
   Expr.sec
   Expr.csc
   Expr.sinh
   Expr.cosh
   Expr.tanh
   Expr.degrees
   Expr.radians
   Expr.arcsin
   Expr.arccos
   Expr.arctan
   Expr.arcsinh
   Expr.arccosh
   Expr.arctanh
```

### Bits, hashes, and text encodings

Per-row bitwise operations on integers, deterministic hashing, and the functions that render a number as text.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.bitwise_and
   Expr.bitwise_or
   Expr.bitwise_xor
   Expr.bitwise_left_shift
   Expr.bitwise_right_shift
   Expr.bit_count
   Expr.hash
   Expr.hash_bucket
   Expr.chr
   Expr.to_base
   Expr.format_bytes
```

### Aggregation

The per-group reductions used in `group_by().agg(...)` or bound to a window with `.over(...)`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.sum
   Expr.count
   Expr.mean
   Expr.min
   Expr.max
   Expr.median
   Expr.count_distinct
   Expr.approx_count_distinct
   Expr.first
   Expr.last
   Expr.any_value
   Expr.min_by
   Expr.max_by
   Expr.arg_min
   Expr.arg_max
   Expr.array_agg
   Expr.histogram
```

### Statistical, logical, and bitwise aggregates

Aggregates that describe a group's distribution (spread, quantiles, most frequent values, shape, and the assembly-contiguity statistics) or fold its non-null values by AND, OR, XOR, product, or a compensated sum.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.std
   Expr.var
   Expr.quantile
   Expr.quantile_disc
   Expr.approx_quantile
   Expr.approx_median
   Expr.mode
   Expr.mode_top_k
   Expr.top_k
   Expr.skew
   Expr.kurtosis
   Expr.mad
   Expr.entropy
   Expr.n50
   Expr.n90
   Expr.l50
   Expr.aun
   Expr.bool_and
   Expr.bool_or
   Expr.bit_and
   Expr.bit_or
   Expr.bit_xor
   Expr.product
   Expr.kahan_sum
```

### Windows, ranking, and offsets

Bind an expression to a window with `over`, or use the window helpers that rank rows, accumulate up to the current row, look back by `n` rows, or flag duplicates, runs, and peaks.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.over
   Expr.cum_sum
   Expr.cum_count
   Expr.cum_min
   Expr.cum_max
   Expr.cum_prod
   Expr.shift
   Expr.diff
   Expr.pct_change
   Expr.rank
   Expr.rank_pct
   Expr.rle_id
   Expr.is_first_distinct
   Expr.is_last_distinct
   Expr.is_duplicated
   Expr.is_unique
   Expr.peak_max
   Expr.peak_min
```

### Rolling, expanding, and exponentially weighted windows

Moving statistics over a fixed count of rows, a time window, every row so far, or an exponential decay.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.rolling_sum
   Expr.rolling_mean
   Expr.rolling_min
   Expr.rolling_max
   Expr.rolling_count
   Expr.rolling_std
   Expr.rolling_var
   Expr.rolling_sum_by
   Expr.rolling_mean_by
   Expr.rolling_min_by
   Expr.rolling_max_by
   Expr.rolling_count_by
   Expr.expanding_mean
   Expr.expanding_std
   Expr.expanding_var
   Expr.ewm_mean
   Expr.ewm_std
   Expr.ewm_var
   Expr.ewm_mean_by
```

### Scaling and feature engineering

Column-wide transforms that standardize, scale, normalize, bin, encode, or flag outliers.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.zscore
   Expr.minmax_scale
   Expr.maxabs_scale
   Expr.mean_center
   Expr.normalize_l1
   Expr.pct_of_total
   Expr.cumulative_pct
   Expr.softmax
   Expr.label_encode
   Expr.cut
   Expr.is_outlier
```

### Activation functions

The neural-network activations, each applied element-wise to a numeric column.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.relu
   Expr.leaky_relu
   Expr.elu
   Expr.gelu
   Expr.sigmoid
   Expr.logit
   Expr.softplus
   Expr.softsign
   Expr.silu
   Expr.mish
   Expr.hardsigmoid
   Expr.hardswish
   Expr.hardtanh
   Expr.tanhshrink
```

### Accessor namespaces

The properties that reach each typed namespace, documented in full on the pages linked from the expression accessors section below.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.str
   Expr.dt
   Expr.list
   Expr.struct
   Expr.json
   Expr.map
   Expr.image
   Expr.audio
   Expr.video
   Expr.seq
   Expr.meta
```

### Expression introspection

The `.meta` accessor reads an expression's tree rather than any row: the output name it would take, the columns it reads, and a drawing of the tree the engine receives.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.meta

.. autoclass:: _MetaNamespace
   :no-members:

.. autosummary::
   :toctree: generated
   :nosignatures:

   _MetaNamespace.output_name
   _MetaNamespace.root_names
   _MetaNamespace.is_column
   _MetaNamespace.has_multiple_outputs
   _MetaNamespace.tree_format
```

### Python protocols and the IR

How an expression behaves under Python's indexing, iteration, and membership syntax, and how it serializes to the engine's JSON IR.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   Expr.__getitem__
   Expr.__contains__
   Expr.__iter__
   Expr.__len__
   Expr.to_ir
```

## AggExpr

Call an aggregate or a window function on an `Expr` and you get an `AggExpr` back, which adds `.over(...)` for binding a window.

```{eval-rst}
.. currentmodule:: batcher

.. autoclass:: batcher.AggExpr
   :no-members:
```

### Windows, naming, and casting on an aggregate

Bind an aggregate to a window, name or cast its result, or lower it to its JSON `AggregateItem`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   AggExpr.over
   AggExpr.alias
   AggExpr.cast
   AggExpr.to_ir
```

### Math on an aggregate result

The scalar math that applies to an aggregate's result, with the same meaning as on `Expr`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   AggExpr.round
   AggExpr.floor
   AggExpr.ceil
   AggExpr.trunc
   AggExpr.abs
   AggExpr.sign
   AggExpr.clip
   AggExpr.sqrt
   AggExpr.cbrt
   AggExpr.square
   AggExpr.exp
   AggExpr.expm1
   AggExpr.ln
   AggExpr.log10
   AggExpr.log2
   AggExpr.log1p
```

## Expression accessors

The typed namespaces reached as {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`, {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>`, and {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>`. Multimodal columns add {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>`, {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>`, and {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>`. Biological sequence columns add {py:class}`.seq <batcher.plan.expr_ir.namespaces.sequence._SeqNamespace>`.

Their methods are documented on three pages: {doc}`string-accessor` for `.str`, {doc}`temporal-and-nested-accessors` for `.dt`, `.list`, `.struct`, `.json`, and `.map`, and {doc}`multimodal-and-sequence-accessors` for `.image`, `.audio`, `.video`, and `.seq`.

## See also

- {doc}`/api/relational/expressions`: the same surface with runnable examples, grouped by the job each method does.
- {doc}`/api/relational/expression-accessors`: every accessor method enumerated in tables, which is faster to scan than the generated accessor pages.
- {doc}`/api/relational/functions`: the scalar, horizontal, aggregate, and window functions these constructors sit beside.
- {doc}`/user-guide/transform/columns/expressions`: the mental model behind an expression, and when it evaluates.
