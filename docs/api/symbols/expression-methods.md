# Expr: the scalar algebra

This page lists the {py:obj}`Expr <batcher.plan.expr_ir.core.Expr>` methods that compute one output value per input row: naming and casting, null handling, arithmetic, trigonometry, and the bit, hash, and encoding operations. The methods that collapse rows are on {doc}`expression-aggregates`, and the ones that depend on what a column holds are behind an accessor namespace on {doc}`/api/accessors/index`.

`Expr` is the fluent builder every column operation returns. Nothing here evaluates. An expression is a tree the plan carries into Rust. Its methods are grouped below by the task you look them up by, and each one links to its own page.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.core

.. autoclass:: batcher.plan.expr_ir.core.Expr
   :no-members:
```

## Naming, casting, and membership

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

## Null and NaN handling

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

## Rounding, sign, and clamping

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

## Powers, logarithms, and special functions

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

## Trigonometry

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

## Bits, hashes, and text encodings

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

## See also

- {doc}`expression-aggregates`: the same object's aggregate, window, and rolling methods.
- {doc}`expression-modeling`: feature scaling, activations, introspection, and the IR.
- {doc}`/api/accessors/index`: the typed namespaces for strings, dates, nested data, and media.
- {doc}`/api/relational/expressions`: the same surface with a runnable example per group.
