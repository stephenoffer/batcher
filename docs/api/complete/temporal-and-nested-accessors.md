# Temporal and nested accessors reference

This page is the generated reference for the `.dt` namespace on date and timestamp expressions and the `.list`, `.struct`, `.json`, and `.map` namespaces on nested and JSON columns. You reach each one from an expression, such as `col("ts").dt` or `col("tags").list`, and each method has its own page.

## The `.dt` namespace

Methods on a date or timestamp expression, reached as `col("ts").dt`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.temporal

.. autoclass:: _DtNamespace
   :no-members:
```

### Date and time parts

Extract one component of a date or timestamp, such as its year, its hour, or its calendar date.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.temporal

.. autosummary::
   :toctree: generated
   :nosignatures:

   _DtNamespace.year
   _DtNamespace.quarter
   _DtNamespace.month
   _DtNamespace.week
   _DtNamespace.day
   _DtNamespace.hour
   _DtNamespace.minute
   _DtNamespace.second
   _DtNamespace.millisecond
   _DtNamespace.microsecond
   _DtNamespace.nanosecond
   _DtNamespace.dayofweek
   _DtNamespace.weekday
   _DtNamespace.dayofyear
   _DtNamespace.week_of_month
   _DtNamespace.iso_year
   _DtNamespace.decade
   _DtNamespace.century
   _DtNamespace.millennium
   _DtNamespace.date
   _DtNamespace.time_of_day
```

### Truncation, rounding, and offsets

Move a date or timestamp to a period boundary, or shift it by an offset.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.temporal

.. autosummary::
   :toctree: generated
   :nosignatures:

   _DtNamespace.truncate
   _DtNamespace.ceil
   _DtNamespace.round
   _DtNamespace.normalize
   _DtNamespace.month_start
   _DtNamespace.quarter_start
   _DtNamespace.quarter_end
   _DtNamespace.year_start
   _DtNamespace.year_end
   _DtNamespace.last_day
   _DtNamespace.next_day
   _DtNamespace.offset_by
```

### Calendar flags

Test where a date falls in its week, month, quarter, or year, and count the days in its period.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.temporal

.. autosummary::
   :toctree: generated
   :nosignatures:

   _DtNamespace.is_weekend
   _DtNamespace.is_business_day
   _DtNamespace.is_month_start
   _DtNamespace.is_month_end
   _DtNamespace.is_quarter_start
   _DtNamespace.is_quarter_end
   _DtNamespace.is_year_start
   _DtNamespace.is_year_end
   _DtNamespace.is_leap_year
   _DtNamespace.is_between_time
   _DtNamespace.days_in_month
   _DtNamespace.days_in_year
```

### Differences

Count whole units elapsed between two dates or timestamps.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.temporal

.. autosummary::
   :toctree: generated
   :nosignatures:

   _DtNamespace.months_between
   _DtNamespace.days_between
   _DtNamespace.hours_between
   _DtNamespace.minutes_between
   _DtNamespace.seconds_between
   _DtNamespace.weeks_between
```

### Epochs, names, formatting, and time zones

Convert a date or timestamp to an epoch count or text, or move it between time zones.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.temporal

.. autosummary::
   :toctree: generated
   :nosignatures:

   _DtNamespace.epoch
   _DtNamespace.epoch_ms
   _DtNamespace.epoch_us
   _DtNamespace.epoch_ns
   _DtNamespace.timestamp
   _DtNamespace.dayname
   _DtNamespace.monthname
   _DtNamespace.strftime
   _DtNamespace.convert_timezone
```

## The `.list` namespace

Methods on a list expression, reached as `col("xs").list`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autoclass:: _ListNamespace
   :no-members:
```

### Access

Read the length of a list, take elements out of it by position, or find a value in it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.len
   _ListNamespace.get
   _ListNamespace.first
   _ListNamespace.last
   _ListNamespace.head
   _ListNamespace.slice
   _ListNamespace.gather
   _ListNamespace.contains
   _ListNamespace.position
```

### Reshaping and transforming

Sort, deduplicate, filter, map, or flatten each list, or join its elements into a string.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.sort
   _ListNamespace.arg_sort
   _ListNamespace.reverse
   _ListNamespace.unique
   _ListNamespace.drop_nulls
   _ListNamespace.filter
   _ListNamespace.transform
   _ListNamespace.flatten
   _ListNamespace.concat
   _ListNamespace.append
   _ListNamespace.prepend
   _ListNamespace.remove
   _ListNamespace.join
   _ListNamespace.cum_sum
   _ListNamespace.diff
```

### Aggregates

Reduce each list to one value.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.sum
   _ListNamespace.product
   _ListNamespace.min
   _ListNamespace.max
   _ListNamespace.mean
   _ListNamespace.median
   _ListNamespace.std
   _ListNamespace.var
   _ListNamespace.n_unique
   _ListNamespace.arg_min
   _ListNamespace.arg_max
   _ListNamespace.entropy
```

### Set operations and overlap

Combine two lists as sets, or measure how much they share.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.union
   _ListNamespace.intersect
   _ListNamespace.difference
   _ListNamespace.has_any
   _ListNamespace.has_all
   _ListNamespace.jaccard
   _ListNamespace.multiset_overlap
   _ListNamespace.lcs_length
```

### Vector arithmetic and norms

Treat each list as a numeric vector, such as an embedding, and compute on it element-wise.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.dot
   _ListNamespace.add
   _ListNamespace.subtract
   _ListNamespace.multiply
   _ListNamespace.l2_norm
   _ListNamespace.l1_norm
   _ListNamespace.max_abs
   _ListNamespace.sum_squares
   _ListNamespace.normalize
   _ListNamespace.softmax
   _ListNamespace.log_softmax
   _ListNamespace.is_unit_norm
   _ListNamespace.is_zero_vector
```

### Vector distances and signatures

Compare two vector columns, or hash a vector into a signature.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ListNamespace.cosine_similarity
   _ListNamespace.cosine_distance
   _ListNamespace.angular_distance
   _ListNamespace.l2_distance
   _ListNamespace.l1_distance
   _ListNamespace.hamming_distance
   _ListNamespace.simhash
```

## The `.struct` namespace

Methods on a struct expression, reached as `col("s").struct`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autoclass:: _StructNamespace
   :no-members:
```

Read a struct's fields by name, or list the field names.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _StructNamespace.field
   _StructNamespace.get
   _StructNamespace.keys
```

## The `.json` namespace

Methods on a string column holding JSON documents, reached as `col("doc").json`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autoclass:: _JsonNamespace
   :no-members:
```

Read typed values at a JSON path, and inspect the document's keys, types, and shape.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _JsonNamespace.extract_string
   _JsonNamespace.extract_int
   _JsonNamespace.extract_float
   _JsonNamespace.extract_bool
   _JsonNamespace.value
   _JsonNamespace.values
   _JsonNamespace.exists
   _JsonNamespace.contains
   _JsonNamespace.keys
   _JsonNamespace.array_length
   _JsonNamespace.type_of
   _JsonNamespace.structure
   _JsonNamespace.pretty
```

## The `.map` namespace

Methods on an Arrow `Map` column, reached as `col("m").map`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autoclass:: _MapNamespace
   :no-members:
```

Look up a key, test for one, or read a map's keys, values, and entries as lists.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.collections

.. autosummary::
   :toctree: generated
   :nosignatures:

   _MapNamespace.get
   _MapNamespace.contains
   _MapNamespace.len
   _MapNamespace.keys
   _MapNamespace.values
   _MapNamespace.entries
```

## See also

- {doc}`/api/relational/expression-accessors`: every accessor method enumerated in one curated page.
- {doc}`expressions`: the `Expr` class these namespaces hang off.
- {doc}`/user-guide/transform/columns/expression-accessors`: how the accessor namespaces work.
- {doc}`/user-guide/transform/columns/map-accessor`: working with map columns.
- {doc}`/user-guide/analyze/time-series`: bucketing and aligning timestamped data.
