# The .dt namespace

This page is the reference for `.dt`, the accessor a date, time, or timestamp expression carries. Reach it as {py:obj}`col("ts").dt <batcher.plan.expr_ir.core.Expr.dt>`. The methods extract calendar parts, truncate to a grain, do calendar arithmetic, and move between time zones.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.temporal

.. autoclass:: _DtNamespace
   :no-members:
```

## Date and time parts

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

## Truncation, rounding, and offsets

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

## Calendar flags

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

## Differences

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

## Epochs, names, formatting, and time zones

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

## See also

- {doc}`index`: the other accessor namespaces, and which column kind each one attaches to.
- {doc}`/api/relational/expression-accessors`: the same methods with a runnable example per namespace.
- {doc}`/api/symbols/expression-methods`: the {py:obj}`Expr <batcher.plan.expr_ir.core.Expr>` these namespaces hang off.
- {doc}`/user-guide/analyze/time-series`: bucketing and aligning timestamped data.
- {doc}`/cookbook/expressions/temporal/index`: runnable recipes for parts, truncation, and time zones.
