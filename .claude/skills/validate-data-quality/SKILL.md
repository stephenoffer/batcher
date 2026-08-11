---
name: validate-data-quality
description: Assert data-quality contracts on a Batcher Dataset with the ds.dq builders and the fail/drop/quarantine trichotomy, and profile columns for nulls, cardinality, distributions, and correlation before writing the checks. Invoke when asked to validate, gate, clean, quarantine, or profile a dataset, or to explain why rows are being rejected.
---

# Validate data quality

`ds.dq` is the constraint surface; the profiling methods are how you learn what to
constrain. Relational basics (lazy plans, expressions, `group_by`) live in
`write-a-batcher-pipeline` — this skill covers only the quality surface.

## Mental model

- **`ds.dq` is an immutable builder.** Each assertion returns a **new** `DatasetDQ`;
  `gate is ds.dq` is `False`. Build the contract once and reuse it — the same `gate`
  can be validated, dropped, and quarantined without rebuilding.
- **Assertions are declarations; terminals do the work.** Nothing runs until you call
  `validate`, `fail`, `drop`, or `quarantine`.
- **`validate`/`fail` are eager** (they execute an aggregate now). **`drop`/`quarantine`
  are lazy** — they only add plan nodes, so they compose into a bigger pipeline.

## The assertions

```python
gate = (
    ds.dq
    .not_null("email", "user_id")                        # *cols, varargs
    .unique("id")                                        # str | list[str] (composite)
    .in_range("age", 0, 120)                             # inclusive; closed= narrows it
    .positive("score")                                   # strict= admits zero
    .is_finite("ratio")                                  # no NaN, no infinity
    .accepted_values("country", ["US", "CA", "MX"])      # enum membership
    .rejected_values("status", ["N/A", "unknown"])       # deny-list
    .matches("email", r"^[a-z]+@[a-z]+\.[a-z]+$")        # regex
    .not_matches("note", r"TODO")                        # regex, negated
    .matches_format("email", "email")                    # email | url | uuid | ipv4
    .str_length_between("iso", 2, 2)                     # characters, not bytes
    .not_empty("name")                                   # "" and "   " are not values
    .compare_columns("start", "<=", "end")               # two columns, one row
    .not_in_future("event_time", tolerance="5m")         # clock-skew tolerant
    .references("customer_id", to=customers)             # referential integrity
    .check(bt.col("score") > 0.0, name="score_positive") # any Expr; name= required
)
```

Every one of those takes two keyword modifiers:

- **`mostly=0.99`** — the fraction of rows that must pass for the constraint to *pass*
  (Great Expectations' tolerance). It moves the pass/fail line only: the violating rows
  are still counted and still dropped.
- **`severity="warn"`** — report without enforcing. A warning never raises in `fail`,
  never removes a row in `drop`, never lands on the rejected side of `quarantine`. This
  is how a new rule is watched in production before it is switched on.

**`where(predicate)` scopes the constraints added after it** to the matching rows; a row
outside the scope passes vacuously. It applies to row-level constraints only, and refuses
(rather than silently ignoring) a uniqueness, referential, or relation-level check.

```python
ds.dq.where(bt.col("country") == "US").not_null("state")
```

**Relation-level constraints** measure the whole table and no row can violate one, so
`drop`/`quarantine`/`annotate` **refuse** them with a `PlanError` rather than enforcing a
subset of your contract. Use `validate`/`fail`.

```python
ds.dq.row_count_between(1, 10_000_000)      # volume
  .mean_between("amount", 1.0, 500.0)       # also sum_/median_/stddev_/quantile_between
  .null_rate_below("email", 0.02)           # tolerated not_null
  .distinct_count_between("status", 1, 12)  # vocabulary size
  .unique_ratio_above("id", 0.99)           # "nearly a key", scale-free
  .fresh_within("event_time", "1d")         # newest row's age vs the wall clock
```

**Schema constraints** are answered from `ds.schema` before anything executes, so put them
first: when a column is missing, every value constraint written against it also fails, and
the report then names five broken checks instead of the one cause.

```python
ds.dq.has_columns("id", "amount")
  .column_types({"id": "int64", "amount": "float64"})
  .no_unexpected_columns("id", "amount")     # catches a widened upstream schema
```

An unsatisfied schema contract raises `DataQualityError` from `drop`/`quarantine`/
`annotate` — a missing column cannot be quarantined.

**NULL policy — the trap that makes a check look like it passed.** Every value constraint
except `not_null` treats **NULL as valid** (they lower to `col IS NULL OR test`). A column
that is 90% NULL passes `in_range` cleanly. Pair every value constraint with `not_null`
when NULL is not acceptable. By contrast, a `check()` predicate that evaluates to NULL
**counts as a violation**.

**`unique` counts duplicated *rows*, not duplicated keys.** Over `[1, 1, 1, 2]` the report
says `unique(id): 3`, which is exactly what `drop()` removes and what `quarantine()`
rejects — the report and the split agree by construction.

**`references` vs `foreign_key`.** `references(cols, to=other)` is a *constraint*: orphans
are counted, dropped, and quarantined with everything else in the chain. `foreign_key`
is a *terminal* that hands back the orphan rows themselves, for when the orphans are the
answer. Neither treats a NULL key as an orphan.

```python
gate = ds.dq.references("customer_id", to=customers)     # mid-chain
orphans = ds.dq.foreign_key("customer_id", references=customers)   # the rows
```

**`on(other)` rebinds a whole chain to another dataset**, which is how one contract runs
against today's partition and yesterday's without a second way to spell it.

## The trichotomy: fail vs drop vs quarantine

This is the decision the skill exists to make. All three consume the same `gate`.

| Terminal | Returns | Eager? | Use when |
|---|---|---|---|
| `validate()` | `ValidationReport` | yes | You want the numbers and will decide yourself |
| `fail()` | `Dataset` (unchanged) | yes | Bad data is a **bug** — halt the pipeline loudly |
| `drop()` | `Dataset` (filtered) | no | Bad rows are **noise** — silently discard them |
| `quarantine()` | `(clean, rejected)` | no | Bad rows are **evidence** — keep them for triage |
| `annotate()` | `Dataset` (+1 column) | no | You want every row kept, each labelled with what it failed |

```python
report = gate.validate()          # never raises
if not report.ok:
    print(report.violations)      # {'not_null(email)': 1, 'unique(id)': 1, ...}

gate.fail()                       # raises DataQualityError if any violation
kept = gate.drop()                # lazy filtered Dataset
clean, bad = gate.quarantine()    # BOTH lazy Datasets — a total partition
```

`quarantine()` returns a **2-tuple `(clean, rejected)`** and the split is total:
`clean.count() + rejected.count() == ds.count()`, always. This is the production
default — write `clean` to the table and `bad` to a dead-letter path in the same run.

```python
clean, bad = gate.quarantine()
clean.write.parquet("s3://lake/users/", mode="append")
bad.write.parquet("s3://lake/_rejected/users/", mode="append")
```

`fail()` raises `DataQualityError`, which carries `.violations`. It is **not** exported
at top level — import it explicitly:

```python
from batcher._internal.errors import DataQualityError
```

`ValidationReport` holds `results: tuple[ConstraintResult, ...]`, one per constraint in
declaration order, plus `violations` (the name→count mapping), `ok`, `total_violations`,
`rows`, `failed`, `warnings`, `result(name)`, `to_dict()`, and a `__bool__` (so
`if report:` reads as "clean"). Each `ConstraintResult` carries `violations`, `rows`,
`pass_rate`, `ok`, `blocking`, `severity`, `mostly`, `kind`, the measured `value` for a
relation-level check, and `detail` for a schema one.
Constraint names are auto-generated (`not_null(email)`, `in_range(age, 0, 120)`); a
`check()` uses your `name=` verbatim — name them well, they are your error message.
Print `str(report)`, not `repr` — the repr is the raw dataclass dump.

Verified output for a 6-row frame with one NULL email, a duplicated `id`, `age=200`,
an unlisted country, a malformed email, and a NULL score:

```
ok: False | total_violations: 7
{'not_null(email)': 1, 'in_range(age, 0, 120)': 1, 'accepted_values(country)': 1,
 "matches(email, '^[a-z]+@[a-z]+\\.[a-z]+$')": 1, 'score_positive': 1, 'unique(id)': 2}
drop() -> 1 row | quarantine() -> clean 1, bad 5
```

`annotate()` keeps all six rows and adds a `dq_failed` column naming each row's failures,
comma-separated, empty for a clean row — which makes "which rule, how often, since when"
a `group_by` instead of a re-run.

## Let the data write the first draft

`ds.dq.suggest()` profiles the relation and **appends** the constraints it already satisfies
to the chain — completeness, keys, sign, small enumerations, and an observed null rate with
headroom. It executes (a profile pass, a numeric-minimum pass, and one `distinct()` per
enumeration candidate, capped at eight), so it is a profiling step, not a pipeline stage.

```python
proposed = ds.dq.suggest()          # or suggest(["col_a", "col_b"])
print(repr(proposed))               # read it, delete the coincidences, keep the contract
```

It deliberately never proposes an `in_range` off an observed min/max: tomorrow's legitimate
value is outside today's, and a check that cries wolf gets deleted. Treat every suggestion
as a draft — a column that happens to be complete in this sample gets a `not_null` it may
not deserve.

## Contract health over time

Every constraint `validate()` evaluates is published to the event bus and folded into
`observe`'s counters, passing ones included. `metrics_snapshot()["data_quality"]` carries
`checks_total`, `failed_total`, `violations_total`, and a `by_constraint` breakdown;
Prometheus sees `batcher_dq_checks_total`, `batcher_dq_failed_total`,
`batcher_dq_violations_total`, and `batcher_dq_constraint_violations_total{constraint="..."}`.

That is the difference between "is today's data good" (the report) and "has this constraint
been degrading for a week" (the series). Pair it with `severity="warn"` to watch a rule
before enforcing it.

## Profiling — decide what to assert

Run these *before* writing constraints; they tell you the real null rates, ranges, and
cardinalities so the thresholds are measured rather than guessed.

```python
ds.profile()        # one row PER COLUMN: column, count, null_count, null_fraction, approx_distinct
ds.describe()       # statistic-rows x column-cols: count/null_count/mean/std/min/25%/50%/75%/max
ds.null_count()     # lazy Dataset, one row: null count per column
ds.value_counts("country")            # column, count — sorted desc by default
ds.class_balance("label")             # column, fraction — label skew before training
ds.crosstab("country", "tier")        # contingency table
ds.corr("age", "score")               # float | None
ds.cov("age", "score", ddof=1)        # float | None
ds.drop_constant_columns()            # prune zero-information columns
ds.n_null("email"); ds.has_nulls("email"); ds.all_null("email")   # int / bool / bool
```

**`ds.profile()` is a data-quality column profiler, not a performance profiler.** This
confusion is common and costs real time. For performance use `ds.explain(analyze=True)`
and `ds.stats()` — the latter returns `RunStats` (measured per-operator rows, wall time,
spill) and is unrelated to descriptive statistics. See `optimize-a-slow-query`.

`describe()` is numerics-first (non-numeric columns come back as `None` for
mean/std/percentiles); `profile()` works on every column and costs **one aggregate pass**
regardless of width, using HyperLogLog for `approx_distinct`. Reach for `profile()` on
wide/unknown data, `describe()` when you already know the numeric columns you care about.

## Quality metrics as expressions

These are `Expr`s, so they compose into `agg` and — the point — into `group_by`, which
turns a global null rate into a per-partition quality dashboard:

```python
ds.agg(
    email_nulls=bt.null_rate("email"),        # fraction NULL
    email_filled=bt.non_null_rate("email"),   # 1 - null_rate
    id_uniqueness=bt.nunique_ratio("id"),     # n_distinct / n_rows -> 1.0 means a key
)

ds.group_by("ingest_date").agg(bad=bt.null_rate("score"))   # find the bad partition
```

`bt.histogram(column)` takes **one** argument and returns a value→count map aggregate
(`[('CA', 3), ('US', 2)]`) — it does **not** bucket numerics. For equal-width numeric
bucketing use `bt.width_bucket(value, low, high, count)`:

```python
ds.select("age", bucket=bt.width_bucket(bt.col("age"), 0, 100, 4))
```

An approximate equal-probability histogram lives at `ds.meta.approx.histogram("amount", 4)`
— note it is `ds.meta.approx`, **not** `ds.approx`, which does not exist.

## Metadata shortcuts — checks that cost nothing

A *total* constraint can sometimes be discharged from Parquet footer stats and zone maps
without scanning: `ds.dq.in_range("x", -10, 10).drop() is ds` returns the **identity
dataset** when the footers already prove the range holds. Keep constraints total to stay
eligible — `check()` is never provable (it always executes), and composite uniqueness
generally falls back to a real scan. The broader metadata surface (`ds.meta`, exact vs
approximate stats, what is provable from footers) is its own topic; see
`docs/user-guide/analyze/metadata-shortcuts.md` rather than duplicating it here.

## Self-check

- [ ] Every `in_range`/`matches`/`accepted_values` on a nullable column is paired with
      `not_null` — otherwise NULLs pass silently.
- [ ] The `fail` / `drop` / `quarantine` choice is deliberate: raise for bugs, drop for
      noise, quarantine when the rejects must be inspectable. Prefer `quarantine` in
      production.
- [ ] `quarantine()` is unpacked as `(clean, rejected)` and **both** sides are consumed —
      a discarded `bad` is a silently dropped row.
- [ ] The gate is built once and reused; assertions are not rebuilt per terminal.
- [ ] `check()` constraints have descriptive `name=` values — they become the report keys.
- [ ] `DataQualityError` is imported from `batcher._internal.errors`, not `bt`.
- [ ] Thresholds came from `profile()`/`describe()` on real data, not from guessing.
- [ ] `ds.profile()` is not being used to diagnose slowness (that is
      `explain(analyze=True)` / `stats()`).
- [ ] A relation-level contract exists at all: `row_count_between` and `fresh_within`
      catch the failures every row-level check is structurally blind to.
- [ ] A schema constraint comes first in the chain, so a missing column is reported once
      rather than as five broken value checks.
- [ ] `mostly=`/`severity=` are used deliberately — a rule that would be switched off for
      flapping is better tolerated or demoted to `warn` than deleted.
- [ ] `validate()`/`fail()` are not called inside a per-batch loop — they execute a pass
      each time; use the lazy `drop`/`quarantine` in the plan instead.

## See also

- `docs/user-guide/trust/data-quality.md`; `docs/user-guide/trust/data-contracts.md`; `docs/user-guide/analyze/metadata-shortcuts.md`;
  `docs/cookbook/data-engineering/maintenance/quality-gates.md`;
  `docs/cookbook/dataset/cleaning/deduplication.md`, `docs/cookbook/data-engineering/modeling/schema-evolution.md`, `docs/cookbook/data-engineering/ingest/cdc-pipeline.md`.
- Skills: `write-a-batcher-pipeline` (relational basics, `map_batches`);
  `write-a-streaming-pipeline` (gating micro-batches with `for_each_batch`);
  `optimize-a-slow-query` (`stats()`/`explain(analyze=True)` — the *performance* tools);
  `debug-a-batcher-query` (a query that raises or returns wrong rows).
