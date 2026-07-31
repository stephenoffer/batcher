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
    .in_range("age", 0, 120)                             # inclusive low/high
    .matches("email", r"^[a-z]+@[a-z]+\.[a-z]+$")        # regex
    .accepted_values("country", ["US", "CA", "MX"])      # enum membership
    .check(bt.col("score") > 0.0, name="score_positive") # any Expr; name= required
)
```

**NULL policy — the trap that makes a check look like it passed.** `in_range`,
`matches`, and `accepted_values` treat **NULL as valid** (they lower to
`col IS NULL OR test`). A column that is 90% NULL passes `in_range` cleanly. Pair every
value constraint with `not_null` when NULL is not acceptable. By contrast, a `check()`
predicate that evaluates to NULL **counts as a violation**.

**`unique` counts duplicated *keys*, not duplicated rows.** Two rows with `id=2`
report `unique(id): 1` (one offending key) — but `drop()` removes **both** of them,
because neither can be shown to be the good one.

**`foreign_key` is not an assertion — it breaks the chain.** It returns a `Dataset`
of the **orphan rows** directly (an anti-join), so it cannot sit mid-chain:

```python
orphans = ds.dq.foreign_key("country", references=ref, ref_columns=None)
if orphans.count():
    ...  # referential integrity violated
```

## The trichotomy: fail vs drop vs quarantine

This is the decision the skill exists to make. All three consume the same `gate`.

| Terminal | Returns | Eager? | Use when |
|---|---|---|---|
| `validate()` | `ValidationReport` | yes | You want the numbers and will decide yourself |
| `fail()` | `Dataset` (unchanged) | yes | Bad data is a **bug** — halt the pipeline loudly |
| `drop()` | `Dataset` (filtered) | no | Bad rows are **noise** — silently discard them |
| `quarantine()` | `(clean, rejected)` | no | Bad rows are **evidence** — keep them for triage |

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

`ValidationReport` has one field, `violations: dict[str, int]`, plus the `ok` and
`total_violations` properties and a `__bool__` (so `if report:` reads as "clean").
Constraint names are auto-generated (`not_null(email)`, `in_range(age, 0, 120)`); a
`check()` uses your `name=` verbatim — name them well, they are your error message.
Print `str(report)`, not `repr` — the repr is the raw dataclass dump.

Verified output for a 6-row frame with one NULL email, a duplicated `id`, `age=200`,
an unlisted country, a malformed email, and a NULL score:

```
ok: False | total_violations: 6
{'not_null(email)': 1, 'in_range(age, 0, 120)': 1, 'accepted_values(country)': 1,
 "matches(email, '^[a-z]+@[a-z]+\\.[a-z]+$')": 1, 'score_positive': 1, 'unique(id)': 1}
drop() -> 1 row | quarantine() -> clean 1, bad 5
```

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
- [ ] `validate()`/`fail()` are not called inside a per-batch loop — they execute a pass
      each time; use the lazy `drop`/`quarantine` in the plan instead.

## See also

- `docs/user-guide/trust/data-quality.md`; `docs/user-guide/analyze/metadata-shortcuts.md`;
  `docs/cookbook/data-engineering/quality-gates.md`;
  `docs/examples/data-engineering/{deduplication,schema-evolution,cdc-pipeline}.md`.
- Skills: `write-a-batcher-pipeline` (relational basics, `map_batches`);
  `write-a-streaming-pipeline` (gating micro-batches with `for_each_batch`);
  `optimize-a-slow-query` (`stats()`/`explain(analyze=True)` — the *performance* tools);
  `debug-a-batcher-query` (a query that raises or returns wrong rows).
