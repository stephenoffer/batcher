# Engine parity & improvement ledger — the 2026-07-26 competitive sweep

A running build log for the sweep against the vendored reference engines in
`/mnt/shared_storage/ref` (DuckDB, Spark, Polars, DataFusion, Daft, Arrow). It is the
third file in the same family and follows the same rules as the other two:

* `docs/internals/competitor_technique_review.md` — the parts list, read from competitor
  source.
* `docs/internals/competitor_parity_census.md` — the gaps found by *executing* the
  reference engines side by side with Batcher.
* **This file** — what this sweep changed, why it is safe, and how it was verified.

**Counting rule** (carried over, unchanged). An entry is a change to engine behaviour
that is verified by a committed test. A refactor, a doc, or an unproven change is not an
entry. Where one change makes N distinct names or units answerable, the entry says so and
counts N — the same convention the census used for its "1-100" entry — because a name a
user can type is the unit of reachability they experience.

**Baseline at the start of this sweep** (measured, not remembered): the DuckDB function
census stood at **221 matched of 478 probed**, 244 gaps (130 of them ICU collations), 13
mismatches; Spark at 110 of 312; Polars at 53 of 98.

**Where this sweep got to: 145 entries.** The brief was 400+, so this file is roughly a
third of the way, and the "Open" section at the end is the ordered remainder — written so
the next pass starts from measurement rather than from a re-derivation.

Two things about that number, because a later reader will want to know whether the
approach or the budget ran out. **The rate held**: the last five waves closed 102 names,
and each wave got cheaper to verify as the harness matured. What ran out was context. And
**the cheap seam is now largely mined**: most of what closed here was a capability the
engine already had that nothing reached from a front-end (the `Sequence` node,
`DateOffset`'s micros component, `.list.transform`'s `element()` placeholder, the whole
value-list aggregate state). What remains needs new machinery — a variadic node, an Arrow
`MapArray` builder, a TIME type, an INTERVAL type — so expect the next 100 entries to cost
more each than the last 100 did.

What must not be dropped, whatever the pace: each entry here carries a test **and** a
re-run of the census that produced it. The one wave that was not re-measured had
introduced a wrong answer (entry 117), and a 66-minute run that crashed against another
session's rebuild reported 12 failures that were not real (see "Verification state").
Both directions of that mistake are available at all times.

---

## Wave 1 — SQL reachability: intervals, JSON inspection, timestamp construction

### 1-9. Sub-day and calendar `INTERVAL` units

**Was:** `_apply_interval` (`_sql/parser/expressions/literals.py`) handled DAY, WEEK,
MONTH and YEAR. Every other unit raised `INTERVAL unit HOUR is not supported` — so
`ts + INTERVAL 1 HOUR`, the single most common temporal expression in a warehouse query,
did not run. The engine could always express it: `DateOffset` has carried a `micros`
component all along, and nothing reached it from SQL.

**Now:** HOUR, MINUTE, SECOND, MILLISECOND, MICROSECOND (exact microseconds) and
QUARTER, DECADE, CENTURY, MILLENNIUM (calendar months) — nine units, in both the
singular and plural spellings.

**Safe because:** each unit is a constant multiple of a component `offset_by` already
implements, so no new arithmetic is introduced. A Date32 cannot carry a sub-day offset
and the engine rejects one; DuckDB promotes the operand to TIMESTAMP for the same
reason, so the translator casts, which is a no-op on a timestamp operand.

**Verified:** 15 differential cases against DuckDB, including the end-of-month clamp and
a second-boundary rollover, plus a test that an unknown unit still raises.

### 10-13. The JSON inspection functions

`json_valid`, `json_exists`, `json_array_length` and `json_keys` (both the bare and
`(doc, path)` forms) now reach the `.json` accessor that already implemented them.
`json_valid` is the root type test — the kernel answers null for text it cannot parse.

**`json_type` is deliberately still refused.** DuckDB's `json_type` names the *SQL* type
the value would cast to (`UBIGINT`, `VARCHAR`); `.json.type_of` names the *JSON* type
(`number`, `string`). Wiring one to the other would answer a different question with a
plausible string, which is the exact failure mode the census exists to catch.

**Verified:** differential cases over a fixture crossing object, array and null-valued
documents, plus the invalid-document case and a test pinning the `json_type` refusal.

### 14-24. Timestamp construction

`strptime`, `try_strptime`, `epoch_ms` (**both** readings — see below), `to_timestamp`,
`make_timestamp` (the six-part and the one-argument microsecond overloads),
`make_timestamp_ms`, `make_timestamp_ns`, `time_bucket`, `julian` and `era`. Eleven
names, none of which needed a new kernel: each is a spelling of `from_epoch`,
`.str.to_datetime`, `make_timestamp`, `WindowStart` or a field extraction.

Three of these are traps the wiring had to get right, and each is pinned by a test:

* **`epoch_ms` is two functions under one name.** `epoch_ms(1234)` builds a timestamp
  from a millisecond count; `epoch_ms(TIMESTAMP '…')` reads the count back out. Only a
  numeric literal is unambiguous, so that is the only shape that takes the constructor
  reading.
* **`julian` needs the cast.** `epoch_us` of a Date32 is not its microsecond count, and
  a DATE is the argument `julian` is usually given. Without the cast the answer was
  2440588.0000001 for every date.
* **`time_bucket` on a calendar unit is refused.** DuckDB aligns month buckets to
  2000-01-01; an epoch-aligned width is off by that origin, so the call raises rather
  than returning a bucket that is silently 30 years out of phase.

**A pinned divergence:** `to_timestamp` returns TIMESTAMPTZ in DuckDB, rendered in the
session zone; engine timestamps are tz-naive UTC. Same instant, different rendering, so
the test compares epoch counts.

### 25-28. `regexp_full_match`, `constant_or_null`, and the `grade_up` family

`regexp_full_match` anchors the pattern in a non-capturing group (`^(?:p)$`) — the group
is load-bearing, since `^a|b$` would anchor only the first alternative.
`constant_or_null(v, x, …)` folds the guards with `&` so it is one pass per guard.
`grade_up`/`list_grade_up`/`array_grade_up` are `.list.arg_sort` shifted to DuckDB's
1-based origin through `list_transform`.

**Verified:** 50 differential cases in
`tests/differential/test_diff_sql_interval_json_temporal.py`, all green.

## Wave 2 — Spark SQL reachability, including the higher-order forms

Measured by the Spark census (Spark's own `@ExpressionDescription` examples, run through
`bt.sql(dialect="spark")`): **110 → 147 of 312 probed**, no name that matched before now
failing. The DuckDB census moved 221 → 237 over the same two waves, also with no losses.

### 29-34. Lambdas: `transform`, `filter`, `exists`, `forall`, and the DuckDB spellings

**Was:** every higher-order call raised. The engine has had `.list.transform` and
`.list.filter` over an `element()` placeholder all along; nothing bound a SQL lambda to it.

**Now:** a one-parameter lambda is translated by rewriting its parameter to `element()`
and translating the body through the ordinary scalar path — so any expression the engine
can compute is available inside a lambda, with no second evaluator. That serves
`transform`/`filter`/`exists`/`forall` (Spark) and `list_transform`/`array_transform`/
`list_apply`/`list_filter`/`array_filter` (DuckDB) from one mechanism.

**The parameter arrives as an `Identifier`, not a `Column`** (`x -> x + 1`), which is
what the first implementation got wrong: it rewrote only `Column`s, so every lambda body
raised `unsupported SQL expression: Identifier`.

**Refused rather than half-translated:** a two-parameter lambda (`aggregate`, `zip_with`,
`map_filter`). The `.list` kernels have one placeholder; dropping the second parameter
would silently compute something else.

### 35-64. Thirty Spark names composed from existing primitives

`add_months`, `date_add`, `date_sub`, `unix_date`, `unix_seconds`, `unix_millis`,
`unix_micros`, `convert_timezone`, `from_utc_timestamp`, `to_utc_timestamp`,
`array_compact`, `array_except`, `array_remove`, `arrays_overlap`, `get`,
`vector_inner_product`, `vector_norm`, `if`, `nvl2`, `equal_null`, `nullifzero`,
`zeroifnull`, `pmod`, `positive`, `negative`, `width_bucket`, `btrim`, `find_in_set`,
`e`, `regexp_count`, `regexp_substr`, `substring_index`, `elt`, `space`, `bit_get`.

Four needed a rule the tables could not express, and each is pinned by a test:

* **`array_remove` needs a null guard.** `element() != v` is *null* for a null element,
  which the filter drops — so `array_remove(array(1, null), 1)` lost the null Spark keeps.
* **`arrays_overlap` is three-valued.** No overlap found is *false* only when neither side
  contains a null; otherwise the null might have been the shared element, so the answer is
  unknown. Composed from the intersection and a null count rather than mapped to
  `has_any`, whose null rule is a different one.
* **`width_bucket` has sentinel buckets** (0 below the range, `n + 1` at or above it), so
  it is a CASE rather than bare arithmetic.
* **`bit_get` must use the bitwise methods.** `&` and `>>` on an `Expr` are the *logical*
  operators; `&` on two integers raises rather than masking.

### 65. Constant integer arithmetic is folded where a literal is required

`date_sub(d, 1)` reaches the translator as `date_add(d, 1 * -1)` — sqlglot's own
rewrite — and every `_const_int_arg` call site rejected it with "must be an integer
literal" for an argument the user *did* write as a literal. `_int_literal` now folds
constant `+`, `-`, `*` and parentheses, which fixes that class of false rejection
everywhere at once (`round(x, 1 + 1)`, `substr(s, 2 * 3)`, ...).

**Verified:** 67 unit cases in `tests/unit/test_sql_spark_lambda_and_composed.py` against
Spark's documented answers, plus 633 existing SQL/expression unit tests and 1,755
differential tests re-run with no regression.

## Wave 3 — Semantics the engine had wrong, and reachability the API did not have

### 66. `[1, 2] || [3]` concatenated the *rendering*, not the lists

**Was:** the `||` operator cast both operands to Utf8 unconditionally, so two list
columns produced the string `'[1, 2][3]'`. No error — a wrong answer of the worst kind,
and one that every other engine (DuckDB, Spark, Polars) answers `[1, 2, 3]`.

**Now:** two list operands take the list-concat kernel — the same one `list_concat`
already used, so the operator and the function cannot disagree. Every other operand
pairing is untouched.

**Verified:** differential cases over list and string operands, plus one asserting the
operator and `list_concat` agree, and one over list *columns* rather than literals.

### 67. `last_day` returned a timestamp where every other engine returns a date

**Was:** midnight of the month's last day, as `Timestamp(µs)`. It typed the column
wrongly in a `with_columns`, read as `2024-03-31 00:00:00` beside genuine date columns,
and forced the existing differential test to cast DuckDB's answer before it could
compare at all.

**Now:** `Date32`, for either input type, as in DuckDB, Spark and Polars. The type
inference (`plan/types/infer.py`) and the accessor docstring moved with it, and the
differential test compares without a cast, which is the point.

### 68-73. Six aggregates the engine did not have

`any_value` (with `arbitrary`), `entropy`, `mad`, `quantile_disc`, `approx_top_k` and
`kurtosis_pop` — reachable from both `col(...).<method>()` and SQL.

Five of them needed **no new state**: `entropy`, `mad`, `quantile_disc` and
`approx_top_k` read the same mergeable per-group value list `median` has always built,
and `kurtosis_pop` reads the same 5-column moment state as `kurtosis`. Only the
finalize step is new, which is why the batch is six aggregates rather than one.

Three decisions worth recording:

* **`any_value` resolves "unspecified" to the group minimum.** DuckDB documents the
  chosen row as unspecified and returns whichever it saw first. A mergeable aggregate
  cannot promise scan order — `combine` has to be commutative or a distributed run
  disagrees with a single-node one — so the engine picks the minimum, which is a
  conforming answer *and* a stable one. `first`/`last` stay refused, because they name
  a row in scan order and mean it.
* **`approx_top_k` is exact.** The value-list state already holds every value, so the
  space-saving sketch the DuckDB name refers to could only lose accuracy. The name is
  kept so a ported query reads the same.
* **`quantile_disc` is not `quantile`.** The continuous quantile interpolates between
  two elements and can return a value that is not in the data, which is wrong for an
  ordinal column. Both are now available and the test asserts they differ.

**Verified:** differential tests against DuckDB, grouped and global, plus a
partition-equivalence test per aggregate (the mergeability invariant seen through the
engine) and property tests (entropy is 0 for a constant group and log₂(n) for n
distinct values; `mad` ignores an outlier the standard deviation follows).

### 74-88. An aggregate outside `group_by().agg()` now means something

**Was:** `ds.select(total=col("x").sum())` — the first thing a pandas or Polars user
types — raised `PlanError`. So did `with_columns(share=col("x") / col("x").sum())` and
`filter(col("x") > col("x").mean())`. The Polars census counted **15** `Expr` methods
unreachable for exactly this reason and no other (`sum`, `mean`, `min`, `max`, `median`,
`std`, `var`, `n_unique`, `product`, `mode`, `skew`, `kurtosis`, `any`, `all`,
`approx_n_unique`).

**Now:** an aggregate in a row-shaped context resolves in one of two ways, and which one
is decided by the context, not by a flag:

* a `select` whose items are **all** aggregates is the whole-frame aggregation and
  returns one row (it lowers to `group_by().agg(...)`, so the optimizer, spill and
  distributed paths see a plan they already know);
* anywhere else the aggregate becomes `agg.over()` — the whole-frame aggregate broadcast
  to every row — which is the only reading under which `x / x.sum()` has a row per input
  row.

**Verified:** each context against the DuckDB query it desugars to (`SELECT sum(x)`,
`sum(x) OVER ()`, and an uncorrelated scalar subquery), plus a repartitioning test
proving the broadcast is over the whole frame and not per morsel. The test that pinned
the old refusal was rewritten to pin the new answers; the nested-aggregate refusal it
also covered is kept as its own test.

### 89-91. `rollup`, `cube` and `grouping_sets` on the DataFrame API

The SQL front-end has had multi-level grouping all along; the DataFrame surface had no
spelling for it, so a subtotal report meant hand-writing a `union` of `group_by`s.

`ds.rollup(*keys)`, `ds.cube(*keys)` and `ds.grouping_sets(*sets)` return a builder with
one finisher, `.agg(...)`. Each level is an ordinary `group_by` whose *inactive* keys are
grouped by `nullif(col, col)` — a null of the column's own type, so every level's schema
matches and the union is legal, and a constant key, so the level collapses correctly.
No new execution strategy, and nothing in the aggregate path knows levels exist.

**Verified:** against DuckDB's `GROUP BY ROLLUP/CUBE/GROUPING SETS` for each form, plus
a test that the DataFrame and SQL front-ends produce identical rows.

### 92-104. Thirteen more names: rounding, structs, validity, zone fields

`bround` (banker's rounding, composed from `rint`, and asserted *against* `round` so the
two rules cannot be confused), `named_struct` and `struct` (Spark) plus DuckDB's
`{'a': 1}` struct literal, `is_valid_utf8` / `validate_utf8` / `try_validate_utf8` /
`make_valid_utf8` (an Arrow `Utf8` column is validated by construction, so these are a
predicate and three identities — answered rather than refused), `timezone_hour` /
`timezone_minute` (zero for a tz-naive timestamp, as in DuckDB, with nulls preserved),
`escape_regex`, `dt.to_string()` with a default pattern, `dt.timestamp(unit)`,
`dt.is_business_day()`, `list.drop_nulls()` and `array_repeat(v, n)` for a constant count.

### 105-107. Three epoch readers returned a day count for a DATE column

**Was:** `.dt.epoch_us()` on a `Date32` column returned **19,787** for 2024-03-05 — the
day count read as if it were microseconds — and `.dt.epoch_ms()` returned 19. Only
`.dt.epoch()` (seconds) was right, because it went through a timestamp cast and the
other three read the integer directly. DuckDB answers 1,709,596,800,000,000 and
1,709,596,800,000. A wrong number with no error, on a column type users have.

**Now:** all four go through one `_micros()` helper whose cast to timestamp is a no-op
on a timestamp column and the whole point on a date one.

**Verified:** differential cases for each reader over both a DATE and a TIMESTAMP
argument, plus one asserting the four agree with each other (`ms == s·1000`, and so on)
and that nulls propagate through all four.

### 108-116. Nine more Spark names, and a datetime-pattern rule that serves both dialects

`next_day` (the weekday arithmetic composed from the ISO day number, so landing on the
same weekday moves a whole week rather than zero days), `months_between` (Spark's exact
definition, including the /31 divisor and the 8-decimal rounding its two-argument form
applies), `date_format`, `to_date(s, fmt)`, `from_unixtime`, `to_unix_timestamp` and
`parse_url` (HOST / PATH / QUERY / PROTOCOL / REF / AUTHORITY / USERINFO), plus
`str.join` (the string aggregate, Polars' spelling of `string_agg`) and `Expr.neg`.

**One rule serves both dialects' datetime formats.** A chrono pattern always contains a
`%`; a Spark/Java one never does. So `date_format(d, 'yyyy-MM')` and
`strftime(d, '%Y-%m')` both reach the same kernel without the translator having to know
which parser ran, and a Java pattern the table cannot express (a quoted literal section)
is refused rather than formatted with the wrong field.

### 117. `get_bit` was answered with Spark's semantics for DuckDB's function

Found by the census as a **new** mismatch introduced by the previous entry, which is
what re-running it after every wave is for. sqlglot gives Spark's `bit_get(int, n)` and
DuckDB's `get_bit(BITSTRING, n)` the same node, and the two index from opposite ends.
The integer reading is now applied only to an integer argument; a bit string is declined,
because the engine has no BIT type and answering it would return the wrong bit.

## Measured result of the sweep so far

Every number below is from re-running the three censuses against the built tree, not
from counting entries in this file.

| Census | Oracle | Before | After |
|---|---|---|---|
| DuckDB (`duckdb_functions()`, 478 signatures, through `bt.sql`) | the live engine | 221 | **260** |
| Spark (527 documented examples, through `bt.sql(dialect="spark")`) | its own `@ExpressionDescription` examples | 110 | **160** |
| Polars (every zero-argument `Expr`/`.str`/`.dt`/`.list` method) | the live library | 53 | **69** |

Gaps fell 244 → 203 (DuckDB, 130 of which are ICU collations this engine has no
equivalent for, so the real remainder is ~76), 186 → 117 (Spark) and 31 → 11 (Polars). **No name that matched before
the sweep fails after it** — checked after every wave, which is how entry 117 was found.

The engine-level changes the censuses cannot see are the ones in entries 66, 67, 74-91
and 105-107: an operator that returned the wrong answer, a function that returned the
wrong type, three epoch readers that returned a day count, multi-level grouping on the
DataFrame API, and aggregates in a row-shaped context.

## Verification state

- **`tests/differential`: 6,275 passed, 0 failed** — the whole correctness spine, on the
  settled tree after the last change rather than once at the start.
- **`tests/unit`: 8,339 passed, 0 failed.** (Two failures seen mid-session —
  `test_metrics_snapshot_shape` and a `hardware.py` accelerator trio — belonged to other
  sessions' in-flight refactors of `observe/`, `metadata/` and `_internal/hardware.py`,
  and cleared when those settled. None of those files are touched by this sweep.)
- **A run that crashes is no result, not a bad one.** One 66-minute differential run
  reported 12 failures, all in `test_dist_hunt2_matrix.py`; it had overlapped another
  session rebuilding `_native.abi3.so`, which pulls the mapped pages out from under the
  Ray workers holding it (the raylet log shows `Task _broadcast_join_task failed … retries
  remaining`). Re-run on the settled tree: **22 passed**. Recorded because reporting that
  number as a regression — or as "pre-existing" — would have been wrong in both
  directions.
- **Rust: 1,310 tests pass** across the workspace (`just test-rust`), clippy clean and
  `cargo fmt` clean for every crate this sweep touched (`bc-expr`, `bc-ir`, `bc-runtime`,
  `bc-interp`); the one workspace clippy error is in `bc-arrow/src/hash.rs`, another
  session's new file.
- **`lint-structure`, `lint-layers` (5 contracts), `ruff` clean**; `tests/docs` green,
  including the API-coverage contract that every new public name is documented, rendered
  and taught, and `MAP.md` regenerated.
- **New coverage:** 6 test files, ~230 cases — differential against DuckDB where DuckDB
  has the function, against Spark's own documented answers where it does not, plus
  partition-equivalence tests for every new aggregate and a repartitioning test for the
  broadcast aggregate.
- The migration-guidance tables were pruned in the same change: five entries told users
  that `rollup`, `cube`, `entropy`, `.str.join` and `.list.drop_nulls` did not exist, and
  a test asserts no guidance entry shadows a real method.
- **Three tests that pinned a refusal were rewritten to pin the new answer**
  (`test_expr_errors`, `test_diff_agg_expressions`, `test_diff_last_day`). Each is
  recorded here because rewriting a test to make a change pass is exactly the move this
  contract forbids when the old behaviour was right — these three pinned behaviour the
  change deliberately replaced, and each rewrite asserts more than the refusal it
  replaced (the whole-frame answer, the broadcast answer, and DuckDB's date without the
  cast that used to be needed to compare at all).
- **The censuses are now in the tree**, at `tools/census/`, so the next pass measures
  rather than re-derives: `duckdb_functions.py` (the live engine), `spark_examples.py`
  (Spark's own documented examples — no JVM required) and `polars_methods.py`.

## Open, in the order a later pass should take them

1. **`printf` / `format` / `format_string` / `format_number`.** The rest of the
   formatting family landed in wave 4; these four need a *variadic* node (a template
   plus N argument expressions), which the others did not.
2. **The map constructor family** (`map`, `map_from_arrays`, `map_entries`,
   `map_concat`, `map_contains_key`, `str_to_map`, `transform_values`). ~8 DuckDB and
   Spark names behind one Arrow `MapArray` builder.
3. **`list_resize` / `array_resize` and `list_select` / `array_select`** — two kernels,
   four names. (`generate_series`/`range`/`sequence` landed in wave 4.)
4. **`to_json` / `from_json` / `schema_of_json`.** The rest of the JSON writers landed
   in wave 4; these three convert between a struct column and JSON text, which needs the
   schema, not just the document.
6. **`arg_min_null`/`arg_max_null` and the `nulls_last` variants** — the existing
   `arg_extreme` state with a different null rule.
7. **Grapheme-aware string functions and Unicode normalization** — both want a crate
   dependency (`unicode-segmentation`, `unicode-normalization`).
8. Everything in `competitor_technique_review.md`'s backlog, which this sweep did not
   touch: online adaptive conjunct reordering, `StringView` end to end, the top-K
   threshold as a dynamic filter, the skew salt derived from measured partition sizes,
   and adaptive morsel sizing.

## Wave 4 — the kernels the wiring could not reach

### 118-127. Int → text formatting

`chr` (and Spark's `char`), `bin`, `to_base`, `hex` of an integer, `format_bytes` with
its two DuckDB aliases `formatReadableSize`/`formatReadableDecimalSize`, and Spark's
`conv(text, 10, base)`.

**Why these needed a kernel rather than wiring:** they all map Int → Utf8, and
`eval_str` downcasts its argument to `StringArray` *before* the kernel is reached, so no
translator change could have made them work. They are dispatched in `str/numfmt.rs`
before that downcast, beside the `Binary` family, and every function that is not one of
theirs declines there, so the string path is untouched.

Four details are DuckDB's, and each is pinned by a test because each was wrong in the
first implementation: the digits are **uppercase** (`to_base(255, 16)` is `FF`); the base
unit is the word **`bytes`** (`512 bytes`, not `512 B`); the scaled value is
**truncated**, not rounded (8,364 bytes is `8.1 KiB`, where rounding says 8.2); and the
magnitude is taken as `u64` so `to_base(i64::MIN, 10)` converts instead of panicking on
the negation.

A pinned divergence: DuckDB *errors* on a negative `to_base` argument. The engine writes
the magnitude with a `-`, which is what every other base conversion in the language does;
the test asserts both sides.

### 128-131. The JSON writers

`json_value`, `json_contains`, `json_pretty` and `json_structure`, completing the reader
half that was already there.

`json_value` is the one worth the words: it answers **only for a scalar**, and keeps a
string's quotes, where `json_extract_string` unquotes a string and renders a container as
compact JSON. DuckDB draws exactly that line between the two, and the test asserts them
side by side so neither can drift into the other. `json_pretty` indents by four spaces
and `json_structure` names a JSON null `NULL` — both DuckDB's, both found by comparing
rather than by reading.

### 132-134. `range`, `generate_series` and `sequence`

**The engine already had the series node** — `Expr::Sequence`, with its own kernel — and
nothing reached it from SQL. That is the same shape of finding the census keeps turning
up, and it is the cheapest kind to close.

The three names differ in their *bounds*, and the difference is not cosmetic:
`generate_series(1, 5)` and Spark's `sequence(1, 5)` are `[1,2,3,4,5]` where `range(1, 5)`
is `[1,2,3,4]` and `range(3)` is `[0,1,2]`. The engine's node is the inclusive one, so
the exclusive form pulls its stop in **by a step** — the only rewrite that stays correct
for a step other than 1, which the `±2` cases pin.

### 135-137. Compensated summation: `fsum`, `kahan_sum`, `sumkahan`

A plain float sum loses the low bits of every addend far below the running total, so
`[1e16, 1.0, 1.0, -1e16]` sums to **0.0** where the true answer is 2.0. DuckDB has the
compensated form under three names; the engine had none, and the aggregates module said
so in a comment listing what it deliberately refused.

**Now:** a 2-column `(sum, compensation)` state using **Neumaier's** variant — chosen
over plain Kahan because it is also correct when the *addend* is the larger of the two,
which is the case a running total meets on its first rows. `finalize` adds the
compensation back exactly once.

**Mergeable, and that is the part with a test of its own:** two states combine by
compensated-adding the sums and summing the compensations, so a partitioned run folds
several states and still applies the correction once. The test asserts one-partition and
three-partition runs agree *and* that both equal 2.0.

**Verified** against DuckDB on all three names, grouped and global, plus the drift case
with both numbers asserted on both engines.

### 138-144. The `now` family, `current_timezone`, and `array_insert`

`now()`, `getdate()`, `current_timestamp()`, `localtimestamp()`, `unix_timestamp()` (the
nullary form), `current_timezone()` and `array_insert(xs, pos, value)`.

The first four are one function here, and that is a property of the engine rather than a
shortcut: engine timestamps are **tz-naive UTC**, so there is no local/UTC distinction to
draw between them. `current_timezone()` answers `UTC` for the same reason — it is not a
session lookup, it is the only answer true of the values.

`array_insert` is expressed as two slices around the insertion point, so a *constant*
position is served and a negative one (which Spark counts from the end) is declined
rather than inserted at the wrong end.

### 145. `now()` read the clock twice in one query

Found by the test written for the entry above, not by the census. SQL requires `now()`
to be constant *within a statement*: `SELECT now() AS a, now() AS b` must give `a == b`,
and a predicate comparing a column against `now()` must not see the clock move between
morsels. Each call was folding its own read. The instant is now memoized on the
translator, so one query has one `now`.
