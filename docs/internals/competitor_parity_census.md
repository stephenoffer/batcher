# Competitor parity census: measured gaps and what closed them

**Status:** running record, opened 2026-07-26.

`competitor_technique_review.md` lists the *mechanisms* the reference engines have that
Batcher does not, read from their source. This file is the other half: the gaps found by
**executing** the reference engine and Batcher side by side over their whole surface, and
the build log against what that measurement found.

The reference sources are the vendored checkouts in `/mnt/shared_storage/ref`: DuckDB,
Spark, Polars, DataFusion, Daft and Arrow.

## Why a census rather than a reading

Reading a competitor's source tells you what it *has*. It does not tell you what Batcher
*answers*, which is the only thing a user experiences. Twice in this pass, a capability
the engine demonstrably had was unreachable from the surface a migrating user would type,
and a third time a name resolved to a function with different semantics and returned a
plausible wrong answer. None of those are visible from a reading of either codebase.

So the method is: enumerate the reference engine's own function catalogue, synthesize a
call per signature, run it in both engines, and sort the results into *matches*, *gaps*
(the reference answered, Batcher raised) and **mismatches** (both answered, differently).
The mismatches are the valuable column — a gap is an honest refusal, a mismatch is a
wrong answer nobody has noticed.

**Counting rule** (carried from `engine_improvements_ledger.md`). An entry is a change to
behaviour that is verified by a committed test. A refactor, a doc, or an unproven change
is not an entry. The rule exists because the temptation on a "make N improvements" task is
to inflate the count, and an inflated ledger is worse than a short one.

## Landed

### 1-100. The DuckDB function surface reachable from `bt.sql`

**Source:** `duckdb_functions()` — DuckDB's own catalogue of every scalar and aggregate
function, 478 of which have a signature this census can synthesize an argument for.

**Method.** For each function, every overload is called in DuckDB and then in
`bt.sql(...)` with the same literal arguments; a function counts as supported when *any*
overload returns DuckDB's answer. Before this wave: **93 of 478**. After: **193 of 478**,
with **no function that previously matched now failing**.

**Was.** Three distinct failures, none of which is visible from the DataFrame API:

* sqlglot promotes the SQL functions it models to typed nodes, and the translator
  dispatched a fixed list of them by class name. Every other typed node raised
  `unsupported SQL expression: Cot` — even though `Expr.cot` had existed all along. Same
  for `Acosh`, `Sinh`, `Factorial`, `MD5`, `SHA`, `Hex`, `Unhex`, `BitLength`, `Unicode`,
  `Levenshtein`, `Split`, `Translate`, `RegexpExtractAll`, `TimeToStr`, `TimeToUnix`,
  `Atan2`, `DateFromParts`, `ArrayIntersect`, `Dayname`, `Monthname`, `LastDay`,
  `WeekOfYear` and `DayOfWeekIso`.
* Everything sqlglot does *not* model arrives as `exp.Anonymous` carrying only a name, and
  was rejected outright: `gcd`, `lcm`, `bit_count`, `century`, `decade`, `millennium`,
  `epoch_ns`, `epoch_us`, `weekday`, `isodow`, `isoyear`, `ord`, `strlen`, `base64`,
  `octet_length`, `damerau_levenshtein`, `jaro_similarity`, `str_split`, `string_split`,
  `string_to_array`, `regexp_split_to_array`, `list_extract`/`array_extract`/
  `list_element`, `list_intersect`, `list_pack`, and DuckDB's function spellings of the
  arithmetic operators (`add`, `subtract`, `multiply`, `divide`).
* **The aggregates were worse than unmapped — they were invisible.** Aggregate collection
  walks `find_all(exp.AggFunc)`, and `product(x)`, `mean(x)`, `favg(x)`, `histogram(x)`,
  `sem(x)` and `count_star()` are not `AggFunc` subclasses. So `SELECT product(x) FROM t`
  was not recognized as a grouped query at all; it reached the *scalar* translator, which
  reported an unknown function. Separately, `_AGG_FUNCS` is a name→tag table with nowhere
  to put a second input or a composite, so the whole two-input family (`corr`,
  `covar_pop`, `covar_samp`, `arg_min`/`arg_max` and their `min_by`/`max_by` spellings,
  and the nine `regr_*` functions) and the composites (`stddev_pop`, `var_pop`, `sem`)
  were unreachable regardless.

**Now.** Three additions, split by the thing that decides how a call is built:

* `_sql/parser/expressions/anonymous.py` — one table per *argument shape* for the
  anonymous names. The shape is what gets a call wrong (a constant string, an expression,
  a 1-based index), so it is a property of the table an entry lives in rather than
  something re-derived from the arity at the call site.
* `_sql/parser/expressions/aggregates.py` — the two-input and composite aggregates, plus
  `is_agg_node`/`iter_agg_nodes`, the widened collection predicate. The composites lower
  to an `Expr` over aggregate leaves, which `GroupBy.agg` already hoists into hidden
  columns and re-evaluates in a following projection, so no new plan machinery was needed.
* Rows in the existing typed-node tables (`_UNARY_MATH`, `_UNARY_STR`, `_DATE_PART`) and
  a `_STR_CONST_ARG` table for the `f(value, constant-string)` shape.

**Safe because** only names whose result is *bit-identical* to DuckDB's are listed. Four
were deliberately left raising rather than mapped to their nearest neighbour:
`first`/`last`/`arbitrary`/`any_value` (DuckDB returns an unspecified row's value; the
engine's `first`/`last` require an explicit ordering) and `fsum`/`kahan_sum` (compensated
summation, which the engine's `sum` is not). Each would have returned a plausible answer
that is not DuckDB's — the failure mode this wave was opened to fix.

**Measured** by the census itself, and pinned by 117 committed differential cases in
`tests/differential/test_diff_sql_duckdb_function_parity.py`. The arguments there are
*columns*, not literals, on purpose: a constant-only query can be answered by the
optimizer's constant folding without the runtime kernel ever running, so a
literal-argument test would pass while the engine path stayed broken.

### 101. `list_unique` returned the distinct list where DuckDB returns the distinct count

**Source:** the census mismatch column.

DuckDB's `list_unique(l)` is the **count** of distinct elements; the distinct *elements*
are `list_distinct(l)`. Both were mapped to `.list.unique()`, so `list_unique` returned a
list where DuckDB returns an integer — a wrong answer, silently, for anyone porting a
query that used it. Now `list_unique`/`array_unique` lower to `.list.n_unique()` and
`list_distinct`/`array_distinct` to `.list.unique()`.

### 102. `list_reverse_sort` sorted ascending

**Source:** the census mismatch column.

`list_sort(l)` and `list_reverse_sort(l)` both parse as sqlglot's `SortArray`,
distinguished only by an `asc=False` argument. The translator mapped the node type to the
method name `sort` and dropped the direction, so `list_reverse_sort([1,2,3])` returned
`[1,2,3]` where DuckDB returns `[3,2,1]`. The direction is now honoured; `SortArray` was
removed from the name-keyed `_LIST_REDUCE` table, because a table of names structurally
cannot carry it.

### 103. `sha2(s, 512)` silently returned sha256

**Source:** found while writing the test for entry 1-100 — the wave's own new code.

`sha256(s)` parses as `SHA2(this=s, length=256)`, so `SHA2` was added to the unary-string
table as `sha256`. That table is consulted first, so `sha2(s, 512)` — a *different* digest
— also matched it and returned a sha256 hash under a sha512 name. `SHA2` is now handled by
a dedicated branch that reads the width and refuses anything but 256, and the table row
carries a comment saying why it must stay absent. Worth recording because it is the exact
failure this wave set out to prevent, reintroduced by the fix for it.

### 104. `weekday` was the ISO day number, not DuckDB's

DuckDB's `weekday(d)` is Sunday-based (`0`-`6`, an alias of `dayofweek`). The engine's
`.dt.weekday()` is the ISO Monday-based `1`-`7`. Mapping by name made Sunday `7` where
DuckDB says `0` — right for six days out of seven, which is how it would have survived a
casual test. Now mapped to `.dt.dayofweek()`.

### 105. `microsecond`/`millisecond`/`nanosecond` dropped the seconds component

DuckDB's `microsecond(ts)` is the microseconds *within the minute* — `8_123_456` for
`…:08.123456` — not the sub-second field (`123_456`) the identically-named `.dt` method
returns. All three are now built as `second() * scale + <field>()`. The same shape of
error was in `yearweek`, mapped to `week_of_year` (returning `10` instead of `202410`);
it is now `iso_year() * 100 + week_of_year()`.

### 106. `epoch(ts)` truncated to whole seconds

DuckDB's `epoch(ts)` is fractional seconds as a `DOUBLE` (`1709618828.123456`); the
mapping to `.dt.epoch()` truncated to the second. Now `epoch_us() / 1e6`.

## Divergences pinned rather than closed

Recorded because a later pass will otherwise rediscover them and "fix" one the wrong way.

* **`sem`.** DuckDB divides the *population* standard deviation; Batcher's `bt.sem`
  divides the *sample* one, which is the standard definition and what `pandas.Series.sem`
  and `scipy.stats.sem` compute. The SQL spelling stays consistent with the DataFrame
  spelling rather than tracking DuckDB. Pinned by
  `test_sem_uses_the_sample_stddev_where_duckdb_uses_the_population_one`, which asserts
  *both* numbers so neither side can drift silently.
* **Degenerate-group statistics.** `corr`, `regr_slope` and `regr_intercept` over a
  single-row group are `NaN` in DuckDB and `NULL` in Batcher.
* **Result representation.** `ceil`/`floor` over a `DECIMAL` return `DECIMAL` in DuckDB
  and `DOUBLE` here; `unhex` returns a `BLOB` there and `VARCHAR` here; `histogram`
  returns a `MAP` there and a list of pairs here; `list_distinct` returns its elements in
  an unspecified order in both, which the census reports as a mismatch and is not one.

## Open

Ordered by measured value, from the same census and a capability probe of the DataFrame
surface.

1. **`last_day(DATE)` returns a `TIMESTAMP`.** An engine-level type bug, not a translator
   one: `.dt.last_day()` widens a `DATE` input to a timestamp. The only mismatch in the
   census that is a defect in the data plane rather than a representation choice.
2. **Window aggregates beyond `{sum, avg, min, max, count}`.** DuckDB, Spark and Polars
   all allow every aggregate over a window; `stddev`, `var`, `median`, `product`,
   `bool_and`, `count_distinct`, `mode` and `arg_min`/`arg_max` all raise
   `unknown window function` here. The whole-partition, running and framed paths are three
   separate hand-written kernels, so this is the largest item on the list. The framed path
   generalizes: its `FifoSum` is the two-stack sliding-window trick, which works for any
   associative operator, so one generalization would serve every new aggregate at once.
3. **Aggregates the engine does not have at all**: `entropy`, `mad`, `approx_top_k`,
   `bitstring_agg`, `histogram_exact`, `quantile_disc`, the `arg_*_null` variants, and a
   true `any_value`/`first`/`last` with no ordering requirement.
4. **`struct` and `map` namespaces.** `.struct` has one method (`field`); DuckDB has
   `struct_insert`/`struct_keys`/`struct_values`, Polars has more. `.map` has three
   against DuckDB's eleven.
5. **`ROLLUP`/`CUBE`/`GROUPING SETS` on the DataFrame API.** The SQL front-end has them
   (`_sql/parser/grouping_sets.py`); `ds.rollup(...)` does not exist.
6. **Scalar gaps with no engine implementation**: `chr`, `bin`/`unbin`, `url_encode`/
   `url_decode`, `to_base`, `uuid`, `random`, `time_bucket`, `list_reduce`, `list_zip`,
   `generate_series`, `bar`, `format_bytes`, `age`, `equi_width_bins`.
