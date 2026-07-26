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
overload returns DuckDB's answer. Before this wave: **93 of 478**. After this entry:
**193 of 478**; after the later entries in this file, **215 of 478** — with **no function
that previously matched now failing**, checked after every wave.

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

### 107. `regexp_extract_all` ignored its capture-group argument

**Source:** Spark's own `@ExpressionDescription` examples, confirmed against DuckDB.

Spark annotates every builtin with an `examples` block holding `> SELECT _FUNC_(args);`
lines and the expected output, and `FunctionRegistry.scala` maps each expression class to
its SQL name. Together they are an executable oracle needing no JVM — which matters here,
because there is no Java runtime on this machine. Running them through
`bt.sql(dialect="spark")` reported `regexp_extract_all('100-200, 300-400', '(\d+)-(\d+)',
1)` as `['100-200', '300-400']` against Spark's documented `['100','300']`; DuckDB agrees
with Spark.

The group index had nowhere to go. The kernel called `find_iter` unconditionally, and
`.str.regexp_extract_all` took only a pattern — so the SQL translator, which had just
gained the function in entry 1-100, silently dropped the third argument. Both halves are
fixed: the index rides the same `start` field the scalar `regexp_extract` already uses
for it (no IR tag changed), and the kernel switches to `captures_iter`.

Three details are DuckDB's, and each needed checking rather than assuming: a group that
did not participate in its match is a **NULL element** (where the scalar
`regexp_extract` yields `''` for the same case), a group index past the pattern's count
is an **error** rather than an empty list, and group 0 stays the whole match.

Verified by a Rust kernel test, six differential cases through both the DataFrame and SQL
paths, and a case pinning the null-element behaviour.

### 108. Every 1-based array subscript was off by one outside the DuckDB dialect

**Source:** the same Spark example run — `element_at(array(1, 2, 3), 2)` answered `3`.

sqlglot rewrites `l[2]` to a 0-based index for dialects whose subscript is 1-based
(duckdb, postgres) and leaves `Bracket.offset` unset. Where it cannot rewrite — Spark's
`element_at(a, 2)` is 1-based while Spark's own `a[2]` is 0-based, so the two cannot share
a normalization — it keeps the written index and records the base in `offset`. The
translator ignored `offset` and treated every subscript as already 0-based, returning the
*next* element.

This is the shape of bug the census exists to find: it is invisible in the dialect the
test suite uses, silent in the dialect it breaks, and off by exactly one, so a small
example looks plausible.

### 109-125. Seventeen scalar functions DuckDB has and the engine did not

**Source:** the census gap column, filtered to the functions whose argument and result
shapes the existing `Expr`/`StrFunc` nodes can already carry.

**Math** (`bc-expr` `MathFunc`/`Math2Func`, five unary and one binary): `even`, `gamma`,
`lgamma`, `sec`, `csc`, `rint`, and `nextafter`. Three of them exist because a rounding
rule is not one thing:

* `round` is half **away from zero** (DuckDB's rule, and the engine's).
* `rint` is half **to even** — IEEE-754's own rule, so a column of rounded values does not
  drift upward the way half-up rounding does. Spark has it; DuckDB does not.
* `even` rounds **outward** to an even integer, so `3.0` becomes `4.0`, which is neither
  of the above.

`gamma`/`lgamma` go through `libm`, the pure-Rust port of the FDLIBM routines DuckDB
reaches through its libc, rather than a series approximation — so they agree to the last
bit rather than to a tolerance. (One input, `gamma(0.5)`, still differs by 1 ULP; the
differential harness rounds to 9 places, so it is inside the tolerance either way.)
`nextafter` steps the two's-complement-ordered bit pattern, and is *not* lowered by the
JIT for the reason `cbrt` is not: a libm libcall need not reproduce the interpreter's
answer bit-for-bit on the subnormal and sign-crossing cases, and the contract says the JIT
must then fall back.

**String** (a new `eval/str/uri_path.rs`, eleven functions): `url_encode`, `url_decode`,
`regexp_escape`, `parse_filename`, `parse_dirname`, `parse_dirpath`, `parse_path`,
`to_binary`, `from_binary`, `hamming` (also spelled `mismatches`) and `jaccard`.

They are one module because they share a property worth naming: each is defined by an
*external specification* rather than by an operation on characters, so the interesting
part of every one is an edge case that has to be read off DuckDB rather than derived.
Four were, and each would have been wrong otherwise:

* `url_decode('a%2')` is `'a%2'` — a truncated escape passes through. It does not raise
  and does not null the row.
* `parse_dirname` is the **first** path component (`/` for an absolute path); the
  directory holding the file is `parse_dirpath`. They disagree on every path deeper than
  one level, and both exist in DuckDB.
* `parse_dirpath('/')` is `/` while `parse_dirpath('/single')` is `''` — the root is its
  own directory, so the carve-out is the one-character path, not "starts with a
  separator". The first implementation got this wrong and the differential test caught it.
* `regexp_escape` is RE2's `QuoteMeta`, which backslashes **every** ASCII byte outside
  `[A-Za-z0-9_]` — not the `regex` crate's `escape`, which escapes only its own
  metacharacters and returned `'a b'` where DuckDB returns `'a\ b'`. Both are patterns
  that match the literal, so the difference is invisible if you match with it
  immediately; the returned *string* is what a user stores and compares. (Escaping the
  wider set is safe for the engine's own matcher, which accepts a backslash before any
  punctuation — verified, not assumed.)

`hamming` raises on unequal lengths rather than comparing a prefix, as DuckDB does,
because a prefix comparison answers a caller's mistake with a plausible number.

**Verified** by 152 differential cases against DuckDB (columns, not literals), two Rust
kernel tests including a round-trip property over Unicode and empty input, and a test that
each escaped value matches itself through the engine's own matcher.

### 126. A date function on a text column worked for half the family and raised for the rest

**Source:** Spark's documented examples — twelve of them are `year('2016-07-30')`-shaped,
and Spark parses the string as a date.

`eval_date` handles twenty-one functions. Some of them (`dayname`, `monthname`,
`last_day`, `is_leap_year`, `days_in_month`, `iso_year`) cast the input to
`Timestamp(µs)` *as a side effect of how they compute*, and so accepted a text column.
The rest (`year`, `month`, `day`, `hour`, `second`, `quarter`, …) hand the array straight
to Arrow's `date_part` kernel and failed with `Year does not support: Utf8`. Which half a
function fell in was an accident of implementation, not a decision — and the ones that
failed are the common ones.

The cast is now hoisted to the top of `eval_date`, so the family agrees. This accepts
*more* than DuckDB, which rejects the string form outright with a binder error, rather
than answering differently from it — the only direction a compatibility convenience may
go when DuckDB is the oracle.

### 127-151. Twenty-five Spark SQL names reachable from `bt.sql(dialect="spark")`

Typed nodes (`Sec`, `Csc`, `Rint`, `BitwiseCount`, `Flatten`, `ArraySort`,
`ArrayToString`, `ArrayPosition`, `ArraySlice`, `TsOrDsToDate`) and anonymous names
(`hypot`, `nanvl`, `isnull`, `isnotnull`, `log1p`, `expm1`, `xxhash64`, `map_values`,
`map_keys`, `make_date`, `try_mod`, `try_to_binary`, `try_url_decode`), plus DuckDB's
`list_slice`/`array_slice`, which the census had not reached either.

Spark's census score went from 75 of 312 probed to 108, with no name that previously
matched now failing.

Two of these were nearly wrong answers, and both are about a second operand:

* `slice(l, start, length)` is 1-based with a *length*; `.list.slice(offset, length)` is
  0-based. sqlglot names the two operands `start` and `end`, which invites reading the
  second as an index. Getting either wrong returns a plausible window one element along —
  the first implementation did, and Spark's own documented example caught it.
* `list_slice(l, begin, end)` is DuckDB's, and its bounds are an inclusive 1-based pair,
  *not* Spark's start-and-count. The two spellings cannot share a translation.

And one was refused rather than mapped: Spark's `to_binary(s, charset)` encodes bytes
while DuckDB's `to_binary(s)` is a `0`/`1` bit string. Same name, different function. The
two-argument form silently returned the bit string; it now raises.

### 152-160. Nine window aggregates beyond `sum`/`avg`/`min`/`max`/`count`

**Source:** the capability probe, and the open list's own item 2. DuckDB, Spark and Polars
all allow *any* aggregate over a window; this engine allowed five, and
`col("x").std().over("g")` raised `unknown window function 'stddev'`.

**Now** `var`, `stddev`, `product`, `bool_and`, `bool_or`, `bit_and`, `bit_or`, `bit_xor`
and `count_distinct` work in both window shapes, in a new
`bc-runtime/src/window_agg.rs`.

**The selection rule is the point.** These nine are exactly the aggregates whose *running*
form costs **O(1) per row**: the moment pair from a running `(n, Σx, Σx²)`, the six folds
from a running application of an associative operator, and `count_distinct` from a running
hash set. Order statistics (`median`, `quantile`, `mode`) are deliberately absent —
their running form needs a sorted structure, and adding them here would put an `O(n log n)`
kernel behind the same call shape as an `O(n)` one with nothing at the call site to say so.
`var_pop`/`stddev_pop` are absent for a different reason worth separating: the engine's
aggregate vocabulary has no tag for them (the DataFrame spellings are composites over
`var`/`stddev`), so a `WindowFn` variant would have had nothing able to construct it —
speculative generality rather than a capability.

**`var`/`stddev` keep the same sum-of-powers state `agg/var.rs` keeps**, so the window and
the `GROUP BY` over the same rows agree by construction rather than by coincidence — and a
test asserts that equality so "by construction" is checkable.

`WINDOW_FRAMEABLE` deliberately did *not* grow with `WINDOW_AGGREGATES`: the framed path
has a hand-written sliding kernel per function and has none for these, so an explicit
`ROWS` frame is rejected at plan time rather than silently ignored.

**Verified** by 30 differential cases against DuckDB covering the whole-partition,
running and unpartitioned shapes over groups of different sizes, a singleton group (where
sample variance is NULL), nulls in every value column, and repeated values so
`count_distinct` is not the row count. The comparison is **ordered**, not the suite's
usual multiset: a running aggregate is a sequence, and an order-independent comparison
cannot see a running kernel that emits the right values in the wrong order.

### 161-165. `list_concat`, `list_has_all`, `list_has_any` and the two negative products

**Source:** the census gap column, the list family.

Three DuckDB list functions the engine did not have, plus SQL access to
`list_negative_dot_product`/`list_negative_inner_product` (the sign-flipped dot product a
maximum-inner-product search minimizes, composed from the existing `.list.dot`).

What separates all three from the set operations sitting next to them is the same thing,
and it is **not** the operation — it is how a null list and an empty list behave:

* `list_concat` keeps duplicates *and* treats a null list as **empty**, so
  `list_concat(NULL, [1])` is `[1]` where `list_union(NULL, [1])` is NULL. That rule holds
  all the way down: `list_concat(NULL, NULL)` is `[]`, so the kernel never produces a null
  row at all. The first implementation nulled the both-null case; the differential test's
  both-null row caught it.
* `list_has_all`/`list_has_any` are null when **either** side is null — even though
  `list_intersect([1,2], NULL)` is `[]` rather than NULL in *both* engines. That asymmetry
  is why they are composed to make both operands load-bearing (`has_all` compares the
  intersection's size to `other`'s distinct count rather than asking whether
  `other - self` is empty, which reads only `other` and so answered `False` where DuckDB
  answers null).

`Concat` rides the existing `ListSetOp` because its shape is the same — two lists in, one
list out — but it takes its own kernel, with a comment saying why the set machinery's
dedup and null rules do not apply.

**Verified** by 11 differential cases whose fixture crosses both operands over disjoint,
overlapping, contained, null and empty lists — every row is one of the edge cases above.

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
* **Spark's semantics are not adopted with Spark's syntax.** `dialect="spark"` selects a
  *parser*, not a semantics. Where Spark and DuckDB genuinely disagree the engine follows
  DuckDB, because DuckDB is the differential oracle the whole suite is written against:
  `regexp_replace` replaces the first match (Spark replaces all), `sort_array` orders
  nulls last (Spark first), `array_distinct` drops nulls (Spark keeps them), `dayname`
  spells the day in full (Spark abbreviates), `weekday` is Sunday-based (Spark's is
  Monday-based), and `round` is half-away-from-zero (Polars is half-to-even). A Spark query that relies on one of these will run and return DuckDB's
  answer, so each is listed in the `migrate-from-spark` skill rather than left to be
  discovered.
* **`round` half-away-from-zero.** Polars rounds half to even (`round(-2.5)` is `-2.0`);
  DuckDB and Batcher round half away from zero (`-3.0`). DuckDB is the oracle.
* **`list_sum` of an empty list** is `NULL` here and in DuckDB, `0` in Polars.

## The three censuses, and what each is good for

| Reference | Oracle | Surface probed | Result |
|---|---|---|---|
| DuckDB | the live engine (`duckdb` is a test dependency) | `duckdb_functions()`, 478 scalar/aggregate signatures, through `bt.sql` | **93 → 215** supported; 6 wrong answers found |
| Spark | its own `@ExpressionDescription` examples, parsed out of the Scala source | 527 documented examples over 438 registered names, through `bt.sql(dialect="spark")` | **75 → 108** supported; 4 wrong answers found (entries 107, 108, and the two in 127-151) |
| Polars | the live library | every zero-argument method on `pl.Expr` and its `.str`/`.dt`/`.list` namespaces | 53 supported, 31 absent, 14 differences (all but two are representation or a pinned semantic choice) |

Spark deserves a note: **it needs no JVM.** There is no Java runtime on this machine, so
`SparkSession` cannot start — but every builtin carries its expected output in an
annotation next to its implementation, which is an oracle a text reader can use. That is
worth remembering the next time a reference engine will not run.

## Open

Ordered by measured value, from the same censuses and a capability probe of the DataFrame
surface.

1. **`last_day(DATE)` returns a `TIMESTAMP`.** An engine-level type bug, not a translator
   one: `.dt.last_day()` widens a `DATE` input to a timestamp. The only mismatch in the
   census that is a defect in the data plane rather than a representation choice.
2. **The window aggregates entries 152-160 did not take.** `median`, `quantile` and
   `mode` need an order-statistic structure for their running form; `arg_min`/`arg_max`
   and the two-input aggregates need a second input, which `WindowCall` has nowhere to
   put; `array_agg` accumulates O(n) state per row in the running form. Separately, none
   of the nine that landed honours an explicit `ROWS` frame — the framed path has a
   hand-written sliding kernel per function. That one **generalizes**: its `FifoSum` is
   the two-stack sliding-window trick, which works for *any* associative operator, so one
   generalization would give every fold a framed form at once.
3. **Aggregates the engine does not have at all**: `entropy`, `mad`, `approx_top_k`,
   `bitstring_agg`, `histogram_exact`, `quantile_disc`, the `arg_*_null` variants, and a
   true `any_value`/`first`/`last` with no ordering requirement.
4. **`struct` and `map` namespaces.** `.struct` has one method (`field`); DuckDB has
   `struct_insert`/`struct_keys`/`struct_values`, Polars has more. `.map` has three
   against DuckDB's eleven.
5. **`ROLLUP`/`CUBE`/`GROUPING SETS` on the DataFrame API.** The SQL front-end has them
   (`_sql/parser/grouping_sets.py`); `ds.rollup(...)` does not exist.
6. **Scalar gaps with no engine implementation.** The ones entries 109-125 did not take,
   and why: `chr`, `bin`, `to_base`, `format_bytes` and `bar` all map **Int → Utf8**, and
   `StrFunc` requires a Utf8 input — the caller downcasts to `StringArray` before the
   kernel is reached — so they need either a new `Expr` variant or a widening of that
   dispatch, which is a design decision rather than a kernel. Then `uuid`/`random`
   (nondeterministic, so they collide with seq == par == JIT unless seeded through the
   plan), `time_bucket`, `age`, `list_reduce`, `list_zip`, `generate_series`,
   `equi_width_bins`, and `nfc_normalize`/`strip_accents` (which want a Unicode
   normalization dependency).
