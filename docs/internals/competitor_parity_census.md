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

**`var`/`stddev` keep the same Welford `(n, mean, M2)` recurrence `agg/var.rs` keeps**, so
the window and the `GROUP BY` over the same rows agree by construction rather than by
coincidence — and a test asserts that equality so "by construction" is checkable.

That sentence was **false when this entry was first written**, and the correction is the
part worth keeping. The kernel shipped with a sum-of-powers state, `(n, Σx, Σx²)`, and
this entry claimed the group aggregate used one too. It does not, and has not since
exactly this case was fixed there: recovering the variance as `(Σx² − n·mean²)` subtracts
two nearly equal large numbers, so over `[1e9+1, 1e9+2, 1e9+3]` the window returned `0`
where the `GROUP BY` returned `1`. The test fixture (values 1 through 8) could not see it.
Two lessons, both cheap to state and expensive to relearn: a claim that two paths "agree
by construction" is only worth as much as the construction actually being shared, and a
numerical-stability defect needs a fixture with a large mean and a small spread — a
differential test over small integers will pass either way.

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

### 166-170. Four rewrites the reference optimizers have, and one they inspired

**Source:** DuckDB's `optimizer/rule/` (27 files) and Spark's
`optimizer/expressions.scala` (15 rules), probed against the plans Batcher actually
produces rather than read for names.

**Most of it is already here** — distributivity, de Morgan, LIKE despecialization to
`=`/`ends_with`/`contains`, `LIKE 'abc%'` straight to a range, IN-list intersection and
refinement, the sargable arithmetic peels, `x + 0`, `abs(abs(x))`. The probe is in this
file's history because the *negative* result is the expensive part to rediscover.

**Two apparent gaps were deliberate refusals the code already documents**, and "fixing"
either would have been a wrong-answer bug:

* `col + k < lit → col < lit - k` is **not** applied. The engine's arithmetic wraps, so
  reassociating an *ordered* comparison breaks monotonicity: at `col = INT64_MAX`,
  `col + 5 > 10` is false while `col > 5` is true. `sargable.py` says so; the equality
  forms are rewritten and the ordered ones are not.
* `x * 0 → 0` and `x - x → 0` are wrong for a null `x`.

**Four were real, and three of them are the same shape of defect**: a predicate the
optimizer *could* make sargable, spelled the way a DataFrame user writes it rather than
the way SQL does.

1. `c = v1 OR c = v2 OR …` → `c IN (…)` (DuckDB `contains_to_in_clause`). The direct win
   is one hash-set probe instead of *n* comparisons. The larger one is that this
   repository already carries **eight rules keyed on `InList`** — `prune_in_list_by_zonemap`,
   `prune_in_list_by_bloom`, `dedup_in_list`, `refine_in_list_by_equality`/`_comparison`/
   `_neq`, `intersect_in_lists`, `push_in_list_across_join_keys` — none of which can fire
   on the `OR` form. Producing the node they match turns all eight on.
2. `starts_with(col, 'abc')` and 3. `substr(col, 1, 3) = 'abc'` → the exact range
   `col >= 'abc' AND col < 'abd'` that `LIKE 'abc%'` has always been given. These are the
   *DataFrame* spellings of that predicate, so `.str.starts_with(...)` scanned every row
   group while the SQL form skipped them.
4. `len(col) = 0` → `col = ''`, unwrapping the column so zone maps, blooms and pushdown
   can see it at all.

Plus `(x + c1) + (y + c2)` → `(x + y) + (c1 + c2)` (Spark `ReorderAssociativeOperator`):
`fold_add_sub_constants` collapses a chain nested down one side but cannot see this one,
because neither operand is a bare literal, so the constants stay one addition apart
however many fixpoint passes run.

**Writing the first surfaced a regression it would otherwise have caused.**
`or_to_in_and_range` derives `min ≤ c ≤ max` zone-map bounds *by reading the `OR` chain*,
so consuming the chain first silently removed those bounds. It now reads the `InList` form
too, and a test asserts both survive together. The name had also misled: despite reading
`or_to_in`, it only ever added bounds — three other modules' comments deferred the
single-column `IN` fold to it, and it never performed one.

**Verified** by 40 cases (14 plan-shape, 26 differential). The differential tests project
each predicate as a *value* rather than only filtering on it: under a filter NULL and
FALSE both drop the row, so a filter-only test cannot tell a three-valued mistake from a
correct rewrite. The reassociation is asserted at `INT64_MAX`, where the ring argument is
the only thing keeping it exact.

### 171-176. Every fold gets an explicit frame, from one generalization

**Source:** this file's own open list, item 2 — "that one **generalizes**: its `FifoSum`
is the two-stack sliding-window trick, which works for *any* associative operator, so one
generalization would give every fold a framed form at once."

`product`, `bool_and`, `bool_or`, `bit_and`, `bit_or` and `bit_xor` now honour an explicit
`ROWS`/`GROUPS` frame. Six capabilities, one change: `window_frame::FifoSum` was the
two-stack "queue from two stacks" specialized to `+`, and nothing about the structure was
ever specific to addition. It is now `window_agg::SlidingFold`, generic over the combine.

**Why that structure and not the obvious slide** is the whole point, and is what the tests
pin. The naive O(1) update applies the entering value and *un-applies* the leaving one,
which needs an **inverse** — and these folds have none. `product` cannot divide a zero back
out; `bit_and` and `bool_and` cannot un-AND at all. The two-stack never un-applies: the
reported value is always the fold of exactly what is in the window, at O(1) amortized cost.
So the differential fixture contains a **zero** and a **false**, and asserts the window
*recovers* once it has slid past them — with a subtract-based slide every later row is
wrong, and no amount of ordinary test data shows it.

Commutativity is the one property the generalization needs beyond associativity, because
the two stacks' accumulators are combined without regard to which side is older. All six
are commutative; the doc comment says so rather than leaving it implicit.

`var`/`stddev` and `count_distinct` still refuse a frame, and the refusal is now for a
*stated* reason rather than an absent kernel: the moment pair keeps a Welford state whose
combine is Chan's parallel formula rather than an operator, and a distinct count needs a
multiset rather than a fold. A test asserts all three still raise.

**Verified** by 28 differential cases — six folds across trailing, centred,
unbounded-preceding and leading frames, since the two-stack only reloads when its pop side
empties and a reload bug appears only once the window has slid far enough to trigger one —
plus the zero-recovery, false-recovery, and all-null-frame-is-NULL cases asserted directly.

### 177-178. `.map.len()` and `.map.contains()`, with no kernel

**Source:** this file's own open list, item 4 — "`.map` has three methods against DuckDB's
eleven."

Two of the eight missing ones need no Rust at all, because the key list already determines
the answer: a map has exactly one key per entry by construction, so the key list's length
*is* the cardinality, and membership in the map is membership in that list. `.map.len()` is
`keys().list.len()` and `.map.contains(k)` is `keys().list.contains(k)`. Both reach SQL as
`cardinality` and `map_contains`.

Composing rather than writing a kernel is only correct if the composition **carries nullness
the same way**, which is the whole risk and what the test file exists to check. Three cases
a naive implementation collapses into one: a *null* map answers NULL, not `0`/`false`; an
*empty* map answers `0`/`false`, not NULL; and an absent key in a non-empty map answers
`false`, not NULL. The fixture asserts the full vectors — `[2, 1, None, 0]` and
`[True, False, None, False]` — so a composition that flattened any pair would fail rather
than merely differ on one row.

Spark's `map_contains_key` is deliberately **not** in the anonymous-function table: sqlglot
gives it a typed node, so a row there would never be reached and would be dead code.

The remaining `.map` gaps (`map_entries`, `map_from_entries`, `map_concat`, `map_extract`,
`element_at` on a map) all need a kernel, since none of them is recoverable from the key
list alone.

**Verified** by five differential cases against DuckDB, two of them through `bt.sql`.

### 179-182. Four boolean normalizations, and the two of them that are half-rewrites

**Source:** a rule-by-rule probe of `duckdb/src/optimizer/rule/` and
`datafusion/datafusion/optimizer/src/` against Batcher's optimizer — 40 candidate
rewrites, run through `Optimizer().optimize()` and checked on the resulting plan.

The headline is how little was missing: **26 of the 40 already fired**, and four more
that the probe reported as gaps were false negatives, already handled at construction
(`5 = x` is built as `x = 5`) or by another rule (`x > x` collapses to a zero limit).
Distributivity, De Morgan, the arithmetic identities, the `LIKE` family and the limit
collapses were all there. Four were real, and are now in
`kyber/rules/normalize/predicates.py`:

* `self_comparison_to_null_check` (DuckDB `comparison_simplification`) — `a = a`,
  `a <= a`, `a >= a` become `a IS NOT NULL`, answerable from a null count instead of a
  scan.
* `boolean_case_to_predicate` (DuckDB `case_simplification`) — `CASE WHEN c THEN true
  ELSE false END` becomes `c`, which unlike the `CASE` can yield a range, push into a
  source, or become a join key.
* `intersect_in_lists` (DuckDB `in_clause_simplification`) — two `IN` lists on one
  column under an `AND` become their intersection, which is what bloom probing and
  source pushdown actually send down.
* `constant_group_key_removed` (DataFusion `eliminate_group_by_constant`) — a literal
  grouping key builds a one-entry hash table and, distributed, sends the whole relation
  to a single node. The key returns as a projection so the schema is untouched.

**The valuable part of this entry is what the tests refused.** Two of the four were
written wrong first, and both mistakes were three-valued:

* `self_comparison_to_null_check` was applied everywhere. `a = a` on a null row is NULL
  and `a IS NOT NULL` is FALSE, so it returned `false` where DuckDB returned `NULL` as
  soon as the predicate was *projected*. It is now filter-only.
* `boolean_case_to_predicate` also rewrote the swapped branch, `THEN false ELSE true`,
  to `NOT c`. That one is wrong **even under a filter**: the `CASE` sends a NULL
  condition to `ELSE` and keeps the row, while `NOT c` is NULL and drops it. The rule is
  now one-directional, and the asymmetry is stated rather than left as an omission.

Neither would have been caught by a filter-only test, which is why every predicate in
the fixture is run twice — once filtered, once projected as a value — over a column that
carries a null. That is the same shape the prefix/length ledger entry used, and it has
now paid twice.

**Verified** by 15 plan-shape cases and 23 differential cases, with the negative tests
(the projection context, the swapped `CASE`, the disjoint `IN` pair, the all-constant
grouping over an empty input) carrying most of the weight.

### 183. The census was measuring the wrong denominator

**Source:** re-running the DuckDB census after entries 177-182 and reading the gap list
instead of the total.

Not an engine change — a correction to the instrument every number in this file comes
from. The headline moved from **54% to 79%** with no change to Batcher, because 135 of
the 478 "missing functions" were `icu_collate_*`, one per locale, and five more were the
harness's own synthesized arguments being invalid. See *What the DuckDB denominator
excludes* above for the full accounting.

It is recorded as an entry rather than a silent edit because a wrong measurement is worth
more attention than a missing function: it had already been quoted in this ledger's
summary table, and every later wave would have been judged against it. Two of the three
bugs found here were in fixes written *during* this correction, and both were silent —
which is why the classifier now has a test rather than a comment.

### 184-186. A provably-empty join side, and the constant that stopped being one

**Source:** Spark's `PropagateEmptyRelation` and DataFusion's `propagate_empty_relation`,
against Batcher's rule of the same name — which folded through unary operators and unions
and stopped dead at a join.

Three things, and the middle one is the reason the other two could not work:

* **The fold reaches joins.** An empty left empties `inner`, `left`, `semi` and `anti`; an
  empty right empties `inner` and `semi`, and turns an `anti` join into *its own left
  input*, removing the join and the entire right subtree.
* **A boolean literal now decides itself** in `_predicate_status`. Constant folding runs in
  NORMALIZE, but `filter_null_join_keys` runs later, in SELECTION, and rewrites a `false`
  predicate under a join into `false AND k IS NOT NULL`. Nothing folds after that, so the
  conjunction read as undecidable, the side was no longer recognizable as empty, and the
  predicate shipped to the engine to be evaluated per row. This is a **phase-ordering
  defect**, not a missing fold: the rewrite was correct, it just arrived after the pass
  that would have simplified it.
* **The asymmetry is stated, not assumed.** An empty *left* leaves a `right` or `full` join
  **non-empty** — both keep the right side's rows padded with nulls. The first version of
  the rule had all six join types in the empty-left set with a comment confidently
  explaining why; it was wrong, and it was a wrong *answer*, not a slow plan.

**What this does and does not buy.** The `anti` and `semi` rewrites remove a subtree, so
they pay immediately. The rest produce `Limit(node, 0)`, the canonical empty marker — and
that marker is currently plan-level only: `bc-interp`'s `Limit` arm executes its input and
*then* discards every row, because `ops::limit` recovers the schema from the first input
batch and there is no Rust-side plan schema inference. So zone-map pruning can prove a scan
yields nothing and the scan still runs. Closing that needs a first-class `Empty { schema }`
relation, which DataFusion (`EmptyRelation`), DuckDB (`LogicalEmptyResult`) and Spark
(`LocalRelation`) all have and Batcher does not. It is the largest engine-level finding
still open here.

**Verified** by 16 differential cases: all six join types against an empty left and an
empty right, plus the right-join asymmetry, the anti-join collapse, an ordinary join left
untouched, and the `false AND k IS NOT NULL` shape asserted directly.

### 187-189. A map lookup that crashed with an error about integers

**Source:** re-probing the `.map` family from SQL after entries 177-178, which added the
DataFrame methods but never asked whether SQL could reach them.

Three spellings of one operation, all of which had a working kernel sitting behind them:

* **`m['a']` raised `invalid literal for int()`.** The `exp.Bracket` handler assumed every
  subscript was a list index and called `int()` on the key, so an ordinary map lookup died
  with an error message about integers on a query containing no integer. It now dispatches
  on the key's *type*: a string key is a map lookup, an integer stays a list index.
* **Spark's `element_at(m, 'a')` parses as the same `Bracket` node**, so it failed
  identically, and is fixed by the same branch.
* **`map_keys(m)` reported "unsupported SQL expression" while `map_values(m)` worked.** The
  two differ only in that sqlglot gives one a typed node and leaves the other anonymous —
  the same asymmetry entries 1-100 were opened to fix, still producing new instances.

**`map_extract` is deliberately still absent**, and this is the entry's real content.
DuckDB returns a **list** for it — `[1]` for a hit, `[]` for a miss — where the subscript
returns the bare value. Mapping it to `.map.get` would have answered `1`/`NULL`: a
plausible result that is not DuckDB's. It was in the table for one test run, and the
differential fixture is what took it out.

The dispatch has a stated limit: a translator has no schema, so an *integer-keyed* map
still resolves as a list index. A string key is unambiguous, because no list is indexed by
one.

**Verified** by eight differential cases, including the two negative ones — an integer
subscript still being a list index, and Spark's 1-based list offset surviving the new
branch.

### 190-194. Five list names the alias sweep structurally could not see

**Source:** the census run in reverse — instead of asking which DuckDB names Batcher
lacks, asking which *Batcher methods* have a real DuckDB name that `bt.sql` cannot reach.
83 methods qualified; 9 were unreachable.

`list_first`, `list_last`, `list_median`, `list_position` and `array_position` all had a
`.list` method that already returned DuckDB's answer, and no SQL spelling that reached it.
They survived the earlier sweeps for a structural reason worth recording: that sweep
paired every `list_X` with its `array_X` twin and probed the pair, and these four have **no
twin in DuckDB's catalogue** — so they were never proposed as candidates at all. A census
only finds what its generator can express.

Two argument shapes, so two tables, following this module's rule that the shape is a
property of the entry rather than something re-derived from arity: `_UNARY_LIST` for the
bare calls and `_LIST_VALUE` for `list_position(l, v)`, which takes a *literal* value and
returns a **1-based** index that `.list.position` already produces — unlike `list_extract`,
there is no origin to shift, and shifting it would be a silent off-by-one.

**The other four of the nine are not closed**, and the reason is a real engine limit rather
than a missing name: `damerau_levenshtein`, `hamming`, `jaccard` and `jaro_similarity`
accept only a *constant* second argument, because the `.str` methods take a Python `str`
and not an expression. Passing a column raises `TypeError: Object of type Col is not JSON
serializable` — a raw serializer error escaping to the user, which is its own defect.

The `array_*` vector aliases added alongside (`array_cosine_similarity`, `array_distance`,
`array_inner_product`, …) are **not** counted here. DuckDB's `ARRAY` names accept only its
fixed-size type and reject a `LIST`; Batcher's kernel is the other way round. The names now
reach the right kernel for a Batcher list, which is a strict improvement, but the
DuckDB-typed call is a separate type-support question and is not claimed as parity.

**Verified** by eight differential cases, including the 1-based/absent pair that a 0-based
implementation gets wrong and a `first`-is-not-`min` check.

### 195-199. A struct is a keyed container, and one kernel now says so

**Source:** the open list's item 4 — "`.struct` has one method (`field`); DuckDB has
`struct_insert`/`struct_keys`/`struct_values`" — re-probed from SQL rather than from the
DataFrame API.

None of SQL's spellings of a struct field access reached the `.struct.field` that has
worked all along. `s['a']` hit the `element_at` kernel, which rejected anything that was
not a `Map`; `struct_extract(s, 'a')` was an unhandled typed node; `struct_keys(s)` had no
implementation at all.

**The fix is one kernel, not three.** A struct is a keyed container too, so `element_at`
now resolves a *name* against the struct's fields where it scans a map's entries. That
placement is the point: the translator has no schema and cannot tell `s['a']` from
`m['a']`, so the disambiguation has to happen where the array's type is actually known.
Doing it in the translator would have meant guessing.

Three things the kernel has to get right, each with its own test:

* **A null struct row answers null**, even though its children hold values underneath.
  Arrow keeps the null mask on the parent and leaves the child buffer unconstrained, so a
  bare `column_by_name` resurrects a value inside a null row. The child is returned with
  the parent's nulls merged in.
* **An absent field is an error, not a null** — the opposite of a map, whose missing key is
  an ordinary result. A struct's fields come from its type, so naming one it lacks is a
  mistake.
* **`struct_keys` repeats the same list on every row** but still nulls a null row, which is
  what keeps it from being a constant.

`.struct` goes from one method to three (`field`, `get`, `keys`). `struct_values` is
**not** implemented: DuckDB returns a container whose members need not share a type, which
is not a `List`, and guessing a homogeneous one would be a different function.

**The dot form `s.a` still fails**, and the reason is worth recording because it is not
where anyone would look: sqlglot parses it as a `Column` qualified by table `s`, so it is
rejected during *column resolution*, before expression dispatch runs. A different layer,
and still open.

**Verified** by nine differential cases and six Rust unit tests, including the null-parent
case and a map-subscript regression check — teaching `element_at` about structs must not
disturb the path it already served.

### 200-209. Half the `.list` namespace rejected an embedding column

**Source:** the `array_*` aliases from entries 190-194, which routed correctly but failed
in the kernel — pulling that thread found the real defect underneath.

A vector column is a `FixedSizeList`: that is how Arrow and Parquet store one, and what
DuckDB's `ARRAY` type maps to. **Half the `.list` namespace accepted it and half rejected
it, on the same column.** `sum`, `l2_norm`, `normalize`, `softmax`, `dot`,
`cosine_similarity`, `sort`, `reverse`, `unique`, `arg_sort`, `cum_sum`, `diff`, `median`
and `n_unique` all worked. `get`, `slice`, `contains`, `position`, `first`, `last`,
`concat`, `intersect`, `transform`, `filter` and `join` raised `expected a List argument,
got FixedSizeList`.

So an embedding could be normalized and summed but **not subscripted** — `e[0]` on a
vector column was an error, in the same query where `e` could be normalized.

The split was not a decision anyone made. The vector kernels had grown a coercion helper
(`list_ops::coerce::as_var_list`, whose own doc comment says the widening cost is
"negligible next to the embedding payload"), and the indexing half of the namespace simply
never used it. `require_list` now routes through the same helper, so `.list` means one
thing for both encodings, and the coercion lives in exactly one place.

This is the kind of gap a function census structurally cannot find: every one of these
names *exists* and answers correctly — on one of the two encodings the same column type
has. Only running them against the encoding a real embedding arrives in shows it.

**Verified** by ten differential cases against DuckDB over the same values in both
encodings, plus two negative tests: a variable-size list is unaffected, and a genuine
non-list argument still raises an error naming the function.

### 210-249. `LargeList` was rejected by the entire `.list` namespace

**Source:** the same encoding-coverage probe that found entries 200-209, extended to the
third Arrow list encoding — and to the string ones, which turned out clean.

Arrow has three ways to hold a list column. After the previous wave, `List` and
`FixedSizeList` both worked. **`LargeList` was rejected by all 39 no-argument `.list`
methods**, so the namespace was entirely unusable on such a column.

That matters more than a missing function, because `LargeList` is not exotic: it is what
an Arrow reader hands back for a `large_list` Parquet column **regardless of its actual
size**, and what Arrow reaches for once a list passes `i32::MAX` offsets. For an engine
whose stated range runs to PB scale, the encoding chosen *for* large data was the one the
list namespace could not read.

One edit fixed all 39, because the previous wave had already made
`list_ops::coerce::as_var_list` the single normalization point. `simhash` was the one
kernel still downcasting on its own; it now goes through the same helper, and the
differential file asserts its output is **bit-identical** across encodings — a similarity
hash that varied by encoding would return different neighbours for the same vector
depending on how its column was written, with no error anywhere.

The narrowing to 32-bit offsets is stated rather than assumed: it is safe *per morsel*
(16,384 rows would need over 131,000 child elements each to overflow) and the Arrow cast
**errors** rather than truncating beyond that. Making the kernels generic over the offset
type removes the bound entirely and is the follow-on, not this change.

**The string encodings were probed the same way and are clean** — 135 `.str` methods
across `Utf8`, `LargeUtf8` and `StringView`, and 135 over a dictionary-encoded column,
with zero rejections. That closes the open list's "StringView end-to-end" item as already
done, and is recorded because a negative result from this probe is worth as much as a
positive one: it says where *not* to look next.

**Verified** by 38 differential cases — eight methods × three encodings against DuckDB,
eleven more compared across encodings where DuckDB has no counterpart, the simhash
identity, a nested `LargeList` flatten, and a non-list column still erroring.

### 250-298. Every math function rejected a `DECIMAL` column

**Source:** the encoding-coverage probe again, turned on the numeric axis — 146
no-argument `Expr` methods across Int8/32/64, UInt64, Float32/64, dictionary-encoded, and
`Decimal128`.

The integer and float encodings were clean. **`Decimal128` rejected 82 methods against
int64's 8.** The 8 are correct refusals (bitwise ops, `chr`, `factorial`). The other 74
were not: every math function raised

    RuntimeError: Abs expected a numeric argument, got Decimal128(10, 2)

on a column that arithmetic, comparison, aggregation and negation all handle — **and
handle exactly**. So a Parquet money column could be summed, compared and multiplied, but
not rounded, floored, or passed to `abs`. That is not precision being protected; it is the
question being refused.

One arm of the type match fixed 49 of them: decimals now take the same Float64 promotion
integers already took.

**The result type is a stated trade.** For `sqrt`/`ln`/`exp`/trig, DOUBLE is what DuckDB
returns too, so value *and* type agree. For `abs`/`floor`/`ceil`/`round`/`sign` DuckDB
keeps DECIMAL — so the number matches and the type does not, and exactness is lost above
2^53. This ledger already pinned that divergence for `ceil`/`floor`; it now covers the
family, and the test asserts *both* halves (the values are equal, the types differ) so
neither side can drift without failing. A decimal-preserving path for that subset needs a
scale-aware kernel per operation and is the follow-on.

**33 methods still refuse a decimal**, and the split matters: the bitwise family, `chr`
and `factorial` are correct. `stddev`, `var`, `median` and the window aggregates are
**not** — they reject decimals in `bc-runtime`'s aggregate/window dispatch, a different
crate and a separate change. Recorded rather than left to be rediscovered.

**Verified** by 29 differential cases: value equality for all thirteen functions, type
equality where DuckDB also returns DOUBLE, the pinned type divergence asserted explicitly,
and a check that the exact-decimal paths (`+`, `*`, unary minus, `sum`) still return
`Decimal` and still sum to `4.250` exactly.

### 299-300. Two internal errors that reached the user

**Source:** the reverse census (entries 190-194) found four string-similarity functions
that "require a constant second argument". Looking at *how* they refuse turned out to be
the finding.

Several accessor methods take a value lowered into the JSON IR as a **constant**:
`.str.jaccard(text)`, `.map.get(key)`, `.list.contains(value)`, `.list.position(value)`.
Passing a column to one is a reasonable mistake, and nothing checked for it — so the
failure surfaced far from the call, as an internal error, in two different shapes:

* `.str.jaccard(col("t"))` reached `json.dumps` and raised **`TypeError: Object of type
  Col is not JSON serializable`** — a serializer error naming neither the function nor the
  argument, from a module the user has never heard of;
* `.map.get(col("k"))` and the `.list` literal slots raised a bare **`TypeError:
  unsupported literal type: Col`** at plan-build time, equally far from the line written.

Both violate the same rule (`python-quality.md`): raise the project's typed exceptions
with actionable messages, and validate user input **at the API edge**. Validation now runs
where the node is constructed, so it fails on the line the user wrote, raises `PlanError`,
and names the function they actually called — `jaccard_similarity()`, not `StrFunc`.

The check is deliberately narrow: it rejects an `Expr` and nothing else, so the optional
slots that legitimately hold `None` are untouched.

Not a parity gap — no reference engine is involved — but it is the same category as the
rest of this ledger: a capability that worked, reached through a path that did not.

**Verified** by 16 unit cases across six methods, including that the message names the
argument, that the error is *not* a `TypeError`, that plain literals still work, and that
`.map.contains` correctly reports via the `list.contains` slot it is composed over.

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
* **Result representation.** The whole math family over a `DECIMAL` returns `DECIMAL` in
  DuckDB where DuckDB keeps it (`abs`/`floor`/`ceil`/`round`/`sign`) and `DOUBLE` here —
  see entries 250-298; `unhex` returns a `BLOB` there and `VARCHAR` here; `histogram`
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

## The four censuses, and what each is good for

| Reference | Oracle | Surface probed | Result |
|---|---|---|---|
| DuckDB | the live engine (`duckdb` is a test dependency) | `duckdb_functions()`, 671 listed names, of which 331 are scored (see below), through `bt.sql` | **93 → 260 of 331 (79%)**; 15 wrong answers found |
| Spark | its own `@ExpressionDescription` examples, parsed out of the Scala source | 527 documented examples over 438 registered names, of which 268 are scored, through `bt.sql(dialect="spark")` | **75 → 160 of 268 (60%)**; 35 wrong answers found |
| Polars | the live library | every zero-argument method on `pl.Expr` and its `.str`/`.dt`/`.list` namespaces | 53 supported, 31 absent, 14 differences (all but two are representation or a pinned semantic choice) |
| Daft | the live library | every public method on `daft.Expression`, 226 of which are scored, resolved against Batcher's *whole* namespaced surface | **174 of 226 (77%)**; the 52 remaining are Daft's multimodal differentiator |

### What the denominators exclude, and why it matters

The DuckDB row read **260 of 478 (54%)** until the denominator was audited. It was wrong,
and the correction is larger than any single wave in this ledger:

* **144 out of scope.** `duckdb_functions()` lists **135** `icu_collate_*` entries, one
  per locale. That is one capability, not 135 functions, and it is 40% of the listed
  surface. The rest are DuckDB's catalog and session introspection (`current_schema`,
  `duckdb_functions`, plan serialization), which belong to a database server rather than
  to a query engine.
* **5 unprobed.** `array_cat`, `array_concat`, `list_concat`, `list_pack` and `strftime`
  were reported missing because the harness synthesizes *literal* arguments: sqlglot
  cannot parse `list_concat([1,2],[3])`, and `strftime(ts, 'abc')` is an invalid format
  Batcher is right to reject. All five work over columns, checked by hand.

So a gap must now be **proven** by an error saying the function is unreachable, never
assumed from any failure. Getting that backwards costs accuracy in both directions, and
did: the first attempt classified by exclusion and reported eight working functions as
missing; the second used a capitalized marker against lowercased text, matched nothing,
and quietly moved eleven real gaps into "unprobed". Neither was visible from the output —
the census ran and printed a number either way. `tests/unit/test_parity_census_classifier.py`
now pins both halves of the classification.

The lesson generalizes past this file: **a parity percentage is a claim about the
denominator first.** Read what a reference engine's function list actually contains before
dividing by its length.

Spark needed the same audit and moved **51% to 60%**: 48 of its documented builtins are a
geospatial extension, an XPath/XML family, its own bitmap and variant encodings, cluster
and session introspection, or values that are nondeterministic by definition. Counting
those as missing functions measures Spark's product scope, not Batcher's expression
surface.

### Daft, and the third denominator trap

Daft keeps everything on a flat `Expression` namespace (`list_sum`, `upper`,
`to_snake_case`); Batcher pushes breadth onto accessors (`.list.sum`, `.str.to_uppercase`).
A naive `dir()` difference reports Batcher as missing **220** functions. Resolving each Daft
name against the full namespaced surface, plus recorded aliases where the two engines chose
different words (`avg`/`mean`, `eq_null_safe`/`eq_missing`, `lag`/`shift`) and recorded
*parameterizations* where Daft spells as N functions what Batcher spells as one function
with an argument, the real figure is **174 of 226, or 77%**.

That parameterization category is worth reading twice. Daft's `as_int8`…`as_uuid` is **35
names for one capability** — `.cast(dtype)` — exactly like DuckDB's 135 `icu_collate_*`.
And its eight case converters (`to_snake_case`, `to_camel_case`, …) are all one argument of
`.str.to_case(style)`, which Batcher implements with a *single* word splitter so the styles
cannot disagree about where the words were. Three separate times in this pass an apparent
gap family turned out to be already covered, better.

The 52 that remain are almost entirely Daft's own differentiator rather than a relational
gap: image operations (9), duration accessors (7), Iceberg partition transforms (6),
tokenize/compress (6), file and object-store IO (5), HDF5 (3) and video (3). The Iceberg
partition transforms and the duration accessors are the two families here worth taking.

Spark deserves a note: **it needs no JVM.** There is no Java runtime on this machine, so
`SparkSession` cannot start — but every builtin carries its expected output in an
annotation next to its implementation, which is an oracle a text reader can use. That is
worth remembering the next time a reference engine will not run.

## Open

Ordered by measured value, from the same censuses and a capability probe of the DataFrame
surface.

1. **`.dt.date()` returns a midnight `TIMESTAMP`, not a `DATE`.** It is written as an
   alias for `truncate('day')` and documented as returning a timestamp, but it is named
   after Polars' `dt.date` and both Polars and DuckDB (`CAST(ts AS DATE)`, `date(ts)`)
   return a `DATE`. Batcher's own `cast('date')` already does. A user who reaches for the
   method the reference engines name gets a type that will not compare against a date
   column. Found by the Polars census; `last_day(DATE)`, the earlier entry here, has since
   been fixed and now returns `date32` for both `DATE` and `TIMESTAMP` inputs.
2. **The window aggregates entries 152-160 did not take.** `median`, `quantile` and
   `mode` need an order-statistic structure for their running form; `arg_min`/`arg_max`
   and the two-input aggregates need a second input, which `WindowCall` has nowhere to
   put; `array_agg` accumulates O(n) state per row in the running form. Separately, none
   of the nine that landed honours an explicit `ROWS` frame. **Closed by entries 171-176**
   for the six folds; `var`/`stddev` and `count_distinct` remain, and cannot be closed the
   same way (a Welford state is not a fold, and a distinct count needs a multiset).
3. **Aggregates the engine does not have at all**: `entropy`, `mad`, `approx_top_k`,
   `bitstring_agg`, `histogram_exact`, `quantile_disc`, the `arg_*_null` variants, and a
   true `any_value`/`first`/`last` with no ordering requirement.
4. **`struct` and `map` namespaces.** `.struct` has one method (`field`); DuckDB has
   `struct_insert`/`struct_keys`/`struct_values`, Polars has more. `.map` is at five of
   DuckDB's eleven after entries 177-178; the six that remain (`map_entries`,
   `map_from_entries`, `map_concat`, `map_extract`, `element_at` on a map) each need a
   kernel, since none is recoverable from the key list the way `len` and `contains` were.
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
