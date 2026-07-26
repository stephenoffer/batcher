# Parity censuses: run the competitor, not just read it

Three harnesses that execute a reference engine and Batcher side by side over the
reference's **whole function surface**, then sort every result into three buckets:

| Bucket | Meaning |
|---|---|
| match | Batcher returns the reference's answer |
| gap | the reference answered, Batcher raised |
| **mismatch** | both answered, **differently** |

The mismatch column is why these exist. A gap is an honest refusal that a user sees
immediately; a mismatch is a wrong answer nobody has noticed. Reading a competitor's source
tells you what it *has*; only running it tells you what Batcher *answers*.

The findings and the build log against them live in
`docs/internals/competitor_parity_census.md`.

## Running them

Each script prints JSON on stdout and a one-line summary on stderr. They need the engine
built (`just build`) and take a few minutes each, mostly in process startup per query.

```bash
python tools/parity/duckdb_census.py > /tmp/duckdb.json   # needs `duckdb` (a test dep)
python tools/parity/spark_census.py  > /tmp/spark.json    # needs the Spark source, no JVM
python tools/parity/polars_census.py > /tmp/polars.json   # needs `polars` (a test dep)
```

To check a change for regressions, run one before and after and diff the `match` sets. The
rule the ledger holds itself to is that **no function that previously matched may stop
matching**; that diff is how it is checked.

## What each one uses as its oracle

**`duckdb_census.py`** reads DuckDB's own catalogue (`duckdb_functions()`), synthesizes an
argument per parameter type, and calls every overload in both engines. A function counts as
supported when *any* overload agrees, so one unrepresentable argument type (BLOB, BIT, a
nested LIST) does not report the whole function as a gap.

**`spark_census.py`** needs **no JVM**, which matters because there usually is not one:
Spark annotates every builtin with an `@ExpressionDescription` holding
`> SELECT _FUNC_(args);` and the expected output, and `FunctionRegistry.scala` maps each
expression class to its SQL name. Together they are an executable oracle a text reader can
use. Point `SPARK` at a Spark checkout.

Remember what `dialect="spark"` means when reading its output: it selects a **parser**, not
a semantics. Where Spark and DuckDB genuinely disagree on what a function means the engine
follows DuckDB, so those rows are *expected* mismatches — they are tabulated in the
`migrate-from-spark` skill.

**`polars_census.py`** calls every zero-argument method on `pl.Expr` and its `.str`/`.dt`/
`.list` namespaces against the same-named Batcher method. Zero-argument only, so the call
is unambiguous without a signature model.

## Reading the output honestly

Three classes of "mismatch" are not defects, and each will appear:

- **Representation.** `ceil` over a DECIMAL returns DECIMAL in DuckDB and DOUBLE here;
  `unhex` returns a BLOB there and VARCHAR here. Different type, same value.
- **Unspecified order.** `list_distinct` returns its elements in an unspecified order in
  both engines, so the comparison is meaningless for it.
- **A pinned semantic choice.** `sem` divides the population stddev in DuckDB and the
  sample one here (matching pandas and scipy), and `round` is half-away-from-zero here and
  half-to-even in Polars. Each is recorded in the ledger with its reasoning.

A mismatch that is none of those three is a bug. That is the whole point of the column.
