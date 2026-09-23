# The column language

This section covers {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`, the language every column computation in Batcher is written in, and the batch UDF for the rare job it cannot express.

An expression describes a computation rather than running one. `bt.col("price") * bt.col("qty")` builds a small typed tree that the optimizer can inspect, push into a scan, or drop when nothing reads it, and that the Rust data plane then evaluates over whole Arrow batches. The Cranelift JIT compiles the arithmetic it supports and falls back to the interpreter for the rest, with identical results either way. That is why column work in Batcher never becomes a Python loop, and why the same expression is fast on three rows or three billion.

The language is broad enough that you rarely leave it. Typed accessor namespaces put hundreds of methods one keystroke away: text cleaning, regex, and compression on `.str`, calendars and time zones on `.dt`, nested data on `.list`, `.struct`, `.map`, and `.json`, whole images, waveforms, and video clips on `.image`, `.audio`, and `.video`, and even genomic sequences on `.seq`. Pandas and Polars spellings are there as aliases, so a ported script mostly runs as written. Every method is callable from SQL as well.

Read {doc}`Expressions <expressions>` first. The rest of the section assumes it.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`code;1.1em` Expressions
:link: /user-guide/transform/columns/expressions
:link-type: doc
Building, combining, and reusing an `Expr`: operators, conditionals, nulls, and math.
:::

:::{grid-item-card} {octicon}`stack;1.1em` Expression accessors
:link: /user-guide/transform/columns/expression-accessors
:link-type: doc
The `.dt`, `.list`, `.struct`, `.map`, and `.json` namespaces, one per column kind.
:::

:::{grid-item-card} {octicon}`typography;1.1em` The string accessor
:link: /user-guide/transform/columns/string-accessor
:link-type: doc
Search, regex, paths and URLs, identifier recasing, and compressed payloads.
:::

:::{grid-item-card} {octicon}`hash;1.1em` Map columns
:link: /user-guide/transform/columns/map-accessor
:link-type: doc
Building a map with `map_from_arrays` and reading it back as values or rows.
:::

:::{grid-item-card} {octicon}`device-camera;1.1em` Media columns
:link: /user-guide/transform/columns/media-accessor
:link-type: doc
The `.image`, `.audio`, and `.video` namespaces: header facts, decoding to tensors, and quality signals.
:::

:::{grid-item-card} {octicon}`beaker;1.1em` The sequence accessor
:link: /user-guide/transform/columns/sequence-accessor
:link-type: doc
DNA, RNA, protein, and FASTQ-quality columns.
:::

:::{grid-item-card} {octicon}`light-bulb;1.1em` Expression recipes
:link: /user-guide/transform/columns/expression-recipes
:link-type: doc
Porting from pandas or Polars, feature engineering, and curating a training corpus.
:::

:::{grid-item-card} {octicon}`number;1.1em` The type system
:link: /user-guide/transform/columns/type-system
:link-type: doc
What each type means, how casts behave, and why a narrow integer widens at the boundary.
:::

:::{grid-item-card} {octicon}`plug;1.1em` User-defined functions
:link: /user-guide/transform/columns/udfs
:link-type: doc
Your Python over whole Arrow batches, per group, per row, or from SQL.
:::

:::{grid-item-card} {octicon}`server;1.1em` Running a UDF at scale
:link: /user-guide/transform/columns/udfs-at-scale
:link-type: doc
Distributing a UDF stage, tolerating bad rows, retries, and idempotency.
:::
::::

## See also

- {doc}`/api/relational/expressions`: every `Expr` method, enumerated.
- {doc}`/api/accessors/index`: the reference for each accessor namespace, with signatures.
- {doc}`/cookbook/expressions/index`: the same language as runnable recipes.
- {doc}`/examples/expressions`: 40 standalone expression scripts, each run on every commit.

```{toctree}
:hidden:

expressions
expression-accessors
string-accessor
map-accessor
media-accessor
sequence-accessor
expression-recipes
type-system
udfs
udfs-at-scale
```
