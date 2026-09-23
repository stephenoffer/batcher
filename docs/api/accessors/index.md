# Accessor namespaces

This section is the reference for the typed accessor namespaces an expression carries. An accessor is the second half of the column language: {py:obj}`Expr <batcher.plan.expr_ir.core.Expr>` itself holds the arithmetic, the comparisons, and the aggregates that make sense for any column, and everything that depends on what the column *holds* lives behind a namespace named for that kind.

```python
import datetime

import batcher as bt

ds = bt.from_pydict(
    {
        "city": ["Oslo", "Lima"],
        "ts": [datetime.datetime(2026, 1, 5), datetime.datetime(2026, 2, 11)],
    }
)
print(ds.select(width=bt.col("city").str.len_chars(), month=bt.col("ts").dt.month()).to_pydict())
# {'width': [4, 4], 'month': [1, 2]}
```

Reaching for the wrong namespace is a plan-time error rather than a wrong answer, and the accessor is the only thing about the method that is special. A namespace method is an ordinary expression: it composes with the scalar algebra, it lowers to the same Rust data plane, it can be pushed into a scan, and it is callable from SQL under the same name.

## The namespaces

The following table maps each namespace to the column kind it attaches to and the page that lists its methods. Rows run from the namespaces most pipelines reach for to the most specialized.

| Namespace | Attaches to | Reference |
| --- | --- | --- |
| `.str` | A string column | {doc}`strings` |
| `.dt` | A date, time, or timestamp column | {doc}`temporal` |
| `.list`, `.struct`, `.map`, `.json` | A nested column, or a string holding JSON text | {doc}`nested` |
| `.image`, `.audio`, `.video` | A binary column holding encoded media | {doc}`media` |
| `.seq` | A text column read as DNA, RNA, protein, or FASTQ quality | {doc}`sequence` |

Two more namespaces are documented next door, because neither one computes over a column. `.name` reshapes the output names a selector expands to, and is listed with the selectors on {doc}`/api/symbols/expression-builders`. {py:obj}`.meta <batcher.plan.expr_ir.core.Expr.meta>` reports what an expression *is* without evaluating it, and is on {doc}`/api/symbols/expression-modeling`.

## Why decode is an expression

The media namespaces are the case where this design earns the most. A JPEG decode is expensive, and putting it behind an expression rather than a Python loop is what lets the optimizer refuse to pay for it: a filter on an image's dimensions is answered from the header, and a projection that drops the decoded column removes the decode from the plan entirely. The same applies to a waveform resample and a video frame grab.

That is also why these methods produce tensor columns rather than Python objects. The decoded pixels stay in Arrow, so a model runs over them without a copy and without leaving the engine. {doc}`/ml/preparing/multimodal/index` covers the pipeline those columns feed.

## See also

- {doc}`/api/relational/expression-accessors`: the same surface with a runnable example per namespace.
- {doc}`/api/symbols/expression-methods`: the `Expr` methods that need no accessor.
- {doc}`/user-guide/transform/columns/index`: the guides these pages are the reference for.
- {doc}`/api/reference`: the one-page lookup table for the whole public API.

```{toctree}
:hidden:

strings
temporal
nested
media
sequence
```
