# Extending Batcher

This is the contributor cookbook: one recipe per extension point, each naming the
files you touch, the shared base class / registry to reuse, and the tests the change
must carry. The whole point of the structure below is that adding the thousandth
function, expression, rule, or format is a small, local, declarative change — never a
god-file edit or a scattered cross-cutting one.

Read these first: `.claude/rules/architecture.md` (which layer owns what),
`.claude/rules/maintainability.md` (the size limits and "group by family + registry"
rule), and `.claude/rules/testing.md` (the DuckDB differential oracle).

## Where does my change go?

| You want to add… | Go to | Reuse |
|---|---|---|
| A scalar/aggregate function (`concat`, `corr`) | `plan/functions/<family>.py` | compose existing `Expr` nodes |
| A new expression IR node (a new wire shape) | `plan/expr_ir/func_nodes.py` (or `nodes.py`/`core.py`) | `IRNode` + `child`/`scalar`/`literal` |
| A typed-accessor method (`.str.slug`, `.dt.iso_year`) | `plan/expr_ir/namespaces/<family>.py` | the namespace class / dispatch table |
| A new function *name* for an existing node | `plan/expr_ir/fn_names.py` | the family vocabulary |
| A Kyber optimizer rule | `kyber/rules/<family>.py` | `@rule` + `transform_up` |
| An IO format (a reader/writer) | `io/formats/<category>/<fmt>.py` | `FileSource`/`FileSink` + `@SOURCES.register` |
| A relational operator | Rust `bc-runtime` + `plan/nodes/` | see the `add-relational-operator` skill |

The golden rule: **per-row work lives in Rust** behind the JSON IR. Python builds and
optimizes the plan; it never iterates a tuple. If a recipe has you writing a Python
loop over data, stop — that work belongs in the data plane.

## Add a scalar or aggregate function

Most functions are *compositions of existing IR nodes* and need no new wire shape or
Rust change. Put one in the family module under `plan/functions/`, then surface it
through the `api/functions.py` façade.

```python
# plan/functions/string.py
def initials(value: IntoExpr) -> Expr:
    """First letter of each whitespace-separated word, concatenated."""
    # Built from existing nodes — no new IR, no Rust.
    ...
```

Steps:

1. Write the function in `plan/functions/<family>.py` (string, math, temporal,
   aggregate, conditional, collection). Keep it pure: it returns an `Expr`.
2. Re-export it from `plan/functions/__init__.py` and `api/functions.py` (add to both
   `__all__`s — these are the public surface).
3. Add a **differential test vs DuckDB** in `tests/differential/` covering nulls,
   empties, and type edges.

## Add an expression IR node

Reach for a new node only when you need a *new wire shape* the engine deserializes
(i.e. you are also adding a `bc_expr::Expr` variant in Rust). Nodes are **declarative**:
subclass `IRNode`, set the wire `tag`, and annotate each field with a factory. The
generic `IRNode.to_ir` assembles the JSON — you write no `to_ir` and no `__init__`.

```python
# plan/expr_ir/func_nodes.py
@expr_node
class StrFunc(IRNode):
    """A string function over a sub-expression."""

    tag = ExprTag.STR          # the "e" discriminator, from ir_tags.ExprTag
    vocab = STR_FNS            # validate `fn` against the family vocabulary
    fn: str = scalar()         # emitted as-is
    input: Expr = child()      # recursed via .to_ir()
    pattern: str | None = scalar(omit_none=True, default=None)  # dropped when None
```

The field factories (`plan/expr_ir/node_base.py`):

- `child()` — a sub-expression, serialized by recursing into `.to_ir()`.
- `children()` — a list of sub-expressions.
- `scalar()` — a JSON scalar (string tag, int, bool). `omit_none=True` drops `None`;
  `omit_falsy=True` drops zeros the engine's serde defaults.
- `literal()` — a Python constant lifted to its tagged wire value (`{"int": 5}`).

Rules:

- The `tag` string must equal the Rust `serde` tag exactly (CLAUDE.md invariant #8);
  add the tag to `plan/ir_tags.py::ExprTag` and the Rust variant in the **same commit**.
- An irregular shape (e.g. paired branches) can override `to_ir` — see `Case` in
  `nodes.py`. Keep the node an `@expr_node` so it still gets a generated constructor.
- Add a representative to `tests/unit/test_ir_snapshot.py`; that snapshot is the
  byte-for-byte lock on the wire contract.

## Add a typed-accessor method

Per-type breadth lives on the `.str` / `.dt` / `.list` / `.struct` / `.json` / `.map`
namespaces, not as new `Expr`/`Dataset` methods (keeping those thin is how v2 avoids
v1's god-objects).

A **parameterless** accessor is one row in the family's dispatch table — pure data:

```python
# plan/expr_ir/namespaces/strings.py
_STR_TRANSFORMS = {
    "upper": "upper",
    "lower": "lower",
    "swapcase": "swapcase",   # ← add the row; the accessor is generated
}
```

A **parameterized** accessor is a thin method that builds the node:

```python
def slug(self, sep: str = "-") -> StrFunc:
    """Lowercase, replacing runs of non-alphanumerics with `sep`."""
    return StrFunc("slug", self._e, replacement=sep)
```

## Add a function name to the vocabulary

Each function node validates its `fn` against a family vocabulary in
`plan/expr_ir/fn_names.py` at construction, so an unknown name fails early with a
clear `PlanError` instead of an opaque engine error. When you add a function to a
node family, add its name here too:

- **Closed families** (a handful of stable ops) are `StrEnum`s — `MapFn`,
  `ListBinaryFn`, `ListSetFn`, `Math2Fn`. Add a member.
- **Open families** that grow toward hundreds are `frozenset`s — `STR_FNS`,
  `MATH_FNS`, `DATE_FNS`, `LIST_FNS`. Add an entry. (A thousand-member `Enum` class
  would itself be the sprawl we avoid; a named set is the scalable vocabulary.)

These sets are the Python mirror of the Rust `match` arms; the test suite keeps them
exhaustive (a valid function missing from its set would raise at construction).

## Add a Kyber optimizer rule

A rule is a pure `node → node | None` (or `plan → plan`) function. Drop it in the
matching family module, decorate it, and the driver discovers it — no pipeline edit.

```python
# kyber/rules/<family>.py
@rule(name="drop_noop_filter", phase=Phase.NORMALIZE, matches=(Filter,))
def drop_noop_filter(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    if _is_constant_true(node.predicate):
        return node.input      # rewritten node…
    return None                # …or None for "no change"
```

For a **whole-plan** rewrite, do not hand-roll the per-node `isinstance` ladder — the
structural recursion (and the identity-preserving rebuild the fixpoint detector relies
on) is the shared `transform_up` from `plan/visitor.py`. Write only the per-node logic:

```python
def rewrite_predicate(plan: LogicalPlan) -> LogicalPlan:
    def push(node: LogicalPlan) -> LogicalPlan:
        if isinstance(node, Filter) and isinstance(node.input, Join):
            return _push_into_join(node.predicate, node.input) or node
        return node
    return transform_up(plan, push)   # children visited and rebuilt generically
```

Invariants: a rule **decides, never executes** (no engine calls, no metric
collection — that is Core's lane). Every rule needs a `tests/unit/` plan-shape test
proving the rewrite is *semantics-preserving* and a `tests/differential/` test
showing the optimized query still matches DuckDB. See the `add-kyber-optimizer-pass`
skill for the full treatment.

## Add an IO format

File formats subclass the template base in `io/base.py`: set the suffix and format
name, implement the read/write of a single file, and the base handles multi-file
schema union, projection, splits, and streaming.

```python
# io/formats/structured/myfmt.py
@SOURCES.register("myfmt")
class MyFmtSource(FileSource):
    suffix = ".myf"
    format_name = "myfmt"

    def _read_schema(self, fh): ...
    def _read_file(self, fh, projection): ...

@SINKS.register("myfmt")
class MyFmtSink(FileSink):
    suffix = ".myf"
    format_name = "myfmt"

    def _write_file(self, table, fh): ...
```

Then import the module from the category `__init__.py` so the `@register` decorator
runs. Heavy/optional dependencies (`deltalake`, `pyiceberg`, …) are imported lazily
*inside* the methods, so `import batcher` stays cheap and the format's dependency is
optional. `io.formats.SOURCES.names()` / `SINKS.names()` enumerate what is registered.

## Before you call it done

Run the gate (`/run-quality-gate`): `just lint-py` → `just lint-layers` →
`just lint-structure` → `just build` → `just test-py`, plus `just test-rust` if you
touched an IR tag or the FFI, and `just docs` for documentation. A relational or
expression change is not done without a **differential test vs DuckDB**.
