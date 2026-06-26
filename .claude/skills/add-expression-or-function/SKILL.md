---
name: add-expression-or-function
description: Recipe to add a scalar/aggregate function, an expression IR node, or a typed-accessor method (.str/.dt/.list/...) to Batcher — using the declarative IRNode base, the fn_names vocabulary, and the function family modules — while keeping the JSON IR wire-contract byte-stable. Invoke when adding or changing an expression, scalar/agg function, or namespace accessor (not a relational operator — that is add-relational-operator).
---

# Add an expression or function

Batcher's expression layer is the Python control-plane representation of the one
scalar `Expr` type. There are three things you might add, in increasing cost:

1. **A function** — a composition of existing IR nodes (`concat`, `corr`). No new wire
   shape, no Rust change.
2. **A typed-accessor method** — breadth on `.str`/`.dt`/`.list`/… namespaces.
3. **An expression IR node** — a *new wire shape* the Rust engine deserializes (you are
   also adding a `bc_expr::Expr` variant). This is a two-sided change.

Read `.claude/rules/python-control-plane.md` (the IR wire contract),
`.claude/rules/maintainability.md` (group by family, keep `Expr`/`Dataset` thin), and
`docs/internals/extending.md` (the cookbook this skill automates) first.

- **Node base:** `plan/expr_ir/node_base.py` — `IRNode`, `@expr_node`, and the field
  factories `child` / `children` / `scalar` / `literal`.
- **Function vocabulary:** `plan/expr_ir/fn_names.py` — the per-family `fn` sets/enums.
- **Wire tags:** `plan/ir_tags.py::ExprTag` — the `"e"` discriminator (mirrors Rust).
- **Function families:** `plan/functions/<family>.py`; accessors in
  `plan/expr_ir/namespaces/<family>.py`; public façade `api/functions.py`.
- **The wire lock:** `tests/unit/test_ir_snapshot.py` — golden `to_ir()` per node.

## The invariants

- **Never touch a tuple in Python.** A function/expression builds an `Expr`; the
  per-row evaluation is Rust. If you find yourself looping over values, stop.
- **The JSON IR is a stable wire contract.** A node's `to_ir()` `tag` and field keys
  must match the Rust `serde` shape byte-for-byte. A new node = a Rust `bc_expr::Expr`
  variant in the **same commit**, with a round-trip/differential test.
- **One obvious way.** Don't add a second spelling of an existing capability. New
  public surface is a commitment — justify it against DuckDB/Polars ergonomics.
- **Thin builders.** Per-type breadth goes on accessor namespaces, not as new methods
  on `Expr`/`Dataset`.

## Recipe A — a function (no new IR)

1. Write it in `plan/functions/<family>.py` as a pure `(…) -> Expr` built from existing
   nodes. Google-style docstring with a runnable `.. doctest::`.
2. Re-export from `plan/functions/__init__.py` and `api/functions.py` (both `__all__`s).
3. If it is an aggregate, return an `AggExpr` (or compose one); confirm it is mergeable
   (`partial → combine → finalize` exists in `bc-runtime`) or it is single-node only.
4. **Differential test vs DuckDB** in `tests/differential/` — nulls, empties, type edges.

## Recipe B — a typed-accessor method

- Parameterless transform/field/reduction → add **one row** to the family dispatch
  table (`_STR_TRANSFORMS`, `_DT_FIELDS`, `_LIST_FUNCS`); the accessor is generated.
- Parameterized → a thin method on the namespace class that builds the node
  (`return StrFunc("slug", self._e, replacement=sep)`).
- Add the engine `fn` name to the matching family vocabulary in `fn_names.py`.
- Differential test through the accessor (`col("s").str.slug()`).

## Recipe C — a new expression IR node (two-sided)

1. **Tag:** add it to `plan/ir_tags.py::ExprTag` and the Rust `serde` variant together.
2. **Node:** add an `@expr_node` `IRNode` subclass in `func_nodes.py` (or `nodes.py` /
   `core.py`). Set `tag`, declare fields with `child`/`children`/`scalar`/`literal`,
   and (if it carries a `fn`) set `vocab` to its family set in `fn_names.py`. Do **not**
   write `__init__` or `to_ir` — the base generates them. An irregular shape may
   override `to_ir` (see `Case`).
3. **Rust:** implement the variant in the `bc-expr` interpreter (the oracle) first;
   teach the JIT only with proven parity, else let it fall back.
4. **Surface:** build the node from a constructor (`plan/expr_ir/constructors.py`) or an
   accessor; re-export public names.
5. **Tests:** add a representative to `tests/unit/test_ir_snapshot.py` (locks the wire
   shape), a Rust round-trip test (Python `to_ir()` deserializes into `bc_expr::Expr`),
   and a **differential test vs DuckDB**.

## Gate before done

`just lint-py` → `just lint-layers` → `just lint-structure` → `just build` →
`just test-py`. A new IR node also runs `just test-rust`. `test_ir_snapshot.py` must
stay green (or be re-baselined with `BATCHER_REGEN_IR_SNAPSHOT=1` only when the wire
contract genuinely changed, in the same commit as the Rust change). See
`/run-quality-gate`.
