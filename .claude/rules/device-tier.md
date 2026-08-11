# Rule: The Device (GPU) Execution Tier

`core/gpu_plan/` + `dist/gpu/` translate the JSON IR onto **cuDF** and run it on a GPU. This
file is that tier's contract. Read it before changing anything under either directory.

## Why this tier is different, and why that is not a mistake

Every other execution tier consumes the *same Rust `bc_expr::Expr`* the interpreter does. That
shared source is what makes invariant #6 ("one `Expr`, one `RelOp`, across tiers") mechanical:
the Cranelift JIT cannot drift from the oracle on a shape it claims, because there is only one
definition of what the shape means.

The device tier cannot work that way. **cuDF has no maintained Rust binding**, so the kernels
are reachable only from Python. It is therefore a *translator* — a second statement of the
engine's semantics, in another language, against another library — and it is the only tier
that is. That is a deliberate, accepted cost, not a layering accident to be "fixed" by moving
it into Rust; there is nothing in Rust to move it to.

What follows is what that cost obliges you to do instead.

## The contract

1. **The device changes *where* a plan runs, never *what* it computes.** Same rows, same
   column names, same column *types*. A result that differs from the CPU engine's is a defect,
   never a decline.
2. **Decline rather than approximate.** Anything outside the translated subset raises
   `Unsupported` and the stage runs on the CPU engine. A silently approximated shape costs a
   wrong answer; a declined one costs a fallback. `backend="gpu"` is documented as always safe
   and must stay so.
3. **Every IR tag is classified.** A `RelOp` is in `ops.SUPPORTED_OPS` or in
   `ops.DECLINED_OPS` with a reason; an `Expr` is in `exprs._HANDLERS` or in
   `exprs.DECLINED_EXPRS` with a reason. Enforced by
   `tests/unit/test_gpu_vocabulary_contract.py`, which also fails when a handler is keyed on a
   tag the engine no longer has.
4. **The tier is opt-in.** `backend` defaults to `"cpu"`; the tier is reached only via an
   explicit `backend="gpu"`/`"auto"` *and* a cluster with visible GPUs. This is the strongest
   safety property it has, and it is pinned by a test rather than left to a default.
5. **A divergence is a defect.** Report it through `note_gpu_failure`, which logs at warning
   level. Never through `note_suppressed`, which is for declines.

## Verification: what the test suite can and cannot see

**The translator's suite runs on pandas, never on cuDF**, because CI has no GPU. `DfBackend`
branches on `is_gpu` at every Arrow boundary, so the two backends do not take the same code
path where it matters most — and that gap has already shipped several defects. **Every one of
them was a column *type* defect with correct values:**

- a DATE column returning `date32` under pandas and `timestamp[ms]` on a real device
  (`backend.py::remember_date_alias`);
- an integer `abs` widening to double (`backend.py::is_integer`);
- an **empty** cuDF string column converting to Arrow `null` (`backend.py::_restore_empty_strings`).

That pattern is not a coincidence and it is what the tier's verification is built around.

**The schema contract is always on and needs no device.** `LogicalPlan.available_schema` is
the engine's own static type analysis — the one `Dataset.schema` is answered from — and it runs
before either backend does, on no rows. So every device result is held against the engine's
declared column types on every query, for the price of a field-list walk
(`api/terminal/gpu_backend/verify.py::enforce_schema_contract`). A result that disagrees is
refused and the CPU engine answers. All three defects above would have declined instead of
returning a wrong column. **Do not gate this behind a flag or skip it for a "cheap" path** —
its value is that it is unconditional, and it is the only check that sees the tier's
characteristic defect without a GPU lane.

It sees *types*, not values. So:

- **Changing anything under `core/gpu_plan/` requires a run with
  `distributed.gpu_shadow_verify=True` on real hardware**, and the clean result recorded in
  `benchmarks/BENCHMARK_RESULTS.md`. Shadow-verify re-runs each result on the CPU engine and
  compares schema first, then values; it is the only oracle for *values* the device has.
- Add the pandas-backed case to `tests/unit/test_gpu_plan.py` as well, and — when the change
  can move a result's type — to `tests/unit/test_gpu_schema_contract.py`, which runs the real
  translation and holds it to the declared schema. Both are necessary and neither is
  sufficient: treat a green suite as "the translation is plausible", not "the device agrees".
- **Derive a type rule, never restate one.** `plan.types.widen` is the control plane's single
  mirror of `bc_py::normalize_to`; `backend.py::widened_type` delegates to it. A second copy
  is how the device came to keep a `dictionary` column encoded where the engine decodes it.

## Structure

- `core/gpu_plan/` — the translator: `eligibility` (which plan shapes match), `ops` (RelOp →
  dataframe op), `exprs` + `vocab/` (Expr → dataframe expression), `aggs`, `windows`,
  `temporal`, and `backend.py` (the cuDF/pandas adapter).
- `dist/gpu/` — sharding, device placement and the mergeable fan-out across several GPUs.
- `api/terminal/gpu_backend/` — routing, the fallback contract (`failure.py`), and
  `verify.py`.

The translator computes per-row work in Python, which `.claude/rules/architecture.md` forbids
on a hot path. It is the one sanctioned exception, for the reason at the top of this file, and
it does not extend to anything else.

## Gate before "done"

The canonical matrix is in `CLAUDE.md`. Device-tier delta: `tests/unit/test_gpu_plan.py` and
`tests/unit/test_gpu_vocabulary_contract.py` green, plus a recorded `gpu_shadow_verify=True`
run on hardware for any change to the translation itself.
