## How to use this map

**Grep this file first.** It is one document holding the docstring summary of every
module in the repo, so `rg -i 'shuffle' MAP.md` answers "where does shuffling live"
in one read instead of a dozen searches that each pull whole files into context.
Open a module only once the map has told you it is the right one.

The three questions the map is built to answer:

| Question | Where to look |
|---|---|
| *Where does this existing thing live?* | Grep the tables below for the concept. |
| *Where does new code go?* | The routing table, then the package one-liner. |
| *What am I allowed to import here?* | The layer number on the package heading. |

Layer numbers are the import matrix from `.claude/rules/architecture.md`: a package
may import anything **strictly lower**, never higher and never sideways. The four
layer-3 subsystems (`kyber`, `carbonite`, `core`, `governance`) are mutually
independent — if two of them need the same helper it goes **down** into `plan`,
`metadata`, `config`, or `_internal`. Copy-paste is the only *wrong* way to share.

## Where does new X go?

`CLAUDE.md` states the rule; this is the lookup table for the cases that come up.

| I am adding… | It goes in | Skill |
|---|---|---|
| A relational operator (join/agg/sort variant) | Rust `bc-ir` + `bc-runtime` (mergeable) → `plan/logical/` → `api` | `add-relational-operator` |
| A scalar or aggregate function | `plan/functions/<family>.py`, surfaced via `api/functions.py` | `add-expression-or-function` |
| A method on `.str`/`.dt`/`.list`/`.struct`/`.json` | `plan/expr_ir/namespaces/` | `add-expression-or-function` |
| An optimizer rewrite | `kyber/rules/<family>.py` or `kyber/rules/extra/<family>.py`, via `@rule` | `add-kyber-optimizer-pass` |
| A cost or cardinality estimate | `kyber/stats/` (cardinality) or `kyber/expr_cost/` (per-row Expr cost) | `add-kyber-optimizer-pass` |
| An IO format or connector | `io/formats/<family>/`, registered as `SourceFormat`/`SinkFormat` | — |
| A spill / memory / backpressure decision | `carbonite/` (policy) + the Rust operator's spill path | — |
| Distributed scheduling of an existing operator | `dist/executors/` — **never** a second semantics | `add-distributed-operator` |
| A new execution tier (JIT/GPU/…) | a `core` `Executor` strategy, not a call-site branch | — |
| A shared plan node, expr, schema, or IR tag | `plan/` (neutral — everyone may import it) | — |
| A user-facing ML/inference surface | `ml/` (layer 6, built *on* the public API) | — |

If the answer is not here and not derivable from a package one-liner, that is a
signal to **ask**, not to invent a home. A misplaced module is how a layer contract
breaks.

## Confusable name clusters

These are the names that collide. Each cluster is genuinely separated — by layer,
by verb, or by exactness — and the table is the separation, so you do not have to
open seven files to rediscover it.

### The seven "stats"

Split by the contract loop's verbs: **Core measures, Kyber decides, Carbonite protects.**

| Path | Layer | Verb |
|---|---|---|
| `plan/stats.py` | 1 | The neutral statistics **algebra** everyone speaks. |
| `plan/source_stats.py` | 1 | The **declaration** a connector makes about a source. |
| `metadata/source_stats_store.py` | 1 | The **durable memory** of stats across queries. |
| `io/stats/` | 2 | **Extracts** cheap stats from connector metadata (footers, catalogs). |
| `core/stats.py` | 3 | **Measures** column statistics from real execution. |
| `kyber/stats/` | 3 | **Estimates** cardinality and column stats to make a decision. |
| `api/source_stats.py` | 5 | **Collects** per-source stats on the conductor's behalf. |

Rule of thumb: producing a number from data → `core`/`io`; consuming one to choose a
plan → `kyber`; the type that carries it → `plan`.

### The four metadata-answer paths

All answer a query without scanning; they differ by *what* they can prove and *who asks*.

| Path | Layer | Answers |
|---|---|---|
| `kyber/shortcuts/` | 3 | The `Facts` provable from a plan, plus the pure derivations over them. |
| `kyber/metadata_filter_count/` | 3 | Filtered counts specifically. Split out only to fit `kyber/`'s file budget. |
| `kyber/metadata_summary/` | 3 | Per-column summaries. Same split, same reason. |
| `kyber/metadata_answer.py` | 3 | Decides *whether* a terminal is provable at all. |
| `api/terminal/metadata_answer/` | 5 | The conductor **using** the above to skip execution. |
| `api/dataset/meta/` | 5 | The user-facing `ds.meta` accessor over all of it. |

Every exact answer returns `None` when it cannot be proved, and `None` means *execute*.
A shortcut is only ever an optimization — never a semantic.

### Families that span crates

Three families are not where a crate doc would lead you. Follow these, not intuition:

- **Sorting** lives in `bc-interp/src/ops/` (`radix_sort`, `sample_sort`, `byte_sort`,
  `external_sort`), **not** in `bc-runtime`. The one *reading* of a byte-lexicographic key
  that the sort and the range partitioner share is `bc-runtime/src/byte_key.rs`.
- **Window functions** split: kernels in `bc-runtime/src/window*.rs`, out-of-core
  execution in `bc-interp/src/window_spill.rs`.
- **Spilling** is per-operator, not one module: `bc-runtime/agg/spill.rs`,
  `bc-interp/ops/{mixed_spill, quantile_spill/, external_sort}.rs`,
  `bc-interp/window_spill.rs`, with the budget owned by `bc-resource` and the policy by
  Python `carbonite/`.
- **Filter / project / limit** have no file of their own; they are arms inside
  `bc-interp/src/ops/mod.rs`.

### Same name, different job

| Pair | Difference |
|---|---|
| `bc-io/src/store.rs` vs `bc-transport/src/store.rs` | Object-store URI resolution vs. shuffle-ticket registry. Unrelated. |
| `bc-expr/src/analyze.rs` vs `bc-codegen/src/analyze.rs` | Static answers *about* an `Expr` (cost, columns read, can-a-skipped-row-hide-an-error, contains-media-decode) vs. JIT-subset validation. |
| `bc-expr/src/select.rs` vs `bc-interp/src/ops/mod.rs` | Computing a filter's keep mask (short-circuiting the `AND` conjuncts) vs. the Filter operator that gathers with it. |
| `bc-sketches` `countmin` vs `frequent` | *How often is this key* vs. *which keys are heavy*. |
| `minhash` (`eval/str/`) vs `simhash` (`eval/list_ops/`) | Jaccard over shingles vs. cosine over embeddings. |
| `plan/expr_rewrite/` vs `kyber/rules/` | The traversal **mechanism** vs. the rewrite **policy**. |
| `api/merge/` | SQL `MERGE INTO` (upsert) — *not* joining two datasets, and *not* the row-level upsert a database performs (`ds.write.sql(mode="upsert")`). This one rewrites data files. |
| `io/stats/sql_catalog/probes.py::dialect_for_driver` vs `io/formats/sql/dbapi/_statements.py::dialect_for_driver` | Two vocabularies for two questions. The first maps *any* driver or scheme name to a **catalog** dialect key (`"postgres"`), by substring, to pick which system-catalog query to run. The second maps a **PEP 249 driver module** to a `uri` **scheme** (`"postgresql"`), exactly, to pick identifier quoting and an upsert spelling. Neither answers the other's question. |
| `io/formats/sql/dbapi/sink.py::WRITE_MODES` vs `io/formats/nosql/base.py::STORE_WRITE_MODES` | The same vocabulary deliberately, minus the two forms that need a statement engine. A sink narrows it further in `supported_modes` and declines the rest by name. |
| `carbonite/transfer/` | The shuffle engine. |
| `dist/spill_breakers/` | Out-of-core sort/join/window ("breaker" = pipeline breaker). |

### Names that read as generic but are not

`kyber/shortcuts` = no-scan answers · `api/orchestration` = the one Kyber→Carbonite→Core
loop · `api/tuning` = *learned* adaptive decisions (not user knobs — those are `config/`)
· `bc-py/src/process.rs` = process-wide singletons · `bc-expr/eval/str/chunk.rs` = the RAG
text splitter · `bc-expr/eval/generate.rs` = `sequence`/`range`, not codegen ·
`bc-runtime/agg/fused.rs` = one pass over `group_ids`, not operator fusion ·
`io/formats/sql/routing.py` = which *backend* serves a call (ADBC / ConnectorX / PEP 249),
not which node runs it ·
`bc-runtime/keys.rs` = the canonical key encoding every hash path must agree on ·
`bc-arrow/float_ident.rs` = the NaN/±0 identity contract for keys.

## Keeping context small

The engine is larger than any one context window, so treat reading as a budget.

1. **Map first, file second.** Grep `MAP.md`; open only what it names.
2. **Prefer the docstring to the body.** Every module states its own responsibility in
   its first line — that is usually the whole answer.
3. **The oracle is small on purpose.** To learn what an operator *means*, read
   `bc-interp::execute` (the sequential reference), not `par.rs` (2,600 lines of
   scheduling that computes the same thing).
4. **Do not read generated or vendored output.** `MAP.md` itself is generated —
   regenerate with `just map`, never hand-edit.
5. **Trailing `#[cfg(test)]` is most of some Rust files.** `bc-transport/src/lib.rs` is
   92% tests; the line counts in this map already exclude them, so a small number here
   means a small file to read.
