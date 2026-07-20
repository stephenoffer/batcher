# Working with coding agents

This page catalogs the agent skills Batcher ships, and explains how a coding agent
picks one and how you use them in your own project.

Batcher ships a set of *agent skills*, which are task-scoped instruction files that
teach a coding agent, whether Claude Code or anything else that reads them, how to use
this engine correctly. They live in `.claude/skills/<name>/SKILL.md` in the repository.

A skill isn't documentation for you to read start to finish, but this guide is. A skill
is a procedure an agent loads when it recognizes the task: how to port a PySpark job,
how to triage a query that returns wrong rows, or how to add an IO format without
breaking the layer contract. Each one carries the API surface it needs, the traps that
surface hides, and, for anything that changes results, the way to verify the work rather
than assume it.

Every skill was written against the live API and its examples were executed, so a name
that appears in one is a name that exists.

## Why these exist

An agent pointed at an unfamiliar engine fails in predictable ways. It invents plausible
method names, assumes eager execution, writes a per-row Python loop where the whole
design depends on batches, and reports success without checking the result. The skills
below exist to pre-empt exactly those failures, and each names the verification step
that catches them.

## Using the engine

These skills cover working *with* Batcher: writing pipelines, moving data, and
diagnosing a query that misbehaves.

| Skill | Use it when |
|---|---|
| `write-a-batcher-pipeline` | The default. Writing or reviewing any relational pipeline: read → transform → write, expressions, joins, aggregations, windows, batch UDFs. |
| `read-and-write-data` | Choosing a reader or writer, wiring object storage, or debugging a format, schema, path, or credential problem at the IO boundary. |
| `manage-a-lakehouse-table` | Delta/Iceberg/Hudi lifecycle: upserts via `MERGE INTO`, slowly-changing dimensions, CDC, backfills, time travel, compaction. |
| `write-a-streaming-pipeline` | The source is unbounded, or the query uses a trigger, checkpoint, or watermark and must run continuously. |
| `build-an-ml-pipeline` | Batch inference, embeddings and vector search, multimodal decode, preprocessors, or feeding a training loop. |
| `validate-data-quality` | Asserting quality contracts with `ds.dq`, choosing between fail/drop/quarantine, or profiling columns before writing checks. |
| `apply-governance-and-security` | A pipeline must restrict who reads which rows or columns, protect PII, produce an audit trail, or trace a sensitive column. |
| `run-a-distributed-job` | Taking a working single-node pipeline to a Ray cluster, sizing it, or debugging a distributed run. |
| `debug-a-batcher-query` | A query raises, hangs, OOMs, or returns wrong rows. |
| `optimize-a-slow-query` | A query returns the *right* answer too slowly. |

## Migrating to Batcher

One skill per source system. Each carries a verb-by-verb translation table, the concept
shifts that actually bite, and a porting recipe that ends by proving the ported script
returns the same rows as the original.

| Skill | Source |
|---|---|
| `migrate-from-spark` | PySpark |
| `migrate-from-polars-or-pandas` | Polars, pandas, and DataFrame-style code generally |
| `migrate-from-duckdb-sql` | DuckDB and SQL. Also covers *writing* new SQL against Batcher |
| `migrate-from-ray-data` | Ray Data |
| `migrate-from-daft` | Daft, especially multimodal and batch-inference workloads |
| `migrate-from-a-sql-warehouse` | A SQL warehouse or JDBC extract: Spark JDBC, pandas `read_sql`, SQLAlchemy, DB-API |

The narrative version of these tables is {doc}`../migration/index`; the skills add the
failure modes and the verification procedure.

## Extending the engine

For work *on* Batcher rather than *with* it. These encode the invariants in
`CLAUDE.md` and `.claude/rules/`, so a change lands across every layer it has to touch.

| Skill | Use it when |
|---|---|
| `add-relational-operator` | Adding or extending a relational operator across Rust IR, interpreter, runtime, parallel/distributed paths, and the Python surface. |
| `add-expression-or-function` | Adding a scalar/aggregate function, an expression IR node, or a typed-accessor method. |
| `add-kyber-optimizer-pass` | Adding an optimizer rule, cost model, or cardinality estimate. |
| `add-distributed-operator` | Wiring an operator through the distributed path so a multi-node result equals single-node. |
| `add-an-io-format-or-connector` | Adding a reader/writer for a file format, lakehouse table, database, or streaming source. |
| `run-quality-gate` | Before committing, opening a PR, or claiming a change works. |

Documentation is part of the engine, so it has its own pair. The contract they apply is
`.claude/rules/documentation.md`.

| Skill | Use it when |
|---|---|
| `improve-a-docs-page` | Writing, rewriting, or reviewing one page under `docs/`: hierarchy, voice, executed code blocks, links, tables, and visuals. |
| `audit-docs-structure` | Reorganizing `docs/`, adding a section, or diagnosing why readers can't find something. Produces a restructuring plan, not edits. |

{doc}`../internals/extending` is the companion contributor cookbook, holding the recipes
and the registries. The skills are the procedures and the gates.

## How an agent picks one

Agents select a skill by matching the task against its `description`, which states both
what it covers and when to invoke it. In Claude Code the skills in this repository are
discovered automatically when the agent is working inside it, and a user can also request
one by name:

```text
/write-a-streaming-pipeline
```

Skills compose. Porting a Spark streaming job that lands in Delta legitimately pulls in
`migrate-from-spark`, `write-a-streaming-pipeline`, and `manage-a-lakehouse-table`. The
routing section at the top of `write-a-batcher-pipeline` exists to hand off to the right
one rather than answer from a surface it doesn't cover.

## Using them in your own project

The skills are part of the repository rather than the wheel, so `pip install batcher`
doesn't install them. They're agent instructions rather than importable code. To use
them in a project that depends on Batcher, copy the directory into your own project's
skill folder:

```console
$ git clone https://github.com/stephenoffer/batcher /tmp/batcher
$ mkdir -p .claude/skills
$ cp -r /tmp/batcher/.claude/skills/* .claude/skills/
```

The usage and migration skills apply anywhere Batcher is installed. The extension skills
(`add-*`, `run-quality-gate`) assume you are inside the Batcher source tree and reference
`just` recipes that only exist there.

## Keeping them honest

A skill that describes an API which has since changed is worse than no skill, because an
agent will trust it. Two things guard against that:

- Every skill was written against the live API rather than from memory, with symbols
  verified by introspection and code blocks executed.
- `tests/docs/test_skill_coverage.py` fails if a skill exists that this page does not
  list, if a listed skill has no file, or if a skill is missing its `name`/`description`
  frontmatter. The catalog above cannot silently fall behind the directory.

When you change an API, the skill that teaches it is part of the change, exactly as its
documentation is.

## See also

- {doc}`../migration/index`: the mapping tables the migration skills build on.
- {doc}`../internals/extending`: the contributor cookbook behind the `add-*` skills.
- {doc}`../user-guide/index`: the task-oriented guides the usage skills point into.
- {doc}`../user-guide/troubleshooting`: the human-facing companion to
  `debug-a-batcher-query`.
