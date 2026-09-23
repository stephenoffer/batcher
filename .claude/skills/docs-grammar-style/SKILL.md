---
name: docs-grammar-style
description: The sentence-level style reference for Batcher prose: which rendering surface a file is on and what syntax survives there, word substitutions and filler, list and procedure shape, table and link conventions, admonition severity, capitalization and the project glossary, and how to write a number so a reader can check it. Invoke while writing or editing any Markdown in the repo, or when you need to settle a specific wording, link, or formatting question rather than audit a whole page.
---

# Docs grammar and style

`.claude/rules/documentation.md` is the contract and says what good looks like.
`improve-a-docs-page` is the procedure for one page, `audit-docs-structure` the one for
the site. This file is the lookup table underneath all three: the sentence, the link, the
table row, the term. Reach for it when you know what you want to say and need to know how
this repo says it.

Two things outrank everything below. Never fabricate a technical detail, and check
`docs/architecture/internals/competitive_architecture.md` before writing any competitive
claim. Both are in the contract; they are repeated here because they are the only rules
whose violation ships a lie rather than an awkward sentence.

## Know which surface you are writing for

The repo has two rendering surfaces and they accept different syntax. Markdown that reads
correctly on one degrades silently on the other, which is the failure mode to watch: no
error, no build break, just a directive sitting on the page as literal text.

`docs/` renders through Sphinx with MyST. Roles (`{doc}`, `{ref}`, `{py:class}`),
directive fences (```` ```{note} ````), colon fences, `deflist`, `tasklist`, and
`attrs_inline` all work, because `docs/conf.py` enables them. Everything else in the
repo renders in GitHub's Markdown viewer, which has no MyST pipeline: `CLAUDE.md`,
`README.md`, `RELEASING.md`, `MAP.md`, `.claude/rules/*.md`, `.claude/skills/**/SKILL.md`,
`benchmarks/*.md` and the result files under `benchmarks/results/`, and the per-package
`CLAUDE.md` files in `python/batcher/` and `crates/`.

Outside `docs/`, substitute:

| Instead of | Write |
|---|---|
| `` {doc}`../user-guide/analyze/sql` `` | a relative Markdown link, or the plain path in a code span |
| `` {py:func}`bt.col` `` | `` `bt.col()` `` in a code span |
| ```` ```{note} ```` and the other directive fences | a GitHub alert (`> [!NOTE]`, `> [!WARNING]`), or a blockquote opening with a bold label |
| `{list-table}` | a pipe table, escaping any literal `\|` inside a cell |
| `.. doctest::` and `# docs: run` markers | a plain fenced block; nothing outside `docs/` and the docstrings executes |

A MyST directive fence on GitHub renders as a code block titled with the directive name,
so the content survives but the emphasis is gone and the page looks broken. A `{doc}`
role renders as literal text with the braces showing.

## Voice

Second person, active voice, present tense, short sentences. Common contractions.
"such as" rather than "like" when introducing an example. Address the reader or name
Batcher; no "we", "our", or "let's". Italicize a term on first use and then use it plainly.

Avoid dashes, parentheticals, and semicolons in prose. Nearly every one of them is a
second sentence that has not been allowed to start. Restructure instead. (This one is
worth measuring rather than trusting: see `write-in-a-human-voice`, which counts them.)

Cut these on sight, either deleting the word or replacing it with the plain one:

| Cut or replace | Write |
|---|---|
| simply, just, easily, of course, note that, it's worth noting | nothing; delete the word and the sentence still says it |
| in order to | to |
| utilize, leverage | use |
| facilitate, enable (as a verb of convenience) | the verb that actually happens |
| robust, powerful, seamless, comprehensive, cutting-edge, best-in-class | a measured property, or nothing |
| delve into, dive into | the verb for what you do to it |
| a variety of, a number of | the number, or nothing |
| in the realm of, in the landscape of | where |
| allows you to, lets you | you can, or the imperative |

The list is a floor, not a specification. A word that survives it can still be empty.
The test that decides every case: delete the word and read the sentence. If it lost
nothing, it was filler.

Two more, because they rot rather than read badly. Timeless terms ("currently", "new",
"recently", "as of today") state the behavior's novelty instead of the behavior, and they
are wrong within a release. Marketing prose ("Batcher delivers blazing-fast...") makes a
claim no benchmark backs; a benefits list is the same failure with bullets.

## Lists

Use a list to enumerate. Use prose to explain. Architecture and how-it-works content is
prose, even when it has three parts, because the reader needs the connective tissue that
bullets delete.

Lead into a list with a grammatically complete sentence ending in "the following" or "any
of the following", then a colon. Number every entry in an ordered list `1.` and let the
renderer count, so reordering a step never means renumbering the list. End items with a
period unless they are brief noun phrases, and keep one style within a list.

Don't bullet-preview the headings the reader can already see in the table of contents.

## Procedures

Prerequisites go before step one, never discovered halfway down. Each step is one action.
The procedure ends in a state the reader can check, and showing the output beats describing
it:

````text
```console
$ python -c "import batcher as bt; print(bt.versions()['engine_profile'])"
release
```
````

## Tables

Every table gets a header row and a sentence of lead-in prose saying what the reader is
looking at. Tables hold reference data and topic-to-page maps: option against meaning,
source system against Batcher equivalent, symptom against cause. Explanation belongs in
prose, and a table with one data column is a list.

Keep cells to a line where you can. When a cell genuinely needs a code block or several
paragraphs, use `{list-table}` with `:header-rows: 1`. Order rows the way the reader
scans, alphabetically for lookup and by importance for guidance, and say which you chose
when it isn't obvious.

## Links

Prefer a Sphinx role over a raw path: roles break loudly under `-W` when a target moves,
and a hardcoded path breaks silently. Use `{doc}` for a page, `{ref}` for a labeled
section, and `{py:class}`, `{py:func}`, `{py:meth}` for an API object so the link tracks
the autodoc target. Bare Markdown links are for external URLs only.

Cross-reference in both directions. When a page is substantial, end it with a "See also"
list pointing at the neighbours its reader wants next. When you move or rename a page,
update every inbound reference in the same change, and search `.claude/` and the
docstrings under `python/batcher/` as well, both of which cite doc pages.

Batcher's docs have no redirect layer. A published page that moves is a dead external
link, so prefer fixing the toctree and the cross-links to relocating the file.

## Admonitions

Pick the level by what happens to the reader who ignores it, and never bold a **Note:**
inline instead. `{tip}` is an optional improvement, `{note}` is context that doesn't
change what they do, `{important}` is something they must know to succeed, and `{warning}`
is an action that can lose data, cost money, or break a running job. If you cannot tell
`{note}` from `{important}`, ask rather than guess: the difference is whether a reader who
skips it still succeeds.

## Code

Every fence carries a language tag: `python`, `bash`, `console`, `sql`, `json`, `text`.
No `$` prompts inside a `bash` block; use `console` when you want to show the prompt and
the output together. Placeholders go in angle brackets, `<your-bucket>`, and the prose
around the block says what to put there.

Code in `docs/` is executed, so it is a contract rather than an illustration.
`tests/docs/test_doc_examples.py` extracts every fenced `python` block and runs it,
sharing one namespace per page in document order. Keep examples self-contained per page
and build them from `bt.from_pydict` rather than a file on disk. A block that needs a
cloud store, a cluster, a GPU, or a real model carries `# docs: skip` as its first line.
Under the `docs/architecture/` overview pages and `docs/architecture/internals/`, blocks
are illustrative and don't run unless the first line is `# docs: run`;
`docs/architecture/deep-dives/` is not one of those and its blocks run by default.

Docstring examples are the other executed surface, governed by
`.claude/rules/python-quality.md`: `.. doctest::` under an `Examples:` heading, run by the
doctest builder in `just docs`, with `# doctest: +SKIP` on the `>>>` lines of an example
that needs an external resource.

## Capitalization and terms

Batcher is always capitalized. The subsystems are proper nouns: Kyber, Carbonite, Core,
Arrow, Ray, Cranelift. The things they are stay common nouns: optimizer, executor, morsel,
control plane, data plane, buffer pool. Crate names are lowercase inside code spans,
`bc-runtime`, `bc-expr`, `bc-interp`.

Spell the competition the way it spells itself: DuckDB, Polars, Apache Spark then Spark,
Daft, Apache Flink then Flink, Ray Data. Spell out "Google Cloud", never "GCP". Type and
API names keep their source casing in code spans: `RecordBatch`, `Expr`, `RelOp`,
`Dataset`, `LogicalPlan`.

## Numbers a reader can check

A number in the docs is a claim, and the reader's next question is where it came from.
Name the script under `benchmarks/` that produced it, the scale factor, and the hardware,
or leave it out. Results live in `benchmarks/BENCHMARK_RESULTS.md` and the docs cite them;
don't restate a timing that no committed benchmark produces. Axes get units, the baseline
is identified, and a y-axis that doesn't start at zero says why in the caption.

The same discipline applies to a ratio. "Faster than DuckDB" is not a claim, it is a mood.
"1.8x faster on TPC-H Q1 at sf10, single node" is one, and
`docs/architecture/internals/competitive_architecture.md` is the file that decides whether
it is currently true.

## What this doesn't govern

`docs/api/symbols/` renders generated signatures and docstrings, so its prose comes
from `python/batcher/` and is governed by the docstring gate in
`.claude/rules/python-quality.md` instead. The rules here apply to the docstrings
themselves, at their source.

Several pages under `docs/architecture/internals/` are excluded from the build in
`docs/conf.py`, each with a comment saying why. They are working records rather than
published pages. Accuracy still binds; page architecture does not.

## Check it

```bash
just docs              # doctest builder, then sphinx-build -E -W: a broken role fails
just test-py           # runs tests/docs/: executes every python block, checks toctrees
just lint-guardrails   # every repo path named in .claude/ guidance exists
just lint-docstrings   # the public-API docstring contract, if you touched one
```

`just docs` proves the links resolve and the examples run. It cannot see voice, a table
that should have been prose, or a number with no source, so read the page once more before
calling it done.
