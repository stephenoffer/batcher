# Rule: Documentation

`docs/` is the published Batcher documentation: a Sphinx site (MyST Markdown, Furo theme)
built by `just docs`. This rule is the full reference for authoring and restructuring it.
The always-loaded contract in `CLAUDE.md` names the gates; this file says what "good" is.

**Rule priority.** When rules conflict: (1) this file, (2) `CLAUDE.md` and the other
`.claude/rules/` files, (3) the Google developer documentation style guide as a general
fallback.

## The one rule that outranks style

**Never fabricate a technical detail.** Every claim needs a verifiable source: an existing
page, code you read, a benchmark you ran, or context the user gave you. Don't extrapolate
an implementation from a feature name, invent a metric or a version cutoff, or assume an
architecture you didn't check. This matters more here than in most projects, because
`tests/docs/test_doc_examples.py` executes the code but nothing executes the prose.

**When in doubt, leave it out.** Incomplete but accurate beats complete but partly invented.

Competitive claims have a second gate: `docs/internals/competitive_architecture.md` is the
code-checked scorecard. Read it before writing that Batcher beats anything, and never
restore a claim it retires.

## Page architecture

### Information hierarchy

Every page moves in one direction, from *what* to *how* to *reference*:

1. What is this? (definition, concept)
1. How does it work? (behavior, implementation)
1. How do I use it? (procedure, configuration)
1. Reference detail (options, API, limits)

Define before describing. A page that opens with configuration before saying what the
thing is has the order backwards.

### Page openings

Open with one or two sentences that say what the page covers: "This page describes how
to ...", "This guide covers ...". Then go straight into the first real section.

Don't bullet-preview the visible headings. The right-hand table of contents already does
that, and needing the preview usually means the page is too dense to be one page.

### Headings

- Sentence case at every level.
- Avoid going past H3. Reaching H4, and certainly H5, is the signal to split into
  sub-pages.
- Question headings for conceptual sections ("What is a morsel?", "How does spilling
  work?"); statement or imperative headings for tasks and reference ("Configure the
  buffer pool", "Supported formats").
- **Never nest heading levels with no prose between them.** An H2 that only introduces
  H3s must carry a lead-in paragraph explaining what the group is.

### Files and directories

- One topic per page.
- Every directory has an `index.md` that introduces its children and carries their
  `{toctree}`. If a directory doesn't warrant an index, the content shouldn't be nested.
- Prefer `index.md` over a legacy `overview.md`; fold the latter in when you touch it.
- Adding, moving, or removing a page means editing the parent `{toctree}` in the same
  change. `tests/docs/test_docs_structure.py` fails on an orphan or a dangling entry, and
  the `-W` Sphinx build fails again at the end.
- Toctree ordering: orientation first, most-used content next, reference and advanced
  topics last.
- A page that is deliberately not published (a contributor RFC, a working ledger, a
  PDF-only paper) goes in `exclude_patterns` in `docs/conf.py` with a comment saying why.
  Follow the existing comment style there.

### When to restructure rather than edit

- H4 or H5 appears → split into sub-pages.
- The page needs an opening bullet preview to be navigable → split.
- Several sections answer the same reader goal → consolidate into one use-case narrative.
- A parent page carries content *and* nests children → move the content down or the
  children up.

## Voice and style

Write in second person, active voice, present tense, short sentences.

- Use common contractions: don't, can't, won't, you're, it's.
- "such as", not "like", for examples.
- Avoid "we", "our", "let's". Say "Batcher" or address the reader directly.
- Italicize a term on first use, then use it plainly.
- Avoid dashes, parentheticals, and semicolons in prose. Restructure into separate
  sentences instead.
- Cut filler: "simply", "just", "easily", "of course", "note that", "in order to".
- Avoid timeless-term rot: "currently", "new", "recently", "as of today". State the
  behavior, not its novelty.
- No marketing. A benefits list is a smell; state the recommendation directly or fold the
  point into prose.

Vary sentence and paragraph shape. Use prose, not numbered lists, for architecture and
"how it works" content. Lists are for enumeration, not explanation.

### Capitalization and terms

Batcher is always capitalized. The subsystem names are capitalized as proper nouns
(Kyber, Carbonite, Core, Arrow, Ray, Cranelift); common nouns are not (optimizer,
executor, morsel, control plane, data plane). Crate names stay lowercase in code spans:
`bc-runtime`, `bc-expr`. Spell out "Google Cloud", not "GCP".

## Procedures

Prerequisites go up front, before step one, never discovered mid-procedure.

Lead into a procedure with a grammatically complete sentence ending in "the following"
or "any of the following", then a colon: "To enable spilling, complete the following
steps:".

Use `1.` for every entry in an ordered list and let the renderer number them, so
reordering a step never means renumbering the list. End list items with a period unless
they're brief noun phrases.

Every procedure ends in a verifiable state. Say what the reader should see, and prefer
showing the actual output over describing it.

## Code examples

Code in `docs/` is executed, so it is a contract, not an illustration.
`tests/docs/test_doc_examples.py` extracts every fenced `python` block and runs it:

- Blocks run in document order sharing one namespace per page, so a page may open with a
  setup block and build on it.
- In the user-facing sections every `python` block runs by default. `# docs: skip` as the
  first line shows a block without executing it. Use it for examples needing a cloud store, a
  cluster, a GPU, or a real model.
- Under `docs/architecture/` and `docs/internals/` blocks are illustrative and don't run
  unless the first line is `# docs: run`.
- Examples must be self-contained per page and use in-memory data (`bt.from_pydict`), so
  no fixture files are needed.

Docstring examples in the public API are a separate contract: `.. doctest::` blocks under
an `Examples:` heading, executed by the doctest builder in `just docs`. See
`.claude/rules/python-quality.md` for the docstring style gate.

Other conventions: fence every block with a language tag (`python`, `bash`, `console`,
`sql`, `json`, `text`). No `$` prompts inside `bash` blocks. Inline code in single
backticks. Placeholders in angle brackets, `<your-bucket>`, and explained in the prose
around the block.

No placeholder prose. A "TBD" or "INSERT LINK" in `docs/` is a bug.

## Markdown and MyST

Use plain Markdown wherever it suffices. The site enables `colon_fence`, `deflist`,
`tasklist`, and `attrs_inline`, plus `sphinx_design` and `sphinx_copybutton`.

Admonitions carry severity, so pick the right one rather than bolding a **Note:** inline:

| Directive | Use it for |
|---|---|
| `{tip}` | An optional improvement the reader can take or leave. |
| `{note}` | Context worth knowing that doesn't change what they do. |
| `{important}` | Something they must know to succeed. |
| `{warning}` | An action that can lose data, cost money, or break a running job. |

Never use Unicode punctuation in source: straight quotes only, no curly quotes and no
em-dashes. Preserve author HTML comments exactly; never refactor or delete one.

Soft-wrap prose: one logical line per paragraph and per list item, and let the editor
wrap the display. Don't hard-wrap inside a paragraph.

## Links and cross-references

Prefer Sphinx roles over raw paths, because they break loudly under `-W` when a target
moves:

- `{doc}` for a page: `` {doc}`../user-guide/joins` ``.
- `{ref}` for a labeled section.
- `{py:class}`, `{py:func}`, `{py:meth}` for API objects, so the link tracks the
  autodoc target.
- Bare Markdown links only for external URLs.

Cross-reference generously and in both directions. When you move or rename a page, update
every inbound reference in the same change. End substantial pages with a "See also" list
pointing at the neighbours a reader would want next.

## Tables

Introduce every table with prose that says what the reader is looking at, and always
include a header row. Tables are for reference data and topic-to-page maps: option name
against meaning, source system against Batcher equivalent, symptom against cause. They
are not for explanation, which belongs in prose.

Keep cells to a single line where you can. Use `{list-table}` when cells need code
blocks, multiple paragraphs, or nested lists, and set `:header-rows: 1`. Order rows the
way a reader scans them, alphabetically for lookup and by importance for guidance, and
say which you chose if it isn't obvious.

Don't build a table with one data column. That's a list.

## Visuals

A diagram earns its place when it shows something prose can't hold in a reader's head:
a topology, a flow with branches, a state machine, a layered stack, a before-and-after.
Sequential steps are a numbered list. A pair of values is a sentence.

Every visual needs a lead-in sentence and must be readable as a supplement, never as the
only carrier of a fact. Anyone reading with images off, or with a screen reader, must
still get the whole page.

### Architecture diagrams

The diagram source of truth is SVG in `docs/_static/diagrams/`, and the SVG is what the
pages embed: it stays sharp at any zoom and carries its own `prefers-color-scheme` block,
so it restyles to the reader's theme instead of sitting on it as a bright slab.

The authoring scripts live in `tools/diagrams/`, deliberately **not** beside the SVGs:
Sphinx copies `html_static_path` wholesale, so anything in that directory ships as a
published website asset. `exclude_patterns` does not filter static files. The workflow:

1. Edit or add the `*.svg` in `docs/_static/diagrams/`, by hand or by writing a small
   script against `tools/diagrams/_authoring.py`, which holds the visual language.
1. Embed the SVG. `just diagrams` (`python tools/diagrams/render.py`) rasterizes to PNG
   for contexts that cannot take SVG, such as a slide deck; it is not part of the docs
   build, and `rsvg-convert` is not required to work on the documentation.
1. Commit the SVG, and the generating script if you wrote one.

Match the established visual language: blue `#2563eb` for the primary subject, amber
`#f59e0b` for the accent or the highlighted path, slate `#1e293b` for text, gradient
cards with soft shadows, and labeled flow arrows. A new diagram that invents its own
palette makes the set look like several documents. Read `docs/_static/diagrams/README.md`
before adding one.

Design constraints that matter more than they look:

- Label the arrows. An unlabeled arrow between two boxes says "related", which the reader
  already knew.
- Cap it at roughly seven boxes. Past that, split into two diagrams at different zoom
  levels.
- Don't encode meaning in color alone. Color-blind readers and greyscale printouts need
  the shape, the label, or the position to carry it too.
- Text in a diagram must survive being scaled to a phone column. If a label needs a
  sentence, it belongs in the caption.
- The diagram states the same thing the prose does. A diagram that contradicts the text
  around it is worse than no diagram.

### Alt text and captions

Every image needs alt text that conveys the *information*, not the *appearance*. Compare
`![architecture diagram](...)` against the pattern the landing page already uses: a full
sentence naming the sources, the engine, and the outputs. Write the sentence you'd say
out loud to someone who asked what the picture shows.

Use a caption (via `{figure}`) when the image needs a visible label or a reference. Use
plain `![alt](path)` when it sits inline with the prose that already introduces it.

### Screenshots

Prefer text and code to a screenshot. A screenshot of a terminal should be a `console`
block instead: it's searchable, copyable, and it doesn't rot silently.

When a UI genuinely needs one, crop to the relevant region, don't paste a whole desktop,
never include credentials, tokens, real account identifiers, or customer data, and say in
the prose what the reader should look at. Screenshots age fastest of any asset, so treat
one as a maintenance commitment.

### Charts and benchmark figures

A performance chart must be reproducible. Name the script under `benchmarks/` that
produced it, the scale factor, and the hardware, or don't publish the number. Axes are
labeled with units, the baseline is identified, and the y-axis starts at zero unless you
say plainly why it doesn't.

Benchmark results live in `benchmarks/BENCHMARK_RESULTS.md` and are cited from the docs.
Don't restate a timing in `docs/` that no committed benchmark produces.

### Multimodal assets

The ML and multimodal pages document image, audio, and video pipelines, which invites
sample media. Keep the repository light: prefer generating a sample in the example code
over committing a media file, keep any committed asset small, and use only content that
is unambiguously licensed for redistribution. Describe audio and video content in prose,
because a reader who can't play it still needs the point.

## Content development

### Rewrite or patch?

Check `git log` on the file first. A page warrants a full rewrite when it's marketing
rather than technical, has no conceptual anchor, or no longer matches the engine. A page
that is structurally sound but stale warrants targeted edits. Judge per file, and say
which you chose and why in the commit message.

### Ending sections

Collect constraints at the end under "Requirements and limitations" or "Known issues"
rather than scattering caveats through the page. A reader deciding whether to use
something wants the limits in one place.

## Gate before "done"

- `just docs`. The doctest builder plus `sphinx-build -E -W --keep-going`. Warnings are
  errors, so a broken `{doc}` reference or an orphan page fails the build.
- `just test-py`. Runs `tests/docs/`, which executes every fenced `python` block,
  checks toctree reachability, and enforces the API-coverage and skill-catalog contracts.
- `just lint-guardrails`. Every repo path named in agent-facing guidance must exist.
- `just diagrams` after touching any SVG, and commit the regenerated PNG.
- Public API changes additionally need `just lint-docstrings`.
