---
name: audit-docs-structure
description: Audit the information architecture of the whole Batcher docs site: section boundaries, toctree wiring and ordering, orphan and oversized pages, duplicate coverage, reader journeys, and the Sphinx configuration. Produces a prioritized restructuring plan. Invoke when reorganizing docs/, adding a new section, or diagnosing why readers can't find something.
---

# Audit the docs structure

This skill looks at the site, not the page. It answers: are the sections the right
sections, is every page reachable from the one place a reader would look, and does a
newcomer's path through the docs actually exist? For rewriting an individual page, use
`improve-a-docs-page`.

Read `.claude/rules/documentation.md` first for the structural contract.

## Gather the facts before judging

Run these and keep the output; the audit is an argument from evidence, not taste.

```bash
# Section sizes: which parts of the site are load-bearing
for d in docs/*/; do echo "$(find "$d" -name '*.md' | wc -l) $d"; done | sort -rn

# Page sizes: candidates for splitting (Python source caps at 500 lines; prose
# has no hard cap, but a page past ~400 lines is usually two pages)
find docs -name '*.md' -not -path '*_build*' | xargs wc -l | sort -rn | head -25

# Orphans and dangling toctree entries, in seconds rather than a full build
pytest tests/docs/test_docs_structure.py -q

# Deep headings: an H4 anywhere is a split signal
rg -n '^#### ' docs/

# Directories with no index page
for d in $(find docs -mindepth 1 -type d -not -path '*_build*' -not -path '*_static*'); do
  test -f "$d/index.md" || echo "no index: $d"
done
```

For duplicate coverage, search for the same concept across sections: `rg -l 'spill' docs/`
and read the openings of what comes back. Two pages that both define a term is the signal.

## What to look for

**Section boundaries.** Each top-level directory should answer a distinct reader
question. Getting started orients, user guide teaches tasks, API reference looks up,
architecture and internals explain design, examples and tutorials demonstrate, migration
translates, benchmarks substantiate. A page in the wrong section is invisible even when
it is excellent. Name the section a misplaced page belongs in and why.

**Toctree wiring.** Every directory carries a `{toctree}` on its `index.md`, listing its
children in reader order: orientation first, most-used next, reference and advanced last.
Alphabetical ordering is almost always wrong. A section whose index doesn't introduce its
children in prose is a bare list of links.

**Orphans and near-orphans.** A true orphan fails `test_docs_structure.py`. The subtler
case is a page reachable only from the toctree and linked from nothing: technically
wired, practically undiscoverable. Cross-link it from the pages whose readers need it.

**Oversized and undersized pages.** A page past roughly 400 lines, or one that reaches
H4, usually wants splitting along its own H2 seams. The reverse is also a finding: three
short pages that each say one paragraph want to be one page with three sections.

**Duplicate and contradictory coverage.** When two pages define the same concept, pick
the canonical home, reduce the other to a sentence and a `{doc}` link, and check they
don't disagree. Contradiction between two pages is a correctness bug, not a style one.

**Reader journeys.** Trace at least three end to end and confirm each hop exists:

- New user: landing page → install → quickstart → first real pipeline.
- Migrating user: landing page → migration index → the source-system page → the user
  guide it hands off to.
- Debugging user: symptom → troubleshooting → the specific guide → the API entry.

A journey that requires the reader to already know a page name is a broken journey.

**Configuration drift.** `docs/conf.py` carries `exclude_patterns` for deliberate
non-pages, each with a comment saying why. Check that every exclusion is still true and
that nothing published should be excluded. Check the extension list still matches the
directives the pages actually use.

## Produce a plan, not a pile of edits

Restructuring is where docs work does the most damage if it's done impulsively: every
move breaks inbound links, toctrees, and anything citing the path. So the output of this
skill is a written plan, ordered by leverage, that a human approves before it runs.

For each proposed change, record the current path, the proposed path, why, and what has
to move with it. Then execute in this order, because each step depends on the last:

1. Create or fix section `index.md` pages and their toctrees. Cheap and reversible.
1. Move pages, one section at a time, updating the toctree in the same commit.
1. Update every inbound reference. `rg` for the old path across `docs/`, and also across
   `.claude/` and `python/batcher/` docstrings, which cite doc pages too.
1. Split or merge pages, applying `improve-a-docs-page` to each result.
1. Re-run the gate and re-trace the reader journeys.

Batcher's docs have no redirect layer, so a moved page is a broken external link. Move a
published page only when the gain is real, and prefer fixing the toctree and the
cross-links over relocating the file.

## Verify

```bash
pytest tests/docs/test_docs_structure.py -q   # fast: orphans and dangling entries
just docs                                     # -W: broken {doc} refs fail the build
just test-py                                  # full docs suite incl. executed examples
```

A green build proves the wiring. It proves nothing about whether the structure is good,
so re-trace the reader journeys by hand before calling the audit done.

## Report

Separate the two kinds of finding, because they need different responses:

- **Objective problems**: orphans, dangling entries, missing indexes, H4 nesting,
  contradictions between pages, dead paths. Fix these.
- **Judgment calls**: section boundaries, whether to split, what the canonical home for a
  concept is, ordering within a section. Propose these with a recommendation and let a
  human decide.

Say explicitly what you did not examine. A partial audit reported as complete is worse
than no audit, because the unexamined half looks cleared.
