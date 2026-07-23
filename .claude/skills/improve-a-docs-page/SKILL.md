---
name: improve-a-docs-page
description: Audit and improve a single page under docs/ against the Batcher documentation contract: information hierarchy, opening, headings, voice, executed code blocks, links, tables, and visuals. Verifies with the Sphinx build and the doc-example suite. Invoke when writing a new docs page, rewriting a stale one, or reviewing a docs change before it ships.
---

# Improve a docs page

One page at a time, in one pass, verified. For whole-site information architecture,
meaning which pages exist, where they live, and what the toctrees say, use `audit-docs-structure`
instead and come back here for the individual rewrites.

Read `.claude/rules/documentation.md` first. It is the contract; this skill is the
procedure for applying it.

## Before you edit

1. **Read the whole page.** Not a section. A page's problems are usually structural, and
   structure isn't visible from an excerpt.
1. **Check the history**: `git log --oneline -5 -- docs/<path>`. A page overhauled last
   month wants targeted fixes. A page untouched since the API changed may want a rewrite.
1. **Find the inbound links**: `rg 'user-guide/joins' docs/` for the page's own name.
   They tell you what readers were promised before they arrived, and they are what you
   must update if you move or rename anything.
1. **Verify the API the page teaches.** Every symbol it names must exist. Check against
   the source under `python/batcher/`, or import it and introspect. This is the single
   most common defect in a stale page.

Decide rewrite-or-patch explicitly before you start typing, and say which in the commit
message.

## The audit checklist

Work top to bottom. Each item is a question with a yes-or-no answer.

**Structure**

- Does the page cover exactly one topic?
- Does it move what → how → use → reference, defining before describing?
- Does it open with one or two sentences naming what it covers, with no bullet preview
  of the headings?
- Are all headings sentence case, and does it stay at H3 or above?
- Does every heading that only contains sub-headings have a lead-in paragraph?
- If it reaches H4, or needs a preview to navigate, does it want splitting?

**Prose**

- Second person, active voice, present tense, contractions?
- "such as" not "like"; no "we"/"our"/"let's"; no filler ("simply", "just", "note that")?
- No dashes, parentheticals, or semicolons doing the work of a second sentence?
- No timeless-term rot ("currently", "new", "recently")?
- Are benefits lists cut or folded into prose?
- Is explanation in prose rather than bullets, with lists reserved for enumeration?
- Straight quotes only, soft-wrapped paragraphs, author HTML comments untouched?

**Substance**

- Does every technical claim trace to code, a page, or a benchmark you actually checked?
- Do prerequisites appear before step one?
- Does each procedure end in a state the reader can verify?
- Are limitations collected at the end rather than scattered?
- Is there no placeholder text left?

**Code**

- Every fence tagged with a language, no `$` prompts in `bash` blocks?
- Are the `python` blocks self-contained per page, using `bt.from_pydict` rather than a
  file on disk, and do they build in document order on one shared namespace?
- Does anything needing a cloud store, cluster, GPU, or real model carry `# docs: skip`?
- Under `architecture/` or `internals/`, does anything meant to execute carry
  `# docs: run`?

**Links, tables, visuals**

- Internal references use `{doc}`, `{ref}`, or the `{py:...}` roles rather than raw paths?
- Is the page cross-linked in both directions, with a "See also" if it's substantial?
- Does each table have a header row and a lead-in sentence, and is it reference data
  rather than explanation in disguise?
- Does every image carry alt text that states the information, not the appearance?
- Would a diagram help here, and does an existing one in `docs/_static/diagrams/` already
  cover it?

## Adding or changing a visual

Full conventions are in the Visuals section of `.claude/rules/documentation.md`. The short
version:

1. Confirm the visual shows something prose can't hold: a topology, a branching flow, a
   layered stack, a state machine. Sequential steps are a list.
1. Edit or add the SVG in `docs/_static/diagrams/`, matching the existing palette (blue
   `#2563eb`, amber `#f59e0b`, slate `#1e293b`) and labeling every arrow. `tools/diagrams/_authoring.py`
   holds the visual language as functions, with the theme-aware defs already wired.
1. Embed the SVG with a full-sentence alt text and a lead-in line of prose.
1. Commit the SVG, and the generating script if you wrote one.

Prefer a `console` block over a terminal screenshot. Prefer a table over a diagram of a
table. Cap a diagram at roughly seven boxes.

## Verify

```bash
just docs        # doctest builder + sphinx-build -E -W: broken refs and orphans fail
just test-py     # runs tests/docs/: every python block executes, toctrees resolve
```

`just lint-guardrails` as well if you touched anything under `.claude/`. If you moved or
renamed a page, re-run `rg` for its old name across `docs/` and confirm nothing points at
it any more.

## Report honestly

Say what you changed and what you left. Two things are worth calling out explicitly
because a reviewer can't see them in the diff:

- **Claims you couldn't verify.** Name them and say you removed them, rather than
  softening the wording and leaving them in.
- **Structural problems you didn't fix.** A page that should be split, a section that
  belongs on a neighbour. Report it as a finding instead of half-doing it.

## Ask rather than assume

These are genuine judgment calls. Raise them; don't decide silently:

- A benefits list: cut, fold into prose, or keep?
- Prose, list, or table for a given block of content?
- A technical detail you can't source: confirm, replace, or omit?
- One page, a split, or a restructure around use cases?
- Admonition severity: tip, note, important, or warning?

Everything the style rules already settle, just apply. These five are the ones they don't.
