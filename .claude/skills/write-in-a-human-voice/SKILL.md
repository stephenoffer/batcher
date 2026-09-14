---
name: write-in-a-human-voice
description: Rewrite or draft Batcher prose so it does not read as generated: strip the assistant shape (heading every eighty words, a third of lines bulleted, bold on every term, a "Key takeaways" close), cut the sentences that say nothing, restore a sentence-length distribution with short punches and long runs, commit to a verdict, and fix diction last. Never alters a fact, a number, a code block, or a claim. Invoke when a docs page, docstring, benchmark write-up, rules file, or PR description reads robotic or templated, and when drafting one that should read human from the first pass.
---

# Write in a human voice

Adapted from the `human-voice` skill (github.com/stephenoffer/human-voice, MIT) and
narrowed to this repository's prose: pages under `docs/`, public API docstrings,
the agent-facing files under `.claude/`, the write-ups in `benchmarks/`, and commit and PR
messages. `.claude/rules/documentation.md` is the contract and outranks this file
everywhere they touch. `docs-grammar-style` settles the sentence. This skill is about the
shape of the whole thing, which is the part a style pass never reaches.

The job is not swapping words. Text reads as generated at four depths, and they are not
equally loud:

1. **Shape.** The document is a chat answer wearing a document's clothes: a heading every
   eighty words, a third of the lines bulleted, bold on every term, a "Key takeaways"
   close. Loudest signal, and the one most passes skip.
2. **Rhythm.** Every sentence collapsed into the 12-to-26-word band, no short punches, no
   long runs, every paragraph the same size.
3. **Substance.** Paragraphs that carry no information, a survey where a verdict belongs,
   specificity invented to fill a gap.
4. **Diction.** delve, leverage, seamless. Real, and least important.

Fix them in that order. A pass that starts at 4 ships a page that is still obviously
machine-written, which is what most "humanizer" tooling produces.

The evidence for the ordering is second-hand and worth stating as such: the upstream skill
reports that pretrained base models are classified human by commercial detectors more than
96% of the time while the same weights after instruction tuning are caught, which puts the
signal in post-training artifacts (markdown preference, response shape, sycophancy) rather
than in "machine-ness". It also cites Liang et al. (2023) finding that GPT detectors
systematically misclassify non-native-English writing as generated. Take both as a reason
to fix shape before diction, and as a reason never to treat any detector score as ground
truth about a person.

## What must not move

Three of this repo's invariants outrank every rhythm target below, and a rewrite that
trades one for a better cadence is a defect rather than a style call.

**Never fabricate a technical detail.** The contract's one rule that outranks style. A
flat sentence gets "fixed" with a vivid invented specific more often than by any other
route, so when a sentence is empty, delete it. Don't dress it. If the page needs a number
it doesn't have, mark the gap and say so in the report rather than filling it.

**The code is executed, so it is not prose.** `tests/docs/test_doc_examples.py` runs every
fenced `python` block in `docs/`, and `just docs` runs every `.. doctest::` in a public
docstring. A block is not available for rhythm work. Edit one only for a reason that
survives the test, and re-run.

**The competitive scorecard decides competitive claims.**
`docs/architecture/internals/competitive_architecture.md` is code-checked and retires
claims. Sharpening "Batcher is comparable on some shapes" into a verdict is exactly the
move this skill asks for, and it is only legitimate if the scorecard already says so.
`CLAUDE.md` is explicit that the engine loses to DuckDB single-node at sf100, has no
`StringView`, cannot express Flink's streaming guarantees, and buys several wall-clock
wins with 1.4x to 4.4x more CPU. Say "aiming to" and mean it.

Alongside those: every number, unit and date; every command, config key, file path and CLI
flag; every link and citation; every API name (check it exists before you keep it); and
every claim's strength. You may sharpen the wording of a claim. You may never strengthen,
weaken, or invent one.

## The surfaces, and what each one allows

"Human" is not one voice. Infer the surface from the file, apply the universal core, then
that surface's conventions. The universal core is fixed everywhere: no vacuity, no
fabrication, no reflexive rule-of-three, no bold-bullet listicle, no puffery, no vague
attribution, no uniform sentence length, one term per concept, and no "not X, it's Y".

| Surface | Voice | Allowed here | Still wrong |
|---|---|---|---|
| `docs/` user guide, tutorials, getting started | Second person, present, direct, contractions | "you", imperatives, numbered steps | "we"/"our", hype, a benefits list |
| `docs/api/` reference pages | Terse, declarative, lookup-shaped | A heading per name, dense tables | Prose padding around a signature |
| Public docstrings in `python/batcher/` | One-line summary, then the sections `lint-docstrings` requires | Google-style `Args:`/`Returns:`, a runnable `.. doctest::` | A multi-line first sentence, types in the prose |
| `docs/architecture/` and `internals/` | Explanatory prose, past tense for what was measured | Long paragraphs, argument, an admitted gap | Bullets standing in for reasoning |
| `.claude/rules/` and `.claude/skills/` | Instructional, specific, evidence-led | Second person, a named incident, a blunt "never" | Generic advice with no failure behind it |
| `benchmarks/*.md` | Measured, past tense, hardware named | Tables of numbers, a stated caveat | A ratio with no scale factor or machine |
| Commit messages and PR bodies | Imperative subject under ~72 characters, body explains why | A stated trade, a named limitation | A recap of the diff the reader can read |

Two rules survive every surface: never fabricate, and match the surface rather than faking
a voice it doesn't have. Bolted-on personality in a rules file reads as generated exactly
as much as stiff formality in a tutorial does.

## The tells, loudest first

The full catalog lives upstream in `references/ai-tells.md`. This is the tiering that
matters here.

### Tier 1: shape and rhythm

- **Assistant shape.** A heading every eighty words, a third of the lines bulleted, bold on
  every term, a "Key takeaways" or "In conclusion" recap. Reformat to what this surface
  looks like written by a person: a user-guide page has headings but not one per
  paragraph, an internals record is mostly prose, a reference page is mostly table. Measured
  across `docs/` on 2026-09-13, the median page carries 12.4 headings per 1,000 prose words
  and the worst carries 100. Re-measure rather than trusting that number.
- **Compressed sentence lengths.** Everything in the 12-to-26-word band, no short punches.
  Reach past both ends. One sentence in eight at eight words or fewer.
- **Even paragraphs.** Let a paragraph be one sentence when that is what the point needs.
- **Bold-lead-in bullets.** `- **Term:** ...` on every item. Convert some to prose; drop
  bold that decorates rather than distinguishes.
- **The five-paragraph mold.** An intro that previews, three even blocks, a recap. Let the
  structure follow the argument and end on the last real point.
- **Em-dashes.** The contract bans them in `docs/` source outright, and `docs/` carried
  1,914 across 124 of 497 pages when this was written, concentrated in
  `architecture/internals/`. Replace them, and *vary* the replacement: a comma here, a
  period there, a colon or a restructure elsewhere. Swapping every dash for the same mark
  installs a fresh uniform signature in place of the old one. One exception, and the
  measurement script now separates it out for you: a dash that is a table cell's whole
  content is a *glyph* meaning "not supported", not punctuation. `docs/index.md`'s support
  matrix holds 19 of them and its legend defines them. Breaking a table to satisfy a rule
  about sentences is the rule fighting the task.
- **Rule of three everywhere.** "fast, reliable, and scalable", and the noun-phrase kind
  too. Vary to two or four, or write a sentence.
- **Paste residue.** `oaicite`, `turn0search3`, `[cite: 4]`, an unfilled `[Your Name]`,
  `utm_source=chatgpt.com` on a link. These are proof rather than evidence: delete them.

The syntactic signature is the part that survives every diction fix, and it is what a
current model writes once the obvious slop is gone. Each construction is ordinary English,
so the tell is the stacking, never the instance:

- **Clefts.** "What actually consumed the time was the shuffle." Put the subject first.
- **Resultative participial tails.** ", making it easier to...", ", allowing the planner
  to...". A consequence bolted on to manufacture payoff, which also slips the claim past
  unexamined. Cutting one usually improves the argument.
- **Copula avoidance.** Nothing is allowed to *be* anything, so it *serves as*, *functions
  as*, *represents*. Say what the thing is.
- **Clause welding.** ", and it is...", ", but this means..." four times a page. End the
  sentence.

### Tier 2: substance and stance, which no regex sees

Vacuity is the highest tell in the list and the only tool that catches it is your read. A
paragraph you can delete with no information loss should be deleted. Fence-sitting is next:
"several approaches, each with tradeoffs" is what this repo's rules files never do, and the
reason they read as written by someone is that they commit. So: name the mechanism, weight
the lopsided trade honestly, lead with the verdict, and call a wrong choice wrong.

Also in this tier: meta-commentary ("This page will explore"), chatbot scaffolding ("Let's
break it down"), empty conclusions, fabricated specificity ("up to 40%" with no source),
vague attribution ("studies suggest"), telling instead of showing ("the implications are
significant"), hedge stacks ("may potentially help to somewhat"), and cowardly passives
("mistakes were made"). An actor-irrelevant passive is fine: "spilled at 3 AM" needs no
subject.

Naming a genuine limit is not hedging, it is the honest version of stance, and this repo
does it constantly. "Not measured above sf10" is a stronger sentence than any confident
generality.

### Tier 3: diction and mechanics

Filler, cliché metaphor, puffery, significance inflation ("paves the way"), over-signposting
("Furthermore" as glue), redundancy ("end result"), terminology drift, doubled words,
mixed quote style. The substitution table in `docs-grammar-style` is the reference. Fix
these last, because they are the cheapest to fix and the least of what a reader notices.

Terminology drift deserves one line of its own: one concept, one name, held across the
whole page. That is the repetition you *keep*.

### The costume is not the fix

Forced lowercase, sprinkled "honestly?", staccato fragments, conspicuous dash-avoidance,
every sentence a punch. That is a fresh uniform signature wearing an anti-AI costume, and
a careful reader catches it just as fast. Deleting a tell is not the same as having a
voice. The measurable tell for the mechanical version: an unusual semicolon rate with not
one dash anywhere means a substitution pass ran instead of an edit.

## Get the material before you subtract

A rewrite that can only subtract cannot produce a voice. Strip every tell from an empty
page and you get clean, generic, unowned prose that still reads as generated, because
nothing in it could only have come from this project.

What makes prose read as authored is material: a number, a date, a thing that went wrong, a
preference held without a full justification, an admitted gap. This repo is unusually rich
in it, and none of it has to be invented:

- **Already in the draft.** "Performance improved" sitting next to a table with a p99
  column means the number is right there.
- **In the repo.** `git log --oneline -- <path>` for what changed and why.
  `benchmarks/BENCHMARK_RESULTS.md` and `benchmarks/results/` for measured numbers with
  hardware attached. The test that covers the behavior, for the edge cases it enumerates.
  `docs/architecture/internals/competitive_architecture.md` for what is currently true
  against DuckDB, Spark, Polars. `MAP.md` for where a thing actually lives.
- **From the user, in one batch.** Three or four questions, each answerable in a sentence.
  What happened here that a stranger wouldn't guess? What number belongs in this paragraph?
  What did you try that didn't work? What do you believe about this that you can't fully
  defend? The last one is the highest-yield question in the set.
- **Nowhere, so mark it.** `[SOURCE NEEDED]`, listed in the report. Never filled silently.

Do this first, because knowing what you can add changes what you delete.

## The pass

Pick the depth by what it costs if a reader decides a machine wrote this. A commit message
or a code comment gets the quick version: strip the shape, cut the vacuity, fix diction on
the way past, no report. An internal note or a rules-file edit gets the standard version:
intake from what is already in front of you, the full rewrite, one self-critique pass, a
three-line report. Anything published under `docs/` gets all of it.

Depth changes how much you run, never how strictly. The fabrication rule and the invariant
list apply at every level.

1. **Read the whole thing** and list the invariants: numbers, code, links, API names,
   claims. You diff against this list at the end.
2. **Measure a baseline.**
   ```bash
   python3 .claude/skills/write-in-a-human-voice/measure_prose.py docs/user-guide/analyze/sql.md
   python3 .claude/skills/write-in-a-human-voice/measure_prose.py --summary docs/user-guide
   ```
   It counts what a regex can see and nothing else. Treat it as a floor: a page can come
   back clean on every line and still be vacuous, unsourced, and uncommitted. It is
   deliberately not a `just` recipe and not part of the gate matrix, because the thing it
   measures is a judgment call and a ratchet on it would be gamed within a week. It also
   counts a tell you are *quoting*, so a page about filler reads as full of filler.
3. **Strip the assistant shape**, before touching a single word. Cut headings dividing a
   continuous argument. Turn bulleted answers back into paragraphs where the items are not
   genuinely parallel. Delete decorative bold. Delete the recap section and end on the last
   real point. This move changes more than every diction fix combined.
4. **Cut the vacuity.** Whole sentences and paragraphs that carry no information. Expect to
   lose 15% to 25% of the words.
5. **Fix the sentence-length distribution**, not just its variance. Aim at the tails: at
   least 12% of sentences at eight words or fewer, under 72% inside the mid band. Drop a
   four-word sentence against a forty-word one.
6. **Dismantle the templates**, then take a second pass for the syntactic signature:
   unstage the clefts, cut or promote the participial tails, give the copula sentences real
   verbs, break the welded clauses apart.
7. **Cut the stance tells** and then sharpen the stance. Commit to the recommendation, give
   the mechanism, name the real limit.
8. **Unify.** One term per concept, one heading convention, one tense for findings, one
   voice end to end.
9. **Fix diction and mechanics last**, including the em-dashes, with the replacement varied.
10. **Calibrate to the surface** from the table above, and hold it to the end of the page.

## Critique it before you hand it back

Read the rewrite as four different readers, because one hostile reviewer misses whole
classes of tell. The **detector reader** asks whether a person would have formatted this
document this way, then listens to the rhythm. The **engineer** asks whether anything here
is vacuous, unsupported, or wrong. The **editor** asks whether it reads like real writing
of this kind. The **colleague** asks whether anything in it could only have been written by
someone who works on this engine; if every specific could have come from a search summary,
the intake did not happen.

Score Shape, Substance, Rhythm, Stance, Consistency, Sourcing, Diction and Surface-fit from
0 to 2. Nothing below 1, and the mechanical two must not be the only things carrying it.

Then hit the numbers rather than guessing at them. Short-sentence ratio at or above 0.12.
Mid-band at or below 0.72. Sentence CoV at or above 0.5, paragraph CoV at or above 0.3.
Heading density in the range its surface warrants. Zero em-dashes in `docs/`. No cleft or
participial-tail stacking. At least one specific that came from the repo. In a rewrite,
15% to 25% fewer words.

Then run the two checks that matter more than any of them:

1. **The claim diff.** Compare the rewrite's claims against the original's and sort every
   difference into added, strengthened, weakened, or dropped. The first three are
   regressions; revert that span. A dropped caveat is silent information loss, so restore it
   unless the deletion was deliberate and you say so.
2. **The gate.** `just docs` and `just test-py` for anything under `docs/`, because the
   examples execute and the references resolve under `-W`. `just lint-docstrings` if you
   touched a public docstring.

Stop when a pass produces no net improvement, capped at three. What is left over goes in
the report rather than getting ground at, because grinding past that point is how a rewrite
starts damaging the prose it was supposed to fix.

## When a rule here fights the task

Every rule in this file has a case that beats it. When one does, relax that rule, in that
place, for a reason you can state, and keep the rest of the shape.

The structure a surface genuinely requires stays. `docs/api/reference.md` measures 28.7
headings per 1,000 words and that is correct: it is a lookup table, and a reader scanning
for a name needs every one of them. A procedure wants numbered steps. A comparison wants a
table. What Tier 1 targets is *assistant* shape, the heading that chops a continuous
argument in half and the bold on every noun. Structure that carries information stays.

Fidelity beats rhythm. If the only way to hit the short-sentence ratio is to drop a caveat
or round a number, the number wins and the ratio goes unhit. Say so.

Required hedging is not hedging. "May spill under memory pressure" is precise where "spills"
would be false. The same protection covers a quoted passage and a defined term.

And the contract outranks the skill. `CLAUDE.md`, `.claude/rules/documentation.md` and
`.claude/rules/python-quality.md` win wherever they collide with anything here.

## Report

Say what moved, what you could not source, and what a skeptical reader would still catch.
Keep it short; a page does not need a report card longer than its own diff.

```text
Surface: <user guide | internals | docstring | rules file | benchmark | commit>
Shape:   headings/1k <before> -> <after>   bullet ratio <before> -> <after>
Rhythm:  short <before> -> <after>   mid-band <before> -> <after>   CoV <before> -> <after>
Words:   <before> -> <after>  (-NN%)
Cut:     <the categories you actually touched, with counts>
Added:   <specifics, each traceable to the repo or the user; "none" is an answer>
Invariants: numbers / code / links / API names / claim strength all unchanged
  (claim diff: +0 added, 0 strengthened, 0 weakened, 0 dropped)
Gaps left for the author: <[SOURCE NEEDED] list, or "none">
Rules relaxed: <which, where, why, or "none">
Residual: <what a skeptical reader would still flag, or "none">
```

"Some tells may remain" is not a residual. It tells the reader nothing they had not already
assumed, and vagueness of exactly that kind is what this skill exists to delete.

## One last thing, about the handoff

The rewrite is the obvious artifact. The message carrying it is the one people miss. No
"Great question", no "I've now rewritten your document to remove several AI tells", no "Let
me know if you'd like me to adjust anything". Open on the finding, stop when the finding is
done. A pass that strips sycophancy out of a page and then hands it over with "Hope this
helps!" has taught the reader nothing.
