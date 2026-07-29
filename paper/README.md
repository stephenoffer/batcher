# Batcher paper — state, sources, and open items

Author: Stephen Offer. Written 2026-07-28 against the working tree at that date.
Two-column, 10pt; **12 pages including references** (~11 body, ~1 refs), down from 15.

`make` builds `main.pdf` (TeX Live is installed). `make check` runs the static
gates without needing LaTeX.

## What I did not claim, and why

You asked for "any type of data or AI application" and "wasn't possible before."
I wrote the strongest version those could honestly carry, and stopped short of
the literal claims, because `docs/internals/competitive_architecture.md` retires
them. Its own verdict is *"the best distributed Arrow engine with a learned
optimizer, and not yet a general-purpose engine that wins at every scale."*

Specifically, the paper says the AI surface is unusually **broad and in-engine**
(native mel-spectrogram and MFCC have no in-engine equivalent in any competitor
surveyed; vector search runs in SQL), and that the value is having it all in
**one optimized plan** rather than three systems joined by files. It explicitly
does not say competitors cannot do these things — Ray Data and Daft both have
strong multimodal surfaces, and the paper says so. Overrule this if you want, but
know that the numbers behind it are in the tree and an informed reader will find
them.

Also not claimed: TPC-DS. The repo records 76 of 99 queries *planning*, which the
source explicitly labels "not an execution result, not a performance result." It
is not in the paper.

## Structure (15 sections, 12 pages)

Four expired assumptions -> three bets (one algebra / an engine that learns /
one engine for the pipeline) -> batch as the bounded case of streaming ->
architecture -> comparison table -> what changed since Spark -> Kyber ->
Carbonite -> execution -> evaluation -> what is not done yet.

## The two fragmentations the paper argues against

**Scale.** Single-node engines are fast until the data outgrows the box;
distributed engines scale but tax every small query. One mergeable algebra
removes the split.

**Shape and tier.** Engines specialize by data shape (structured /
semistructured / unstructured / multimodal) and frameworks specialize by
workload tier (classical ML / deep learning / LLM). Batcher spans both: 30 file
and table formats across the four shapes, 125 public ML entry points from GLMs
and mixtures through GPU inference to LLM batch generation, all as expressions in
one optimized plan. Figure 4.

Both counts were verified against the tree before being written
(`python/batcher/io/formats/*`, `python/batcher/ml/__init__.py::__all__`). The
claim in the paper is deliberately bounded: **not** that no engine can reach any
given cell, but that no engine surveyed reaches all of them natively, in one
plan, under one optimizer. "First to do X" is not claimed anywhere, because it is
not verifiable from this repo.

## The story the paper tells

Single-node speed that does not get taxed on the way to a cluster. Every stateful
operator is one mergeable triple, so distribution is a scheduling decision per
query rather than a runtime you start — no fixed cost on the small case, and no
rewrite on the way out. The evidence at the two ends: TPC-H sf1 parity with
DuckDB and a PyArrow sweep on one node; superlinear aggregate shuffle throughput
(2.0 → 6.9 → 15.2 GB/s at 2 → 4 → 8 nodes) on a cluster.

Kyber and Carbonite are the centerpiece, because they are what no comparator has:
cost coefficients refit from runtime, physical strategy from a bandit, memory
sized from measured per-operator footprints.

## Figures

| # | What it shows |
|---|---|
| 1 | The fork: two implementations vs. one algebra, three schedulers |
| 2 | Three systems and two copies vs. one plan and zero copies |
| 3 | Coverage grid: 4 data shapes x 4 workload tiers, and how far each comparator reaches |
| 4 | The contract loop: Core measures, Kyber decides, Carbonite protects |
| 5 | Shuffle throughput vs. cluster width, against a linear reference |

Four figures were cut in the consolidation pass (the credit-window diagram, the
Kyber phase pipeline, the fan-out bar chart, and the three-clocks diagram); their
content survives as prose. Cutting them was the only lever that actually moved
the page count, since float placement was absorbing the text cuts.
| 3 | The contract loop — Core measures, Kyber decides, Carbonite protects |
| 4 | Kyber's seven phases and where measured evidence enters each |
| 5 | The three layers governing one credit window |
| 6 | Join fan-out: 13,139 → 5,179 → 728 MB peak RSS (log scale) |
| 7 | Shuffle throughput vs. cluster width, against a linear reference |

Plus Table 1 (the generational argument: 2010s assumption → what changed →
Batcher's answer), Table 2 (the engine-by-engine capability matrix), Table 3
(TPC-H, six engines, two scales), Table 4 (operator mix incl. PyArrow). All figures are
inline TikZ/pgfplots on the Okabe-Ito colorblind-safe palette in
`tikz-paper-styles.tex`, so they stay vector and restyle from one place.

## Where every number came from

| Claim | Source |
|---|---|
| TPC-H sf1/sf10, six comparators (Table 2) | `~/tpch-bench-results/SF1_TABLE.txt`, `SF10_TABLE.txt`, 2026-07-28, after the Ray Data adapter fix in `b2dab69` |
| Geomeans and win counts | Recomputed pairwise from those tables |
| `duckdb_arrow` 21/22, 1.89x at sf10 | `benchmarks/BENCHMARK_RESULTS.md`, 2026-07-27 section (**uncommitted at time of writing**) |
| Operator mix incl. PyArrow (Table 3) | `BENCHMARK_RESULTS.md`, "full competitor sweep, sf1" |
| ClickBench | `BENCHMARK_RESULTS.md`, 2026-07-25 section |
| Shuffle 2.0/6.9/15.2 GB/s; shared memory 23x | `docs/internals/carbonite.md` |
| Partition-pair cliff (16 → 128 partitions) | `BENCHMARK_RESULTS.md`, 2026-07-18 section |
| Distributed scan ~90 MB/s, flat sf10→sf100 | `BENCHMARK_RESULTS.md`, "Distributed scale-out (sf10/sf100)" |
| Streaming fan-out 18x / 3.2x; short-circuit filter; dict compare; range join | `docs/internals/competitive_architecture.md`, ceilings 1, 2, 7 |
| Runtime filter q5; dense map fill; ndv seeding; adaptive 20% | `BENCHMARK_RESULTS.md`, 2026-07-27 section |
| Plan cache q8 340→165 ms; staged vs one-shot | `BENCHMARK_RESULTS.md`, 2026-07-26 section |
| Kyber internals | `docs/internals/kyber.md`; `python/batcher/kyber/rules/`, `learned_tuning/{bandit,crossover,priors}.py` |
| Carbonite internals; P80 per-row ratio | `docs/internals/carbonite.md`; `python/batcher/carbonite/memory/learned.py` |
| GPU workload ratios | `BENCHMARK_RESULTS.md`, "Final coverage — 10 GPU workload families" |
| Rule count (375), LOC | Counted from the tree, 2026-07-28 |

Nothing in the paper is a number this repository does not produce. Claims that
`docs/internals/competitive_architecture.md` explicitly retires are **not** made:
no "beats DuckDB on every TPC-H query", no adaptivity finer-grained than Spark
AQE, no 57x filter figure.

## The unstructured-data citation — read this before quoting it

You asked for sources on unstructured-data growth. I found something worth
knowing: **the report everyone cites for "80-90% of enterprise data is
unstructured" does not contain the word "unstructured" at all.**

I downloaded IDC's *The Digitization of the World: From Edge to Core* (Reinsel,
Gantz, Rydning, IDC #US44413318, Nov 2018 — the standard citation), extracted the
text, and grepped it. Zero hits. What it *does* support, and what the paper now
cites it for, is the datasphere growth figure: **33 ZB in 2018 to a forecast
175 ZB by 2025**, which appears on pages 3 and 6.

For the structured/unstructured split the paper cites the IDC report series that
actually forecasts it (*Worldwide Global DataSphere Structured and Unstructured
Data Forecast, 2025–2029*, IDC #US52800025). That one is paywalled, so we cite it
by title and document number and **do not quote a figure from it**, because we
have not read it. The 80–90% range appears in the paper as "analyst estimates
commonly place" with an explicit note that it is a projection and not a census.

I did not write "10x more than structured." An 80–90% share implies roughly
4–9x depending on how you count, and no source I could reach states 10x.

Both bib entries carry comments recording exactly this. If you have institutional
IDC access, read US52800025 and we can replace the estimate with a quoted figure.

## Two disclosures that are load-bearing

**The scaling claim is about the shuffle, not the read path.** The repo contains
two conflicting sf100 measurements; the newer one reports distributed
object-store scan throughput at ~90 MB/s aggregate, roughly *constant* from sf10
to sf100, with a competitor several times faster on the same corpus. Section 8.4
says this in the paper rather than quoting the shuffle number as though it
characterized an end-to-end petabyte run. Do not let a later edit blur that.

**Spark reads Parquet inside the timed region** where Batcher, DuckDB, Polars,
and PyArrow receive an in-memory Arrow handle. Its ratio is therefore overstated
by an unisolated scan cost, flagged on the table row and in the setup paragraph.
The proper fix is timing all of them from the same on-disk Parquet — an
experiment not yet run.

## Positioning: who the competitors are

The competitive set is **DuckDB, Daft, and Spark**. Polars is a reference point.
**Ray Data is a yardstick, not a rival**, and the paper is deliberately built
that way:

- It is **out of the headline TPC-H table entirely.** A SQL suite is the wrong
  instrument for a runtime with no relational optimizer, and the 0.0015x figure
  it produced was the most inflammatory and least meaningful number in the draft.
  The table footnote says where Ray Data is evaluated instead.
- In the capability matrix its relational rows read **n/a**, not amber "gap" —
  a category difference, not a deficiency.
- The background section says plainly that Ray Data sits in a different
  category, that we benchmark against it because a data engine claiming to serve
  AI workloads should be held to the standard of the runtime those workloads use
  today, and that it is a yardstick and not a rival.
- The GPU/ML section frames the question as *whether an engine that treats decode
  and inference as relational operators can hold its own against a
  purpose-built runtime*, answers yes, and says the two mechanisms behind the
  large ratios (session-warm pools, native preprocessing) are scheduling choices
  any runtime could adopt and that we expect the gap to close.
- The harness note about the single-block adapter defect stays, framed as our
  defect, because the uncorrected numbers were circulating internally.

If anyone quotes a big Ray Data multiplier out of this paper, they had to work
against the text to do it.

## Open items

1. **Repetitions and confidence intervals.** Table 2 is a single run per scale on
   a shared machine (±25% swing at load average 16–41). Run ≥5 repetitions per
   engine per scale on an idle box and report mean with a bootstrap CI.
2. **Cumulative ablation.** Nothing attributes the end-to-end TPC-H total to the
   individual mechanisms of Section 7. Disable, in order: short-circuit
   filtering, dictionary comparison, the runtime filter, the parallel dense fill.
3. **Cluster shape.** The repo records the cluster as both "8 nodes × 16 CPU" and
   "16 nodes × 8 CPU". Both are 128 vCPU, but the node count changes what the
   2.0/6.9/15.2 GB/s series means. Confirm before quoting Figure 6 elsewhere.
4. **Pin the evaluated commit hash** in the reproducibility paragraph.
5. **`mosaicstreaming` citation** points at a project repository; confirm whether
   a technical report is the citable form.

## Weaknesses a reader will find

- **Mechanism novelty.** Short-circuit conjunction, dictionary-native comparison,
  runtime filters, and IEJoin are all known. The defense is the soundness
  constructions — the infallibility guard, the cmp/gather commutation, the
  inner-join sinking argument — and honest measurement. Keep those prominent; do
  not let a length edit reduce Section 7 to a table of speedups.
- **The central claim is evaluated by testing, not measurement.** Section 8.2
  says so and admits the one place the fork reappeared: a distributability gate
  that never checked the watermark.
- **The adaptive loop is a net loss on the headline benchmark** (20% at sf10).
  Reported in Section 9. If the gate fix lands, re-measure and lead with it.
- **DuckDB wins at sf10.** Reported, and split into storage (q1/q6 at ~1.5x) vs.
  engine (q5/q7/q9/q21 at ~3-4x), which is what keeps it from being fatal.
