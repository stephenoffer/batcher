# Perf campaign ledger: resume here

Working record of the 2026-10-02 performance and memory campaign on branch
`perf/beat-competitors-2` (worktree `batcher-wt98`). Nothing has been pushed. The detailed
chronological log follows the "Resume here" section. A mirror lives at
`$ANYSCALE_ARTIFACT_STORAGE/bench_results/claude98/LEDGER.md`, and the job scripts at
`/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/claude98-persist/jobkit/`.

## Resume here (state at 2026-10-02 ~22:30 PDT)

**Update 2026-10-02 ~23:15 PDT**

- `39127198` fix(dist): `dist/executors/aligned/memory_fit.py` sizes aligned unit tasks by
  node memory as well as cores (task need = 4.5x the largest unit; per-task share passed as
  the engine `memory_bytes`, so an overrun re-runs on the spilling materializing executor),
  and caps a warm actor's calls in flight. Unit-tested. Why: the q9 diagnostic showed units
  are whole files (39 per cut regardless of `BATCHER_ALIGNED_UNIT_BYTES`; 1 GiB only split
  the other cut to 147), broadcasts are 82 MiB, and two tasks a node at 28 GB each OOM.
  With 1 GiB units the query instead failed later: `ResourceError: no surviving worker to
  recover the join shuffle on` (a shuffle-join stage after node losses).
- Validation jobs at 39127198: c98-dist4-4w prodjob_fgqe7xdlqcw9cvurilaqf9qycf (distributed
  sf1000 full suite, live timestamped log), c98-disttest-fit prodjob_mr8iuf4j299dkuigv5a7sxd9ka
  (aligned integration tests on a 2-worker cluster; local Ray has no CPUs here),
  c98-sf1000-b prodjob_1vbuhtfqf4x4ne3a83qt4p4vds (single-node q12,q16,q10,q9 incl. the
  78640bfe join fix), c98-route-q9x3 prodjob_h83gkhqa8j1ui346zuljdt5jwl (q9 x3 in one process).
- sf1000 profiles read: **q13** is 70% `fold_partial`, half of it `combine_sized`
  re-hashing ~150M `c_custkey` groups (fold already doubles, O(log n) rehash); the lever is
  eager aggregation of `orders` by `o_custkey` below the LEFT join (a Kyber rewrite).
  **q17** is 55% Parquet zstd decode and 40% join probe; lineitem is scanned twice.

**Latest commits on the branch, newest first:**

- `78640bfe` perf(interp): a join's build-side selection sparser than 1 in 32 of its source
  is gathered instead of masked. The sf1000 q12 profile put the whole-source mask (memset,
  `true_count`, page faults) at about a third of the query. **Not yet validated on the
  cluster.** First thing to do: run sf1000 q12 (and the full sf1000 board) at this commit.
- `78526d79` fix(api): a streamed plan whose join builds exceed memory runs in key-partitioned
  passes (`api/orchestration/chunked_sideways.run_partitioned_build`). Validated: sf100 on a
  32 GiB node, q9 370 s -> 19.2 s (DuckDB 25.2), so that board is now 21/22 wins (only q12
  loses, 6.85 vs 6.24 s).

**Open problems, in priority order:**

0. **sf1000 single-node q9 still does not finish in the benchmark harness.** c98-sf1000-part
   at `78526d79`: q9 KILLED at the 2400 s per-case timeout (q18 40.1 s, q21 73.4 s, q12 56.8 s
   all OK). A standalone q9 at sf1000 on older code (route trace job) took the chunked route
   and finished in **92 s**, so the harness's repeated runs (first + best-of) or state carried
   between runs is the difference. Reproduce with `routespy.py`/`routeq.py` running q9 several
   times in one process and watch which route each run takes (the partitioned-build fallback
   may not engage at that scale, or a later run may fall to the spill route).

1. **Distributed sf1000 OOM, root cause confirmed.** On 4x m6id.4xlarge (64 GB), distributed
   q9 kills workers: Ray's OOM report shows two `ray::_aligned_units_task` per node at **28 GB
   each**. The aligned executor (`dist/executors/aligned/run.py`) gives unit tasks no memory
   budget (`engine_config_json(num_cpus=unit_cpus)`, no `memory_bytes`), units are up to
   `_UNIT_BYTES` = 6 GiB decoded, a second unit is prefetched, and held broadcasts are
   unpacked per task. Planned fix: size `unit_bytes` from per-slot node memory and pass a
   per-slot `memory_bytes` so the unit engine uses its budget-aware grace join. A diagnostic
   job (c98-diag-q9, prodjob_yph8ug8wz78y21kw7axpksvdc7) logs unit counts and held-broadcast
   sizes and reruns q9 at `BATCHER_ALIGNED_UNIT_BYTES=1073741824`; read
   `bench_results/claude98/c98-diag-q9-78526d79/diag.log` and `nodemem.log` before choosing.
   q1-q8 ran without OOM at sf1000 distributed.
   **Diagnostic result (default config):** q9 makes two aligned cuts of **39 units** each
   (`unit_bytes` 6 GiB, `min_units` 48, `streams` 12); held broadcasts are tiny (one table,
   10.9M rows, **82 MiB**). So the ~28 GB per unit task is unit input (current + prefetched)
   plus engine join/aggregate state, not broadcasts. The default run timed out at 1800 s with
   repeated `_aligned_units_task` OOM kills (nodes at 55-58 of 60 GB). The 1 GiB-unit rerun was
   still running at the stop; its outcome is in `c98-diag-q9-78526d79/diag.log` (`=== 1073741824
   q9` onward) and `nodemem.log`. If 1 GiB units stay under memory, the fix is to derive
   `unit_bytes` from per-slot node memory (and pass `memory_bytes` to the unit engine config).
2. **sf1000 single node (m6id.16xlarge) loses most queries to DuckDB** (old code f5765cbb):
   Batcher wins only q1, q6, q11, q19, q22. Worst: q12 6.5x, q16 2.6x, q10 1.85x, q5 1.8x,
   q8 1.7x, q3/q4/q20/q21 ~1.5-1.7x. Profiles (frame pointers) for q12, q13, q17 are in
   `bench_results/claude98/c98-prof1000-78526d79/` (`report_*`, `children_*`, `graph_*`); q13
   and q17 have not been read yet. All three take the chunked route.
3. q12 at sf10/sf100: the remaining gap is string decode of `o_orderpriority` for every order
   (62 vs 30 ms at sf10) and a slower filtered `lineitem` scan (165 vs 143 ms).
4. Operator board: right/full outer join 2-3x DuckDB (in-memory 500K x 3M), memmove/memset
   dominated; range join 1.4x; window top-k 1.3x; as-of vs Polars.

**Jobs that may still be running** (`anyscale job status --id <id>`):
c98-sf1000-part prodjob_kxcklbz2dyawtn34y47c9w6r96 (q9,q18,q21,q12 sf1000 at 78526d79),
c98-dist3-4w prodjob_mab72ge9cpgscn993t1jykeks8 (distributed sf1000, OOMing in q9; consider
terminating), c98-diag-q9 prodjob_yph8ug8wz78y21kw7axpksvdc7, c98-dist2-4w
prodjob_qpc2euiirsw5ndv4wwd1tnbh86 (older distributed run, OOMs; consider terminating).

**Things to report to the user when finishing:**

- An Anthropic API key was seen in plaintext in the atlas project's nightly job env vars; it
  should be rotated. It was not used.
- `perf_event_paranoid` was set to 1 on the workspace box for profiling.
- `050c91a3` makes Float64 `IN` lists treat -0.0 == 0.0 and NaN == NaN, which differs from
  DuckDB on `-0.0 IN (0)`; surface as a decision.
- Full gate at `2e6ff929` had only environmental failures (see log below); re-run the gate at
  the end.

---

work is local commits.

## The user's goal (verbatim intent)

1. Run, and fix where failing, all tests, examples and benchmarks for Batcher and the
   competitor engines.
2. Improve Batcher until it beats every competitor on every benchmark, staying within the
   architecture (CLAUDE.md), across all scopes: streaming, batch, single-node, distributed.
3. Test sf100, sf1000 and the AI/GPU suites, using dedicated Anyscale jobs for benchmarking.
4. Later additions:
   - Continue until Batcher wins every benchmark.
   - sf1000 must run with **no OOMs**.
   - Run many **large distributed larger-than-memory** workloads.

## Where things are

- **Worktree:** `/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/batcher-wt98`.
  - Branch `perf/beat-competitors-2`, based on `d5ebcc8b`.
  - Don't `just build` in the shared tree. Build with
    `cargo build --release -p bc-py --features pyo3/extension-module`, then
    `cp target/release/lib_native.so python/batcher/_native.abi3.so`.
- **Shared tree:** `.../batcher`. A peer session edits `docs/` and `README.md` there. Leave it
  alone.
- **Job kit (persistent copy):** `claude98-persist/jobkit/`.
  - `launch.sh` (single node) and `launch_dist.sh` (multi-node).
  - Entry scripts: `suite.sh`, `ab.sh`, `test.sh`, `hang.sh`, `dist.sh`, `planab.sh`,
    `planonly.sh`, `disttest.sh`, `gpu.sh`.
  - Drivers: `nodemem.py`, `explainq.py`.
  - `prebuilt/head2.so` is the engine at HEAD's Rust; `leaf.so` is the morning engine. Both
    are shipped as `.engine` files.
  - The originals live in `/tmp/claude-1000/.../scratchpad/jobkit`, which won't survive a
    reboot.
- **Results:** `$ANYSCALE_ARTIFACT_STORAGE/bench_results/claude98/<RUN>/`.
- **Write-up:** the top two sections of `benchmarks/BENCHMARK_RESULTS.md` in the worktree.
- **Local probe data:** `/home/ray/default_cld_g54aiirwj1s8t9ktgzikqur41k/scratch-li/`
  (sf10 subset; `lineitem` is 3 files).

## Commits on the branch (oldest first), with measured effect

| commit | change | measured |
|---|---|---|
| 1625c83c | transport CAS loop (clippy on Rust 1.99) | gate |
| 4ff4acc8 | ragged-CSV diagnosis | test |
| f507956c | codemod enum tables (py3.11) | test |
| a2d7fda5 | outer → inner under an inner join | TPC-DS q93 118→66 ms |
| ceaf858e | runtime join filter tested during the Parquet decode | TPC-H sf100 q19 1461→1220 |
| 612c9aa0 | shard the spine above a materialized breaker | TPC-DS sf10 sum 0.941x |
| 926ba7b1 | in-memory scan signature keyed by column names | q72 contamination fixed |
| 6ef7a691 | results write-up (morning) | |
| 450f78f1 | integer as-of join fast path | 1M rows 1054→53 ms (Polars 35) |
| ac128e34 | as-of / explode / unpivot competitor cases | new benchmark cases |
| da188d39 | explode by slice; faster unpivot | explode 24→4.5 ms (DuckDB 16.5); unpivot 163→130 (DuckDB 213) |
| e5ec9dea | explode/unpivot-created columns have no base column | first run 131→104 ms |
| 1d5f7001 | late filter explores only its samples | q6 (3 sf10 files) ~225→~184 ms |
| 4149251b | spilling join partitions in chunks, not 8 MiB morsels | sf1000 q9 spill writes 27→111 MB/s |
| 40b9d8cc | semijoin placement: a rounding tie is a tie | TPC-H sf100 0.970x; q18 3838→2742 ms |
| 40d101bd | identity gather shares columns | as-of 1M 60 ms (Polars 55) |
| 050c91a3 | **wrong answer fixed**: float IN compares by float identity | seq == JIT; now differs from DuckDB on −0.0 |
| 522ece87 | arith folds fused into one traversal | re-plan −5..13% |
| 6be6ae9e | temporal folds fused into one traversal | re-plan −2..5% |
| 2e6ff929 | results write-up (afternoon) | |
| f5765cbb | harness `BENCH_CASE_TIMEOUT_S` | |
| 5592374a | DuckDB reader uses the S3 credential chain (private bucket) | unblocks distributed runs |
| 47f8c9ef | stage the aggregate over a held oversized input (q18 sf1000 OOM) | q18 sf1000 SIGKILL → 38.4 s |
| a153a57a | implied transitive join edges in join-order regions | q9 local 1164→856 ms; sf1000 plan unchanged |
| a117b97c | **revert** of a153a57a | TPC-DS q25/q17/q29/q31 +10–19% every round; total 1.004x |

**A/Bs of the day,** morning tree against `4149251b`:

- TPC-H: sf10 0.978x, sf100 0.985x.
- TPC-DS: sf10 1.002x.

## Results that matter

- **Single-node sf1000 at d5ebcc8b and 6ef7a691:**
  - q9 doesn't finish in 1–4 h. It goes down the out-of-core route. The partition phase is
    fixed (4x faster); the bucket-join phase still runs at a load of about 2 on 64 cores.
  - q18 is SIGKILLed within about a minute: the non-streamed `lineitem` read climbs past
    123 GB in 20 s. `47f8c9ef` targets this.
  - DuckDB: q9 50 s, q13 27 s, q17 15 s, q18 30 s.
- **GPU vs Ray Data (4×A10G):**
  - Parquet inference: 10M rows 3.64x, 40M rows 1.71x.
  - ResNet: 2.42x.
  - Ray workers need torch installed (`prep_nodes.py`).
- **Boards vs best rival (morning `after2` board):**
  - TPC-H sf1 0.715.
  - TPC-H sf10 1.134 (11 losses). q6 sf10: 94 ms after the fix vs DuckDB 56.
  - TPC-H sf100 1.130. Worst: q12 1.92x, q16 1.71x, q18 1.68x (now fixed?), q10 1.48x,
    q5 1.40x.
  - TPC-DS sf10 vs DuckDB: 1.19x sum, 40 losses, mostly convergence and planning cost. Worst:
    q70, q39, q64, q5, q82, q37, q21, q95, q78.
  - JOB: 0.50x sum (Batcher faster).
  - H2O groupby: losses only at 64 cores.
  - Operator mix: full-outer join 1.5x, range join 1.4x, window-topk 1.3x.
- **Full gate at `2e6ff929`:**
  - main suite: 11 environmental failures (agentic_runner without git, dist_runtime_env,
    broker timing);
  - integration: 4 failures (2 device-request environmental, 2 contention flakes that pass
    serially, 22/22);
  - examples: pass.

## Jobs in flight at the time of writing (check with `anyscale job status --id ...`)

| job id | name | what |
|---|---|---|
| prodjob_82ssry94wafh9qci3gzplvrmxt | c98-sf1000-all | all 22 at sf1000, Batcher then DuckDB, code f5765cbb (before the held fix), 1800 s/case |
| prodjob_qpc2euiirsw5ndv4wwd1tnbh86 | c98-dist2-4w | distributed TPC-H sf100 + sf1000 on 4× m6id.4xlarge (288 GiB total), code 5592374a; `nodemem.log` has per-node memory and OOM counts |
| prodjob_revfuxl3prpnid64zvjpjnslee | c98-held2-sf1000 | q18, q17, q21, q20, q2, q15 at sf1000 with the held fix (47f8c9ef) |
| prodjob_b86jclevq5j88i1pdfvwkb5lgf | c98-sf100-small2 | all 22 at sf100 on m6id.2xlarge (32 GiB, larger than memory), Batcher and DuckDB, 47f8c9ef |
| (terminated) | c98-ab-impl-h | not needed after the revert |
| prodjob_nnq4sqktpp9hy1qdal3khdfps9 | c98-ab-impl-ds | done: 1.004x, four star-join regressions, so reverted |

Summarize an A/B by fetching `ab.log` from the run directory and summing the min ms per query
for arm a and arm b (see `abreport.py`).

## Results 2026-10-02 ~20:30

- **Distributed sf100** (c98-dist2-4w, 4× m6id.4xlarge with 64 GiB each, code 5592374a):
  - All 22 correct, none killed.
  - Best times 0.6–6 s (q1 2.3 s, q9 5.9 s, q8 6.0 s). First runs take 10–31 s each.
- **Distributed sf1000** (same cluster, larger than memory): **node loss**.
  - At 20:20:41 one worker reached 48 GB used and 14 GB free.
  - Probes then timed out, the GCS marked two worker nodes dead ("raylet overloaded"), and
    the kernel OOM-killed processes on two nodes.
  - Before that, workers held 35–42 GB used and released it between queries.
  - The failing query is named in `tpch_dist_sf1000.log` once the suite finishes.
  - **Next target:** worker-side memory bounding in the distributed path.
- **Held-input fix at sf1000** (c98-held2-sf1000, 47f8c9ef, one m6id.16xlarge): all correct.
  - **q18 38.4 s** (was SIGKILLed; DuckDB 29.8).
  - q17 23.8 s (DuckDB 15.3).
  - q21 70.0 s, q20 19.3 s, q2 3.2 s, q22 4.2 s.
  - q15 11.4 s, and its decline guard held.
- **sf100 on one 32 GiB m6id.2xlarge** (larger than memory, code 47f8c9ef):
  - All 22 correct, **no OOM**.
  - q9 took **370 s** (spill route); the rest 0.6–15 s.
  - DuckDB's numbers on the same node are pending.

- **Why q9 goes out of core (found 20:45, local sf10 subset under a 0.2 GB cap):**
  - `core.execute_local_parquet` raises `MemoryBudgetExceeded`: "the streaming executor's
    join build sides do not spill" (`bc_interp::stream::builds::check_total`).
  - `run_chunked` then declines and the conductor takes Python `spill_to_disk`.
  - That route spills every join, nested, and materializes each join's output in memory
    (`dist/spill/staging.stage_breaker_inputs` returns a `pa.Table`).
  - The bucket pairs are joined serially. Locally 3.9 s vs 0.9 s; sf100 on 32 GiB 370 s.
  - **Planned fix:** a spillable (hybrid/grace) build in the streaming executor, so the
    streamed spine survives an oversized build.
  - The q9 sf1000 route trace (c98-route-q9, prodjob_qmxkx74bhvfkr6b1bjw1d99lf6) confirms
    whether sf1000 exits the same way.

- **sf100 on one 32 GiB node, Batcher vs DuckDB** (c98-sf100-small2, 47f8c9ef): all correct,
  no OOM on either engine.
  - **Batcher wins 18 of 22.** Losses: q10 1.10x, q12 1.11x, q13 1.01x, and **q9 14.66x**
    (370 s vs 25 s).
  - Total 2.30x because of q9; excluding q9, ~0.78x.
  - **q9's out-of-core route is the top target.**
  - An unmeasured patch for concurrent bucket pairs is saved as
    `claude98-persist/pair-concurrency.patch` (not committed).

## Found and not fixed (next targets)

1. **sf1000 q9 OOM / out of core.** The plan builds 800M `partsupp` and 1.5B `orders` rows. The
   cost model over-estimates the composite join (800M estimated vs ~326M actual). The spill
   route's bucket join is serial. Ideas:
   - a memory-aware build-side and route choice;
   - parallelize the bucket pairs under Carbonite reservations;
   - a native grace hash join.
2. **Only one Parquet scan streams;** the others are read whole. `47f8c9ef` covers aggregate
   subtrees only.
3. **Repeated-query convergence:** 2–3 re-plans at 0.25–1.2 s each. The planner profile is
   flat.
4. The spilling sort's bucketing pass writes one batch per staged batch, as the join did.
5. Scan decode costs on q6/q12 (memset, memmove, `extend_from_dictionary`).
6. TPC-DS q13 regression to explain (+7% in the tie A/B).
7. The harness reports FAILED when only Daft is wrong.

## Notes to report to the user

- An Anthropic API key was seen in plaintext in the atlas project's nightly job env vars. It
  should be rotated. Don't repeat it.
- `kernel.perf_event_paranoid` was lowered to 1 on the head node.
- Batcher now differs from DuckDB on `x IN (...)` with `-0.0` (DuckDB is inconsistent with
  itself). This was a deliberate decision.

## 2026-10-02 21:05 PDT update
- Commit `78526d79`: partitioned-build fallback (chunked_sideways.run_partitioned_build). When the
  streamed path's join builds exceed budget, restrict the largest inner/semi build to
  hash(key) mod P == p, run the top aggregate per pass, combine. Crosses an eager pre-agg only
  when the aggregate above folds each partial with its combining function (`_agg_reads`).
  Local: all 22 TPC-H match uncapped at 0.45 GB and 0.2 GB caps; q9 partitioned (5 passes, 4.3 s).
  Tests: tests/differential/test_diff_chunked_partitioned.py, tests/unit/test_partitioned_build_shape.py.
- sf100 on m6id.2xlarge (32 GiB) at 47f8c9ef: Batcher wins 18/22; losses q9 (370 s vs 25 s),
  q10 (12.5 vs 11.4), q12 (6.9 vs 6.2), q13 (14.5 vs 14.4).
- sf1000 m6id.16xlarge at f5765cbb (old code): 20/22 OK; q9 KILLED at 1800 s, q18 SIGKILL
  (q18 fixed since in 47f8c9ef). DuckDB half still running in c98-sf1000-all.
- Distributed c98-dist2-4w: sf100 done (all OK); sf1000 running, nodemem shows worker OOM kills
  (counts 2,0,3,3,1 by 20:57). Need the per-query attribution once tpch_dist_sf1000.log uploads.
- Launched: c98-sf100-part (prodjob_tzzl1q31u3sjqu2tlatrycjcnr; q9,q10,q12,q13 sf100 32 GiB),
  c98-sf1000-part (prodjob_kxcklbz2dyawtn34y47c9w6r96; q9,q18,q21,q12 sf1000).
- Local findings 21:10 (sf10 subset, 8 cores):
  - q12 (269 vs DuckDB 208 ms): staging the selective lineitem side first does NOT help
    (263 ms). Cost is the filtered lineitem scan (165 vs 143) and reading o_orderpriority for
    all 4.5M orders (62 vs 30 ms; string decode, no StringView). Needs key-filtered late
    materialization in the Parquet reader. Not started.
  - q7: chunked runs 478-580 ms run to run; a route bandit sends ~1/4 runs adaptive. Noisy.
  - Full/right outer join (prefiltered 500K x 3M, in-memory): INNER 19.7 vs DuckDB 41;
    LEFT 35 vs 32; RIGHT 85 vs 29; FULL 67 vs 33. Parallel (rayon, partitioned shuffle), but
    memmove 12% + memset 8% dominate. Parked.
- dist.sh in jobkit now timestamps every log line and uploads the live tpch_dist_sf*.log
  every 120 s, so an OOM can be attributed to a query.
- Launched c98-dist3-4w (prodjob_mab72ge9cpgscn993t1jykeks8): distributed sf1000 at 78526d79,
  4x m6id.4xlarge, attributable logs.
- Suspect (unverified) for distributed OOM: aligned unit tasks get no per-task memory grant
  (aligned/run.py `engine_config_json(num_cpus=unit_cpus)`), and a warm fleet actor runs up
  to 3 units at once.
- c98-sf100-part (78526d79, m6id.2xlarge 32 GiB): q9 19.2 s (was 370 s; DuckDB 25.2) WIN,
  q10 11.3 (DuckDB 11.4) win, q12 6.85 (6.24) loss, q13 13.5 (14.4) win. => sf100 32 GiB now 21/22.

## 2026-10-02 ~21:50 PDT
- **Distributed OOM attributed**: c98-dist3-4w (78526d79) OOM-killed during distributed **q9**
  at sf1000 (q1-q8 fine). Ray's report: two `ray::_aligned_units_task` per node at **28 GB each**
  on a 64 GB node. Root: aligned executor unit tasks (dist/executors/aligned/run.py), which get
  no per-task memory budget (`engine_config_json(num_cpus=unit_cpus)`), _UNIT_BYTES 6 GiB
  decoded + a prefetched next unit + held broadcasts + engine state.
  Diagnostic launched: c98-diag-q9 (prodjob_yph8ug8wz78y21kw7axpksvdc7) logs units/held sizes,
  runs q9 at default and at BATCHER_ALIGNED_UNIT_BYTES=1 GiB with nodemem.
- **sf1000 single node (m6id.16xlarge) is a big loss** (c98-sf1000-all, f5765cbb), Batcher vs DuckDB ms:
  q1 13900/16304 W, q2 3142/2725, q3 20591/13584, q4 15060/9970, q5 29155/16286, q6 5684/5824 W,
  q7 20222/16072, q8 32377/18768, q9 KILLED/50903, q10 32077/17352, q11 1485/2901 W,
  q12 54653/8362 (6.5x!), q13 42747/27718, q14 11133/10355, q15 11995/10133, q16 10989/4302,
  q17 23008/15297, q18 KILLED/29933, q19 14011/14383 W, q20 19473/11261, q21 70062/41306,
  q22 4168/5041 W. Profile job c98-prof1000 (prodjob_vqn71uh95rarpyuniq733na4qn) on q13,q17,q12.
