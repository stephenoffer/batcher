# Spill, OOM, and larger-than-memory improvements ledger

A running record of the work that makes Batcher survive and go fast on data that does not fit
in memory: the spill stores, the out-of-core operators, the memory envelopes that decide when
to spill, and the failure modes that turn a bounded-memory design back into an OOM.

This is a contributor's working record, not a published page. Every entry names what was
wrong, why it passed the gate anyway, and the test that now fails without the fix.

## How to read an entry

Each entry is `#N — one-line claim`, then:

- **Was:** the behavior before, stated so it can be reproduced.
- **Now:** what changed.
- **Why it matters:** the failure it removes, at the scale where it appears.
- **Proof:** the test or measurement that fails without the change.

---

## Program 1 — The spill store: descriptors, syscalls, and the scratch path

The `DiskSpillStore` is the floor every out-of-core operator stands on. Three of its
properties were sized for the aggregate's use (a fixed handful of partitions) and broke under
the external sort's use (a partition *is* a run, and there are as many runs as morsels).

### #1 — Spill writes are buffered

- **Was:** `StreamWriter<File>` wrote straight to the file. Arrow IPC issues a separate
  `write` per message *and per buffer inside it*, so a batch with `k` columns cost on the
  order of `2k` syscalls, most of them a few KB of validity/offset data.
- **Now:** a `BufWriter` sits under every spill writer, sized as a **shared 32 MiB budget
  divided among the store's partitions** (clamped to [8 KiB, 1 MiB]) rather than a flat size
  per file. The bytes on disk are identical, so nothing about the read path or the result
  changes.
- **Why it matters:** a spill of a few thousand morsels over a dozen columns spent hundreds
  of thousands of syscalls on data that coalesces into a handful of large writes. This is
  pure throughput on the path that is already the slowest thing in the query.
- **Why the budget is shared:** the partition count is neither small nor bounded by the
  caller — a skewed grace aggregate re-partitions up to 4,096 ways — so a flat 1 MiB per
  writer would have been 4 GiB of buffers held by the subsystem whose entire purpose is to
  *stop* using memory, growing with exactly the skew it exists to absorb. The 8 KiB floor is
  the `BufWriter` default, so the widest fan-out is never worse than the unbuffered path.
- **Proof:** `write_buffering_is_a_shared_budget_not_a_per_file_size` pins the floor, the
  ceiling, and the total at every fan-out; correctness is covered by the existing spill
  suites.

### #2 — Spill reads use a spill-sized buffer

- **Was:** `BufReader::new` — the 8 KiB default. An IPC `StreamReader` reads a length prefix,
  then a metadata block, then a body, so a sequential scan paid several syscalls per message.
- **Now:** 256 KiB. Deliberately smaller than the write buffer: a bounded-fan-in merge holds
  one reader open *per run* (16 by default), so this is multiplied by the fan-in.

### #3 — `SpillStore::close_partition` releases a finished partition's descriptor

- **Was:** a partition's writer stayed open from its first append until the partition was read
  back. Fine for the aggregate (a fixed handful of partitions), fatal for the external sort,
  where pass 0 writes each run once and never returns to it.
- **Now:** a trait method that finishes the writer while leaving the data readable, and
  `read` no longer depends on the writer still being live to decide there is data.
- **Why it matters:** pass 0 held **one open file per input morsel**. A sort large enough to
  spill has thousands to millions of morsels, so it hit `EMFILE` on exactly the inputs
  spilling exists to serve — with the disk nowhere near full, and nothing in the error naming
  spill as the cause.
- **Proof:** `crates/bc-runtime/tests/spill_descriptor_bounds.rs` — 512 partitions written
  and closed must not grow the process's open-descriptor count, and every partition must
  still read back.

### #4 — Pass 0 and the merge passes close each run as they finish it

- **Was:** the external sort never closed a run.
- **Now:** `spill_run` closes the run it just wrote, and each merge pass closes its output
  group. Open descriptors are O(1) in pass 0 and O(fan-in) in a merge pass, regardless of
  input size.

### #5 — A planted symlink can no longer capture spilled rows

- **Was:** the store claimed its private scratch directory with `create_dir_all`, which
  *succeeds* when the name is already a symlink. Any local user who can write the shared
  spill root (`/tmp` by default, or one directory serving several worker processes) could
  pre-create the name and have every spilled row — the query's actual data — written through
  it. The owner-only mode applied afterwards lands on a path the attacker still controls, and
  the query reports nothing, because from its side everything worked.
- **Now:** the root is still created permissively (it is shared and may legitimately exist),
  but the leaf is claimed with `create_dir`, which fails on an existing name, symlink
  included. A clash advances the process-wide counter and retries, bounded, so a stale
  directory left by a reused pid does not fail the query.
- **Proof:** `crates/bc-runtime/tests/spill_symlink_hardening.rs` — symlinks planted over the
  next names the counter will hand out must neither fail the store nor receive a single byte.

### #6 — Scratch abandoned by an OOM-killed process is reclaimed

- **Was:** a store removes its own directory on drop, which covers success, error, and
  panic. It does not cover `SIGKILL` — and the process most likely to be `SIGKILL`ed is the
  one spilling, because that is the process the kernel OOM killer picks.
- **Now:** the first time a process touches a spill root it removes `bc-spill-{pid}-{seq}`
  directories whose pid is no longer a live process. A live pid is left alone, which is what
  makes it safe for concurrently spilling siblings sharing one root; a reused pid reads as
  "alive" and the directory is kept, which is the safe way to be wrong.
- **Why it matters:** the abandoned scratch sits on the *spill filesystem*. The next query
  has less room, spills harder, and is likelier to be killed in turn — a node ratchets into
  a state where every large query fails for space while the data that filled the disk
  belongs to no process at all.
- **Proof:** `crates/bc-runtime/tests/spill_orphan_sweep.rs`, which asserts as much about
  what is *not* deleted (a live sibling's scratch, unrelated directories, names that merely
  resemble the store's own) as about what is.

### #7 — A full spill filesystem says so, and says which one

- **Was:** `RuntimeError::Io("No space left on device")`, or — when it arrived through the
  IPC writer, which is the usual path — a generic arrow error.
- **Now:** a distinct `SpillOutOfSpace` naming the directory, the volume already written,
  and the three settings that change the outcome.
- **Why it matters:** spill scratch defaults to the system temp directory, which in a
  container is routinely a small overlay or a tmpfs sized far below the query's spill
  volume, while the large volume the user believes is in use sits elsewhere. The bare errno
  gives no way to discover that.
- **Proof:** `a_full_spill_filesystem_is_reported_as_such_through_both_error_paths` — both
  the direct `io::Error` path and the arrow-wrapped one, plus `EDQUOT`, and a
  permission error that must keep its own identity.

---

## Program 2 — The external sort: fewer, larger runs

### #8 — Pass 0 builds runs to a byte target instead of one run per morsel

- **Was:** every input morsel became its own sorted run and its own file.
- **Now:** morsels are accumulated to `run_target_bytes`, concatenated, and sorted as one
  run. The input is already resident at this point, so the only new cost is the transient
  concat + sort scratch for a single run.
- **Why it matters:** the merge is multi-pass with fan-in `f`, so it rewrites the entire
  dataset `ceil(log_f(runs))` times. At a 2 MB morsel a 10 GB sort produced ~5,000
  single-morsel runs and paid **four** full passes; 64 MB runs give ~160 and pay **two**.
  Halving the pass count halves the spill I/O — the dominant cost of an out-of-core sort. It
  also stops the sort from creating one file per morsel in a single directory.
- **Proof:** `external_sort_run_coalescing_matches_inmemory` in
  `crates/bc-interp/src/ops/external_sort.rs` — both the no-merge case (one run) and the
  several-multi-morsel-runs case must equal the in-memory oracle **row for row**, on a key
  carrying NaN, `-0.0`, and ties.

### #9 — The run target is derived from the operator's memory envelope

- **Was:** n/a (runs were one morsel).
- **Now:** the sort uses a quarter of the operator's envelope, floored at 1 MiB and capped at
  the 64 MiB module default; callers with no envelope (the quantile/median spill paths) take
  the default. A quarter leaves room inside the envelope for the run's own concat and sort
  scratch.

---

## Program 3 — One skew guard, applied to every grace operator

The join fix in Program 2 is not a join fix. Every grace operator sizes its bucket count the
same way — total bytes over the memory envelope — which is an *average*-case fit, and that
sizing is the only thing between a spilling operator and an OOM. `bc-interp/src/spill_split.rs`
holds the guard once so it cannot drift between operators: measure a bucket **before** reading
it, and if it is over budget, stream it into a child store re-partitioned by the same keys
under a salt derived from the recursion depth.

### #10 — The grace join re-splits a bucket that does not fit

- **Was:** the bucket count was sized from the build side's *average* bytes per bucket, then
  **both** sides of each bucket were materialized whole before being joined. Under key skew
  one bucket holds far more than its share, so it OOMed at exactly the point spilling was
  supposed to have prevented it. This is the standard reason a skewed Spark join dies, and
  the grace *aggregate* already guarded against it; the join did not.
- **Now:** both sides are asked for their bucket's size *before* the bucket is read — the
  whole point, since the decision has to happen without first pulling in the thing that does
  not fit — and an over-large bucket is streamed into child stores re-partitioned under a
  depth-derived salt. Peak is one batch plus one sub-bucket.
- **Why the probe side counts:** the bucket count is sized from the build side alone, so a
  fact table with a hot key leaves a *probe* bucket orders of magnitude over the envelope
  even when every build bucket fits. Both sides are measured.
- **Proof:** `spilling_join_with_skewed_buckets_matches_sequential` — one hot key carrying
  the bulk of both sides plus cold keys that match partially, against the sequential oracle,
  for all six join types.

### #11 — Re-splitting needs a hash independent of the split that produced the bucket

- **Was:** n/a — nothing re-split.
- **Now:** `bucket_of_salted`. A salt of 0 is byte-for-byte the historical assignment,
  because the unsalted bucket of a key is a **cluster-wide contract** that both sides of a
  distributed join and every reducer must agree on; salting is only ever a local decision
  about how to re-split.
- **Why a different bucket *count* is not enough:** `bucket_of` reads the low bits at a
  power-of-two count and the high bits otherwise, so re-partitioning a bucket produced by a
  high-bit split with another high-bit split maps a contiguous hash range onto about one
  sub-bucket — nearly every row lands together again and the recursion makes no progress.
- **Proof:** `salt_zero_is_the_historical_assignment` (identity at five partition counts),
  `a_salted_resplit_spreads_a_bucket_that_an_unsalted_one_does_not`, and
  `equal_keys_still_co_locate_under_a_salt` — the last being the property that makes
  re-splitting both sides of a join legal at all.

### #12 — Empty shards are no longer written

- **Was:** partitioning wrote every shard, empty ones included. At a 256-way fan-out over
  thousands of morsels that is millions of IPC messages carrying no rows.
- **Now:** skipped, which means an untouched bucket has no file — so the join treats an
  empty read as an empty relation of that side's schema rather than as the "no input at all"
  error it used to be.

### #13 — The window spill re-splits an over-large bucket

- **Was:** buckets were sized from the average and then materialized. A skewed `PARTITION BY`
  put one bucket far over the envelope, and the window kernel materializes its bucket, so
  that bucket OOMed at the point spilling was meant to prevent it.
- **Now:** the same measure-then-split guard as the join.
- **Why the correctness bar is higher here:** a join bucket split across two sub-buckets is
  merely slower — every pair still meets somewhere. A *window* partition split across two
  sub-buckets is **wrong**: it produces two independent rankings. Re-splitting therefore
  re-partitions by the same `PARTITION BY` keys, so equal keys still co-locate and every
  window partition lands whole. This also means the split cannot help a single hot key,
  whose rows re-hash together under any salt; the depth bound is what makes that terminate.
- **Proof:** `window_spill_skewed_partitions_match_in_memory` asserts `row_number` against
  the in-memory kernel over a hot key plus 60 cold ones — `row_number` being exactly the
  function a split partition cannot survive.

### #14 — Every grace fan-out is capped, including the aggregate's

- **Was:** three call sites each computed `bytes / budget` with no upper bound. An input
  three orders of magnitude over the envelope asked for thousands of buckets: thousands of
  spill files, each receiving shards too small to write efficiently.
- **Now:** all of them route through `grace_bucket_count`, capped at 256 per level, with the
  buckets that still do not fit re-split. The aggregate's own bucket that is still too large
  was already handled out of core by `combine_finalize_spilling`, so capping it is strictly
  better rather than a trade.

### #15 — The re-split is shared code, not three copies

- The join, the window, and (in `bc-runtime`, over the partial-state layout only it can see)
  the aggregate all needed the same measure-then-split-with-a-salt shape. Two of the three now
  share one implementation. This is the case `.claude/rules/architecture.md` names: the
  subsystems cannot import each other, so copy-paste is the only *wrong* way to share.

### #16 — The aggregate's measure-before-read decision reaches every recursion level

- **Was:** the streaming split was added where the *top-level* merge chooses a partition. The
  recursive levels kept the older shape — read the sub-partition whole, then let
  `merge_partition` discover it was too big and split it.
- **Now:** `merge_child_partitions` makes the same decision the top level does, so a
  sub-partition that is still over budget is streamed into its child rather than materialized
  first.
- **Why it matters:** the leak was in exactly the case the guard exists for. One level of
  re-hashing does not separate a key from itself, so severe skew is *precisely* what leaves a
  sub-partition over budget at depth 1 — the bound held for the first level and leaked at
  every level below it.
- **Proof:** `every_recursion_level_splits_without_materializing`. The discriminator is where
  the batches come from: a streamed partition reports a `drain`, a materialized one a whole
  `read`. Before the fix the entire run produced exactly **one** drain however deep the
  recursion went. The earlier test could not see this because its counting store handed back
  a plain store from `child`, so everything below depth 0 was invisible to it.

### #17 — The aggregate's per-level split is capped at 256 ways, not 4,096

- **Was:** `sub_partition_count` clamped at 4,096 per level. Recursively, a 4,096-way split of
  a partition that was itself the product of a 4,096-way split creates millions of
  sub-partitions holding a handful of rows each — and on a disk store, a file for every one.
- **Now:** 256 per level. This is not a lower ceiling: four levels at 256 ways is four billion
  sub-partitions, far past any real ratio of state to envelope. It bounds the cost of a single
  level, and the recursion reaches the same per-partition size through more, cheaper levels.

---

## Program 4 — The handoff to the spilling executor must precede the allocation it avoids

The streaming executor's breakers do not spill. When one finds its input over the envelope it
returns `MemoryBudgetExceeded`, and `bc_py::execute_plan` re-runs the query on the
materializing executor, which does. That handoff is the mechanism that turns a would-be OOM
into a spill.

### #18 — Breakers give way *while* draining, not after

- **Was:** `let batches = drain(...)?;` then `check_budget(batch_bytes(&batches))`. The check
  ran after the allocation it exists to prevent. An input ten times the envelope is ten
  envelopes of resident memory before a single byte of the check runs — so the process dies
  at the drain, and the executor that could have spilled is never reached. **In exactly the
  case the guard exists for, the guard was unreachable.**
- **Now:** `drain_within_budget` accumulates and checks per morsel, so the held bytes are
  bounded at roughly the envelope plus one morsel. Applied to `Sort`, `Distinct`,
  `UNION DISTINCT`, and the deferred-breaker path (`Window`, `Sample`, `AsofJoin`), the last
  two budgeting each branch or child against what the earlier ones already hold, since the
  envelope covers all of them at once.
- **Cost:** the reported `needed` is where the accumulation crossed the line rather than the
  input's true size. That is deliberate — knowing the exact total means holding the whole
  thing, which is what is being prevented — and the decision it feeds ("this does not fit,
  use the executor that spills") is the same either way.
- **Proof:** `an_over_budget_breaker_gives_way_before_materializing_its_input` in
  `crates/bc-interp/tests/stream_memory.rs`, measured on live heap by that binary's counting
  allocator. Refusing a 64 MB input under an 8 MB envelope: **32 MB peak before, under 16 MB
  after**. The sort's input is a *projection*, not a scan — a scan over already-materialized
  sources yields zero-copy slices, so collecting them allocates almost nothing and would have
  hidden the very thing being measured.

---

## Program 5 — The out-of-core bucket pipeline (control plane)

`iter_batches` routes a top-level sort, join, or window over bounded sources to
`dist/spill_breakers`: the input is consumed to disk, then the result is yielded one bounded
bucket at a time. This is the path that makes a larger-than-memory sort possible at all, so
the properties of *its* buckets decide whether "bounded" is true.

### #19 — The out-of-core sort sizes its buckets from the input, not from a constant

- **Was:** `n_buckets = _fd_safe(num_partitions)` — a constant 16 by default. Each bucket is
  read back **whole** to be sorted, so peak memory was `input / 16`: it grew linearly with
  the input.
- **Now:** staging already measures the mapped input, so the count is derived from it and
  `memory.spill_bucket_max_bytes` — the same envelope the aggregate reduce already used. The
  caller's count becomes a floor (small sorts keep their parallelism) and `_fd_safe` the
  ceiling (the partition phase holds one writer per non-empty bucket).
- **Why it matters:** this is the *one* number in an out-of-core sort that must not be a
  constant. A sort large enough to need this path was exactly the sort that then OOMed on its
  first bucket — and it looked like an ordinary OOM, not a misconfigured spill.
- **Still inherent:** a range partition cannot split a single key value across buckets, since
  equal keys must share one or the concatenation is not sorted. A sort whose key is one hot
  value is still bounded by that value's rows. That is a property of ordering, not of the
  sizing.
- **Proof:** `test_spill_sort_bucket_count_tracks_the_input_not_a_constant` — shrinking the
  envelope must produce strictly more buckets for the same data, with an identical result.
  The streaming global window shares `stage_and_partition`, so it is fixed by the same change.

### #20 — `iter_batches` documented a limitation it does not have

- **Was:** the docstring said "Other plans (sort / join / window / multi-source) materialize
  first". The router has streamed those from the out-of-core bucket pipeline for some time.
- **Why it matters:** this is the API doc a user reads when deciding whether Batcher can sort
  something bigger than memory. It told them no, and the answer is yes.

### #21 — The aggregate's grace recursion re-partitioned nothing

- **Was:** when a spilled aggregate bucket exceeded `spill_bucket_max_bytes`, the reduce
  re-partitioned it into `_SUB_BUCKETS = 8` sub-buckets and reduced them one at a time. The
  parent bucket count is `_fd_safe(num_partitions)` — 16 by default. **Both are powers of
  two, and bucket assignment reads the low hash bits at a power-of-two count**, so every row
  in parent bucket `b` re-partitioned to `b & 7`. One sub-bucket, always, at every level.
  The reduce wrote and re-read the whole over-large bucket three times, changed nothing, and
  then combined it anyway.
- **Now:** the re-partition is salted by recursion depth (`partition_batches_salted`, new
  through the FFI). A salt of 0 is byte-identical to the historical assignment, which is the
  cluster-wide contract every reducer and both sides of a distributed join agree on; salting
  is only ever local.
- **Why nothing caught it:** the result was always correct — a re-partition that moves no
  rows is still a valid partition — and the only symptom was memory, on exactly the skewed
  inputs the guard exists for. This is the shape `CLAUDE.md` warns about: a green gate is not
  a green light.
- **What it explains:** the comment above `_MAX_SPILL_RECURSION` records a measurement of
  "86 buckets that reached the floor over budget, peak RSS identical (716 MB either way)" and
  concludes that lifting the ceiling needs per-group spillable state. That measurement was of
  a re-partition that moved no rows. The conclusion needs re-testing now that it moves them.
- **Proof:** `tests/unit/test_spill_resplit_salt.py`. It pins the defect directly —
  `test_an_unsalted_resplit_of_a_power_of_two_bucket_moves_no_rows` asserts the *old*
  behavior so it cannot return as an optimization — alongside the salted split reaching all
  8 sub-buckets, salt-0 identity at four partition counts, and equal keys staying together.

### #22 — `dist/spill.py` became a package along its real seam

- **Why:** the module sat at 498 lines against a 500-line limit, so #21 could not land without
  crossing it, and `dist/` was simultaneously at its 12-files-per-directory ceiling.
  Package-izing relieves both.
- **The seam:** *where the bytes go* (`scratch` — the scratch directory, the tiered store, the
  open-file cap, the morsel iterator that keeps the input from bounding peak memory, shared
  with `dist.spill_breakers`) versus *what the operator does with them* (`aggregate` —
  partition-and-spill aggregation and the `spill_collect` dispatcher).
- **The bucket reduce stayed with its caller**, deliberately. A test monkeypatches
  `_reduce_agg_bucket` on its module, and that only works if the caller resolves it there.
  The test's target was re-pointed at the defining module in the same change — verified, not
  assumed: patching the package leaves `aggregate._reduce_agg_bucket` untouched, so the trace
  would have come back empty and the test would have passed **while measuring nothing**.
  This is the exact hazard `.claude/rules/concurrent-agents.md` names about moved files.
- **The import path is unchanged.** Everything either half exposes is re-exported, including
  the underscore-prefixed names other modules and tests reach for.

### #23 — The out-of-core window re-splits a skewed bucket

- **Was:** `stream_spilling_window` hash-partitioned into a constant number of buckets and
  read each one **whole** for the kernel. A skewed `PARTITION BY` left one bucket far over
  the envelope, so it OOMed at exactly the point spilling was meant to prevent it. (This is
  the control-plane out-of-core window, a separate implementation from the in-engine one
  fixed in #13.)
- **Now:** the handle's resident size is checked *before* the read — which is the point, since
  the decision must happen without pulling in the thing that does not fit — and an over-large
  bucket is streamed into salted sub-buckets and recursed on, bounded by depth. The parent is
  released as soon as it has been re-partitioned, so recursion costs the same disk at each
  level rather than more.
- **`logical_nbytes`, not `nbytes`:** the uncompressed size is what reading it back costs in
  RAM; the on-disk size can be several times smaller for a compressible bucket and would let
  an over-large one through.
- **Why the salt is load-bearing here too:** without it, re-partitioning a power-of-two bucket
  count into another power of two reads the same low hash bits and moves no rows — the same
  inert recursion as #21.
- **Proof:** `test_spill_partitioned_window_resplits_a_skewed_bucket` — one key holding 4,000
  of 4,200 rows against an 8 KiB per-bucket envelope must both engage the split (a bucket
  processed below depth 0) and keep `row_number` equal to the in-memory kernel. This is a
  forward guard rather than a red-to-green regression: before the change there was no split
  to test.

### #24 — The out-of-core join re-splits a skewed bucket pair

- **Was:** `stream_spilling_join` hash-partitioned into a constant number of buckets and read
  **both** sides of each pair whole before joining. Under key skew one pair holds far more
  than its share, so it OOMed at exactly the point spilling was meant to prevent it — the
  standard reason a skewed Spark join dies.
- **Now:** both handles are measured before either is read, and an over-large pair is
  streamed into salted sub-buckets — the same salt and count on both sides, so equal keys
  co-locate and each sub-pair is an independent join whose union is the same relation.
- **The probe side counts:** the bucket count is sized from the build side alone, so a fact
  table with a hot key leaves a *probe* bucket orders of magnitude over the envelope even
  when every build bucket fits. Both sides are measured.
- **Proof:** `test_spill_join_resplits_a_skewed_bucket_pair`, for inner **and** the outer
  joins — the union of sub-pair joins is only the same relation if unmatched-row emission is
  per-pair correct, which is the subtle part.

### #25 — The distributed join reduce's sub-bucketing was inert too

- **Was:** `_spill_paths_to_subbuckets` re-partitioned a reducer's staged bucket with
  `partition_batches` — the same unsalted hash the shuffle used to build that bucket. When
  both counts are powers of two (the reduce uses 16), every row lands in `bucket & 15`: one
  sub-bucket, and a full write and re-read to change nothing. The same defect as #21, in the
  distributed half.
- **Now:** salted, so the re-partition actually separates keys the shuffle put together.

---

## Program 6 — A truncated spill file must fail, not shorten the answer

### #26 — Spill partitions carry a row count, checked on the way out

- **Was:** nothing verified that a partition read back what was written to it.
- **The failure this hides:** an Arrow IPC stream truncated **at a message boundary** — the
  last complete batch present, the end-of-stream marker gone — is byte-for-byte a shorter
  *valid* stream. The reader returns the batches it finds and reports success. Measured, not
  argued: five batches of 1,000 rows truncated after the third read back as **3,000 rows with
  no error at all**. The aggregate, join, or sort then computes a correct answer over the
  wrong rows.
- **Now:** `DiskSpillStore` counts rows per partition on append and compares on `read` and
  `drain`; a short read is `RuntimeError::SpillTruncated`, naming the partition, both counts,
  and how many rows would have vanished. The external sort makes the same comparison after
  streaming its final run, which `open_reader` cannot make for it.
- **Why it matters beyond the obvious:** every way this arises is a way a query returns a
  wrong answer rather than failing — a filesystem that reported a short write as success, a
  spill file that outlived the process still writing it, a truncation on a full disk the
  write path did not observe. It is the exact shape `CLAUDE.md` warns about: passes every
  gate while being wrong.
- **Proof:** `crates/bc-runtime/tests/spill_truncation.rs`. The decisive case is
  `a_cut_at_a_message_boundary_is_caught_by_the_row_count_alone`, which constructs the
  boundary exactly (a reference file of three batches, minus its 8-byte EOS marker) rather
  than hoping a halved file lands there — arrow catches a mid-message cut on its own, and
  only the boundary case needs the count. An intact partition must still read without
  complaint, or the check would be a way to fail every spilling query.

---

## Program 7 — The shuffle store's spill

`bc-transport`'s `PartitionStore` spills published shuffle buckets to disk under a byte cap.
It writes the Arrow IPC **file** format (footer of per-batch block offsets) rather than the
stream format, so a truncated file loses its footer and fails to open — it does not have the
silent-shortening hazard of Program 6. Two other things it shared.

### #27 — Its spill writes are buffered

- **Was:** `FileWriter` straight onto the `File`. Arrow issues a separate write per message
  and per buffer within it, so a spilled bucket of a few hundred morsels over a dozen columns
  was thousands of syscalls for bytes that coalesce into a handful of 1 MiB writes.
- **Now:** a 1 MiB `BufWriter`. A **fixed** size is right here, unlike the runtime store which
  budgets in total: this path writes exactly one file at a time, so the buffer cannot
  multiply by a fan-out.
- **The error path matters more than the buffer:** `into_inner` on the `BufWriter` is what
  surfaces a failed flush of the buffered tail. Dropping it would swallow that error and the
  rename that publishes the bucket would publish a **truncated** one — the single failure
  this path must not make silent.

### #28 — Its memory-release test no longer fails under load

- **Was:** the test measured `/proc/self/statm` deltas. That is process-wide, and `cargo test`
  runs the binary's tests in parallel threads, so a sibling test allocating during the
  measurement landed in the delta. It passed alone and failed inside a full-workspace run.
- **Now:** best of three trials. Not a weakened assertion — the noise can only *inflate* the
  bounded figure and cause a false failure; it cannot fabricate a pass. A run that observes
  the expected shape once has observed it.
- **Why it was worth fixing:** a test that fails only under load is the worst kind. It guards
  a real property (spilling returns pages to the process, not just to a counter) and teaches
  people to ignore it.

### #29 — The tiered store's row count is now actually checked

- **Was:** `SpillHandle.num_rows` has carried the count all along, with a docstring saying it
  is there so "a caller can detect a truncated bucket". **No caller did.** The capability
  existed; the guard did not.
- **Now:** `read_stream` — the single read path, which `read` and `read_reserved` both go
  through — counts as it yields and raises a typed `ResourceError` if the bucket comes back
  short, naming both counts and the likely causes.
- **Why this tier is more exposed than the Rust one:** it writes to the **remote** tier as
  well, where a partially-written object is an ordinary outcome of an interrupted upload,
  and the bucket may be read back much later by a different process.
- **Proof:** `test_a_truncated_bucket_is_refused_rather_than_read_short` constructs the
  boundary exactly (a reference bucket of three batches, minus its 8-byte end-of-stream
  marker) rather than cutting an arbitrary fraction — a cut inside a message arrow catches on
  its own, and only the boundary case needs the count. Paired with an intact-bucket test, so
  the check cannot become a way to fail every spilling query.

### #30 — A local volume that fills *mid-bucket* now carries over to the remote tier

- **Was:** the spill tier is chosen per bucket, by re-measuring free disk at open. That
  handles a volume that is *already* low; it cannot handle one that fills **during** a
  bucket — and at scale that is the case that matters, because a bucket is a whole
  partition. A 16-way spill of a terabyte writes ~60 GB per bucket, and the check at open is
  one sample taken before any of it was written. `memory.spill_remote_uri` is documented to
  keep an out-of-core query alive when local disk fills, and it only did so at bucket
  boundaries.
- **The reasoning that blocked this was wrong.** The code said mid-stream failover would
  "silently drop" the batches already streamed because they "are not retained". They are not
  retained *in memory* — they are **on disk**, in the bucket's own file. They can be read
  back and re-written to the remote tier **one batch at a time**, so the carry-over is
  bounded by a single batch and does not undo the spilling it is rescuing.
- **Now:** on `ENOSPC` with a remote tier configured, the local stream is abandoned (not
  finished — writing its end-of-stream marker would need the disk that just refused), the
  remote stream is opened, every complete batch is streamed across, the local file is
  deleted, and the failed batch is retried remotely.
- **It refuses rather than half-succeeds:** if the rows recovered do not match the rows the
  writer knows it wrote, the half-made remote bucket is removed and the caller raises the
  original actionable error. A partial carry-over would turn a loud out-of-space failure
  into a silently short bucket — strictly worse than the failure it replaces. The file may
  legitimately end mid-message (the write that filled the disk was partway through one), so
  the *read* is allowed to stop there; a failure of the *re-write* propagates, because that
  is the remote tier refusing and not something to absorb.
- **Proof:** `test_a_volume_that_fills_mid_bucket_carries_over_to_remote` — four batches, the
  volume filling after the second, must end as a REMOTE bucket holding all 400 rows with the
  local partial file gone. Paired with `test_a_mid_bucket_fill_still_fails_loudly_with_no_remote_tier`,
  because with nowhere to carry the bucket the query must still fail rather than pretend.

---

## Program 8 — Finishing the sweep: every grace fan-out, and the ASOF join

Programs 3 and 5 capped and skew-guarded the operators I had found. A sweep for the
*shape* — `div_ceil(budget).max(2)` — found three more sites that had never been capped or
had drifted away from the thing they claimed to mirror.

### #31 — The spilling ASOF join's fan-out is capped

- **Was:** `bytes.div_ceil(budget).max(2)`, unbounded. A side three orders of magnitude over
  the envelope asked for thousands of buckets per store, each receiving shards too small to
  write efficiently — and the bucket that still did not fit was materialized anyway, which is
  the failure the fan-out was trying to avoid.

### #32 — The spilling ASOF join re-splits a skewed `by`-group bucket

- **Was:** both sides of every bucket pair were materialized whole. The fan-out is sized from
  the *larger side's total*, which says nothing about how any one `by` value is distributed,
  so a hot group OOMed the bucket.
- **Now:** the same measure-then-split guard, and legal here for the hash join's reason plus
  one more: a nearest-`on` match never crosses a `by` group, so re-partitioning **by the `by`
  keys** keeps every group whole in one sub-bucket and each sub-pair stays an independent
  ASOF join. Ordering within a group is untouched — rows move only *between* sub-buckets, and
  `asof_join_batches` orders what it is given.
- **Proof:** `spilling_asof_join_with_skewed_by_groups_matches_in_memory` — one `by` group
  carrying the bulk of both sides, against the in-memory oracle.

### #33 — `mixed_spill`'s constant-state fan-out had drifted from what it mirrors

- Its doc comment said it "mirrors `par::grace_partitions`". That one grew a cap in Program 3;
  this one did not. Both now route through `grace_bucket_count`, and the reason is stated
  once rather than copied.

### #34 — The distributed reducer's grace fan-out uses the shared cap

- `reduce_grace_partitions` capped at 4,096 — the same number lowered to 256 in #17, for the
  same reason: a disk-backed store creates a file per partition, and `combine_finalize_spilling`
  re-partitions anything still over budget once it sees the true in-memory size.

### #35 — The shm tests no longer collide across processes

- **Was:** the same-node shm root is a **cross-process** namespace, and a peer's directory is
  named only after its address — which the tests hardcoded (`host_3:55503`). Two test
  processes of this crate running at once therefore shared directories, and one's
  `clear_shared` deleted the file the other had just published. Reproduced at roughly **one
  run in four**, surfacing as an intermittent `NotFound` from `publish_shared` or a fetch that
  found nothing.
- **Now:** addresses and planted peer directories embed the pid. **0 failures in 20 runs.**
- **Why it belongs in this ledger:** a full-workspace build, a second agent's suite, or CI
  running two jobs on one machine all trigger it, and an intermittently-red transport suite is
  how a real spill regression gets waved through.

### #36 — The external sort checks that its merge returned every row it was given

- **Was:** the merge passes were unverified. Each pass writes runs and reads them back, so a
  spill file that lost its tail turns the sort into a **sorted prefix** of the relation rather
  than an error — an IPC stream truncated at a message boundary reads back as a shorter valid
  stream (#26).
- **Now:** pass 0 counts rows in; after the merge loop the final run's row count (which the
  store already tracks) is compared against it. **No extra I/O**, and it covers every merge
  pass at once, for the sort and for all five quantile/median/mode/histogram callers that
  stream the final run themselves.
- **It also catches a plain bug:** a merge that simply dropped rows. No result comparison in a
  spilling test would notice, because under a forced spill the spilled path is the *only* path
  being run — there is no in-memory answer alongside it to disagree with.
- **Soundness of the bound:** partition 0 of the returned store holds the whole relation in
  every case — whether pass 0 produced one run (no merge ran) or many (the last pass produced
  one group). Compared with `<` rather than `!=`, so a hypothetical over-count is not a
  failure.
- **Proof:** every existing spilling-sort test now runs this check unconditionally and passes,
  which is the evidence that it does not false-fire; the negative case is pinned at the store
  level by `crates/bc-runtime/tests/spill_truncation.rs`, which is the same row-count
  mechanism.

---

## Program 9 — Measuring the relation honestly

Every spill decision is made from one number: how big the relation is. An over-count there
does not fail anything — it makes the engine spill a query that would have fitted, which reads
as slowness rather than as a bug.

### #37 — A shared dictionary is counted once for the relation, not once per morsel

- **Was:** `batch_bytes` summed each column's slice size. Morsels of a dictionary-encoded
  column all point at the *same* values array, but each morsel's slice size includes that whole
  dictionary — so the relation was counted as `morsels x dictionary` instead of
  `dictionary + indices`.
- **Measured in the real code path:** 600 morsels over a 50,000-entry string dictionary
  reported **609 MB for 40 MB resident — a 15.1x over-count**.
- **Why it is the common case:** dictionary encoding exists *for* low-cardinality string
  columns, so this is the most common encoding's normal shape. The consequence is a sort,
  join, window, or grace fan-out spilling when it would have fitted in memory, and the
  streaming breakers handing off to the materializing executor 15x too early.
- **Now:** each distinct dictionary is charged once, identified by its buffer address —
  unambiguous because every morsel being measured is alive, so two live dictionaries cannot
  share one. The walk covers the whole array tree, so a dictionary nested in a struct or list
  is deduplicated too. A relation with no dictionary anywhere in its schema takes a fast path
  and is measured byte-identically to before.
- **This is the same bug the function's own docstring already recorded, one level down.** The
  slice fix ("122 morsels made this report 3.9 GB") solved re-counting a shared *buffer*; this
  solves re-counting a shared *dictionary*, which the slice fix does not reach.
- **Proof:** four unit tests beside the function — the shared-dictionary case (asserting both
  that the count is honest and that the naive sum it replaces was many times larger), two
  distinct dictionaries both counted, a struct-nested dictionary deduplicated, and a
  dictionary-free relation measured `assert_eq!`-identically to the naive sum.
