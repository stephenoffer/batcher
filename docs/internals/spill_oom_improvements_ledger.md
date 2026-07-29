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
