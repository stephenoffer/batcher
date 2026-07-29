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
- **Now:** a 1 MiB `BufWriter` sits under every spill writer. The bytes on disk are
  identical, so nothing about the read path or the result changes.
- **Why it matters:** a spill of a few thousand morsels over a dozen columns spent hundreds
  of thousands of syscalls on data that coalesces into a handful of 1 MiB writes. This is
  pure throughput on the path that is already the slowest thing in the query.
- **Proof:** covered by the existing spill suites for correctness; the syscall reduction is
  structural (buffer size vs. per-buffer write).

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

---

## Program 2 — The external sort: fewer, larger runs

### #6 — Pass 0 builds runs to a byte target instead of one run per morsel

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

### #12 — Scratch abandoned by an OOM-killed process is reclaimed

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

### #13 — A full spill filesystem says so, and says which one

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

### #7 — The run target is derived from the operator's memory envelope

- **Was:** n/a (runs were one morsel).
- **Now:** the sort uses a quarter of the operator's envelope, floored at 1 MiB and capped at
  the 64 MiB module default; callers with no envelope (the quantile/median spill paths) take
  the default. A quarter leaves room inside the envelope for the run's own concat and sort
  scratch.
