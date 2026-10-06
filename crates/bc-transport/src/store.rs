//! Internal partition store: the in-memory registry mapping a ticket string to
//! the batches served under it, plus the per-exchange in-flight gauge used to
//! prove the credit bound.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::array::RecordBatch;
use tokio::sync::RwLock;

/// Tracks, for one partition's exchange, how many batches the producer has
/// pushed past the consumer (current in-flight) and the high-water mark of that
/// count. Used to *prove* the credit bound in tests, and harmless in prod (a
/// couple of relaxed atomic ops per batch).
#[derive(Default)]
pub(crate) struct InflightGauge {
    current: std::sync::atomic::AtomicI64,
    max: std::sync::atomic::AtomicI64,
    /// How many credit-grant *control messages* the consumer sent (one per pump
    /// wakeup, regardless of how many credits it carried). With per-batch grants this
    /// equals the batch count; with low-watermark batched refill it is ~`2N/window` —
    /// so it *proves* the control-message reduction the batched refill exists for.
    grant_messages: std::sync::atomic::AtomicI64,
}

impl InflightGauge {
    /// Producer is about to send one more batch: bump in-flight and the max.
    ///
    /// The high-water update uses `AcqRel`: the gauge is read to *prove* the
    /// credit bound was honored, so the max must not be under-reported on a weak
    /// memory model. It is off the per-batch data path, so the stronger ordering is
    /// negligible.
    pub(crate) fn on_send(&self) {
        use std::sync::atomic::Ordering::{AcqRel, Relaxed};
        let now = self.current.fetch_add(1, Relaxed) + 1;
        self.max.fetch_max(now, AcqRel);
    }

    /// Consumer acked one batch (a top-up credit arrived): drop in-flight.
    ///
    /// Saturates at zero. The count is a measurement of batches the producer has handed to
    /// the encoder and the consumer has not yet acknowledged, so a negative value is not a
    /// smaller number of batches — it is a broken instrument. It could go negative because
    /// acks are driven by *granted credits*, and a consumer that grants more than it
    /// consumed pushed the counter below zero and pinned the high-water `max` at whatever
    /// it had reached first. That matters because `max` is what the crate's flow-control
    /// tests read to *prove* the credit bound: an over-granting consumer could make the
    /// proof pass while the bound it certifies was not being enforced.
    pub(crate) fn on_ack(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        // A CAS loop rather than `fetch_update`, which Rust 1.99 deprecates in favour of
        // `try_update` -- and that is newer than this workspace's 1.89 MSRV.
        let mut cur = self.current.load(Relaxed);
        while let Err(seen) =
            self.current
                .compare_exchange_weak(cur, cur.saturating_sub(1).max(0), Relaxed, Relaxed)
        {
            cur = seen;
        }
    }

    /// One credit-grant control message arrived from the consumer (independent of
    /// how many credits it carried). Counts the exchange's control-message traffic.
    pub(crate) fn on_grant_message(&self) {
        self.grant_messages
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// High-water mark of simultaneously in-flight batches.
    pub(crate) fn max(&self) -> i64 {
        self.max.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total credit-grant control messages the consumer sent for this exchange.
    pub(crate) fn grant_messages(&self) -> i64 {
        self.grant_messages
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Where a published partition's batches currently live.
enum Body {
    /// In this process's heap — the fast path a reducer serves straight from.
    Memory(Arc<Vec<RecordBatch>>),
    /// Written to local disk and dropped from the heap. Read back on fetch.
    Spilled(PathBuf),
}

/// One registered partition: its body, its in-flight gauge, and its footprint.
pub(crate) struct Partition {
    body: Body,
    gauge: Arc<InflightGauge>,
    /// Resident bytes this partition holds *while in memory*, measured at registration.
    /// A spilled partition still knows this: it is what returns to the total if it is
    /// ever read back, and what makes the spill decision reversible in principle.
    nbytes: usize,
    /// A spiller has chosen this partition and is writing it out, without the map lock.
    /// Other spillers skip it so two never write the same bucket; readers still get the
    /// in-memory copy until the write commits.
    spilling: bool,
}

impl Partition {
    /// The batches, reading them back from disk if this partition was spilled.
    ///
    /// A spilled read deliberately does **not** re-populate the heap copy. The store spilled
    /// it because memory was short; silently restoring it on the first fetch would undo the
    /// bound exactly when it is being relied on, and a bucket is typically fetched once.
    ///
    /// **A spilled read that fails is an error, never an empty bucket.** The spill file is
    /// the partition's only copy, so a file that has gone missing, been truncated, or can no
    /// longer be opened means the rows are lost. This used to be `.ok()`'d into `None`, which
    /// the server reported as an unknown ticket and the consumer read as an empty bucket: a
    /// disk fault became a successful query with fewer rows. The error carries the ticket so
    /// the server can tell the reducer which bucket is lost, and the reducer's recovery loop
    /// can recompute that mapper.
    fn batches(&self, ticket: &str) -> Result<Arc<Vec<RecordBatch>>, SpillReadError> {
        match &self.body {
            Body::Memory(b) => Ok(b.clone()),
            Body::Spilled(path) => crate::shared::read_ipc_file_strict(path)
                .map(Arc::new)
                .map_err(|cause| SpillReadError {
                    ticket: ticket.to_string(),
                    path: path.clone(),
                    cause,
                }),
        }
    }

    fn in_memory(&self) -> bool {
        matches!(self.body, Body::Memory(_))
    }

    /// Delete this partition's spill file, if it has one. Best-effort: a leftover file is
    /// wasted disk, and failing an eviction over it would be worse.
    fn discard_spill_file(&self) {
        if let Body::Spilled(path) = &self.body {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// A spilled partition whose file could not be read back.
///
/// Registered-but-unreadable is a different state from never-registered, and the whole point
/// of this type is that the two never collapse into one `None` again.
#[derive(Debug)]
pub(crate) struct SpillReadError {
    pub(crate) ticket: String,
    pub(crate) path: PathBuf,
    pub(crate) cause: std::io::Error,
}

impl std::fmt::Display for SpillReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "spilled shuffle bucket {} could not be read back from {}: {}",
            self.ticket,
            self.path.display(),
            self.cause
        )
    }
}

/// A publish the store could neither hold under its cap nor write to disk.
///
/// The policy this type carries: when the cap is set and spilling fails, the publish fails.
/// The store used to stop spilling on the first write error and keep every bucket resident,
/// so a full or unwritable scratch disk turned into unbounded memory growth on exactly the
/// worker that was already short of it. Refusing the new bucket keeps the store inside its
/// envelope; the map task fails with an error naming the disk, and the bucket's memory goes
/// back the moment the caller drops it.
#[derive(Debug)]
pub(crate) struct SpillWriteError {
    pub(crate) ticket: String,
    pub(crate) cap: usize,
    pub(crate) retained: usize,
    pub(crate) cause: std::io::Error,
}

impl std::fmt::Display for SpillWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "shuffle bucket {} cannot be published: the shuffle store holds {} bytes against a \
             {}-byte cap and spilling to disk failed ({}); free or enlarge the shuffle spill \
             directory, or raise the shuffle store cap",
            self.ticket, self.retained, self.cap, self.cause
        )
    }
}

/// What a lookup found: `Ok(None)` is a ticket that was never published (or was evicted),
/// `Err` one that was published and spilled but whose file is now unreadable.
pub(crate) type Lookup<T> = Result<Option<T>, SpillReadError>;

/// Make a ticket safe as a filename (`plan/stage/src/dst/epoch` → `plan_stage_src_dst_epoch`).
fn sanitize_ticket(ticket: &str) -> String {
    ticket
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Bytes a batch actually holds, including buffer padding and any slice it shares.
///
/// `get_array_memory_size`, not the logical size: what matters here is the memory the
/// process cannot give back while the partition is registered, and a sliced batch keeps
/// its whole parent buffer alive.
fn batch_bytes(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::get_array_memory_size).sum()
}

/// In-memory registry mapping a ticket string to the batches served under it.
///
/// The store keeps a running byte total. It is the single largest thing the control
/// plane's memory accounting cannot see: Carbonite's buffer pool tracks reservations the
/// engine *asks* for, and a published shuffle partition is never asked for — it is simply
/// held until a reducer fetches it. `PressureMonitor` names this store by name as the
/// reason it has to fall back to reading process RSS. A number the store keeps itself is
/// cheaper than that inference and, unlike RSS, attributes the memory to the shuffle.
pub(crate) struct PartitionStore {
    partitions: RwLock<HashMap<String, Partition>>,
    /// Sum of every *in-memory* partition's `nbytes`. Atomic so a reader does not have to
    /// take the map lock — the point is to be cheap enough to poll.
    retained: AtomicUsize,
    /// Byte cap above which buckets spill to disk; `0` is unbounded.
    ///
    /// Read from the process tunable **once, at construction**, not per publish. A store's
    /// memory bound should not shift under it mid-query — the tunable is set once per
    /// worker by the control plane — and a captured value also makes the bound testable
    /// without a global that concurrent tests would fight over.
    cap: usize,
    /// The scratch root spilled buckets go under (`crate::shuffle_spill_root`), captured at
    /// construction for the same reason as `cap`.
    spill_root: PathBuf,
    /// Scratch directory for spilled buckets, created on first spill.
    spill_dir: std::sync::OnceLock<Option<PathBuf>>,
    /// Makes every spill file name unique, so an abandoned write (its partition was replaced
    /// or removed mid-write) can never collide with a later one for the same ticket.
    spill_seq: std::sync::atomic::AtomicU64,
}

/// Remove every `<pid>_<store>` directory under `root` whose process no longer exists.
///
/// Linux only, where `/proc/<pid>` answers the question; elsewhere nothing is removed. A pid
/// the kernel has since reused keeps its directory, which is the safe direction to be wrong.
fn sweep_orphaned_spill_dirs(root: &std::path::Path) {
    let proc_fs = std::path::Path::new("/proc");
    if !proc_fs.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(pid) = name.to_str().and_then(|n| n.split('_').next()) else {
            continue;
        };
        if pid.parse::<u32>().is_ok() && !proc_fs.join(pid).exists() {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

impl Default for PartitionStore {
    fn default() -> Self {
        Self::with_cap(crate::shuffle_store_cap())
    }
}

impl PartitionStore {
    /// A store bounded at `cap` bytes of resident buckets (`0` = unbounded).
    pub(crate) fn with_cap(cap: usize) -> Self {
        Self::with_cap_in(cap, crate::shuffle_spill_root())
    }

    /// A store bounded at `cap` that spills under `spill_root`.
    pub(crate) fn with_cap_in(cap: usize, spill_root: PathBuf) -> Self {
        Self {
            partitions: RwLock::new(HashMap::new()),
            retained: AtomicUsize::new(0),
            cap,
            spill_root,
            spill_dir: std::sync::OnceLock::new(),
            spill_seq: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Publish `batches` under `ticket`, reserving their bytes against the cap first.
    ///
    /// **Reserve, then admit.** The bytes are charged to the running total before the bucket
    /// is visible. If that takes the store over its cap, resident buckets are spilled to make
    /// room; if there is still no room, the new bucket goes straight to disk and is never
    /// held in memory at all. Charging after insertion, as this once did, let concurrent
    /// publishes each see a total that did not yet include the others.
    ///
    /// **Disk failure is a refusal, not an overrun** (see [`SpillWriteError`]). With no cap
    /// configured nothing changes: the bucket is held in memory as it always was.
    pub(crate) async fn register(
        &self,
        ticket: String,
        batches: Vec<RecordBatch>,
    ) -> Result<(), SpillWriteError> {
        let nbytes = batch_bytes(&batches);
        let batches = Arc::new(batches);
        let reserved = self.retained.fetch_add(nbytes, Ordering::Relaxed) + nbytes;
        let mut body = Body::Memory(batches.clone());
        if self.cap > 0 && reserved > self.cap {
            // A failed victim write is not decisive on its own: the new bucket's own write
            // below is the last resort, and only its failure refuses the publish.
            let _ = self.spill_resident_down_to(self.cap).await;
            if self.retained.load(Ordering::Relaxed) > self.cap && nbytes > 0 {
                // Still no room: the new bucket is the one that goes to disk, and its
                // reservation is handed back either way.
                let written = self.write_spill(&ticket, &batches);
                let retained = self.retained.fetch_sub(nbytes, Ordering::Relaxed) - nbytes;
                match written {
                    Ok(path) => body = Body::Spilled(path),
                    Err(cause) => {
                        return Err(SpillWriteError {
                            ticket,
                            cap: self.cap,
                            retained,
                            cause,
                        })
                    }
                }
            }
        }
        let previous = self.partitions.write().await.insert(
            ticket,
            Partition {
                body,
                gauge: Arc::new(InflightGauge::default()),
                nbytes,
                spilling: false,
            },
        );
        // Re-registering a ticket (a recompute republishing under a bumped epoch, or a
        // retried map task) replaces the entry. Charging the new bytes without crediting
        // back the old ones makes the total drift up forever, and a monotonically rising
        // "retained bytes" that never falls is worse than no number at all: it reads as a
        // leak in the one place someone would look to find one.
        if let Some(old) = previous {
            if old.in_memory() {
                self.retained.fetch_sub(old.nbytes, Ordering::Relaxed);
            }
            old.discard_spill_file();
        }
        Ok(())
    }

    /// Spill resident buckets, largest first, until `retained <= floor` or none is left.
    ///
    /// **The gap this closes.** Everything else Carbonite bounds is *reserved* memory — an
    /// operator asks the pool before it allocates, and spills when refused. A published
    /// shuffle bucket is never asked for: a mapper hands it to this store and it stays
    /// resident until a reducer fetches it. With `workers` mappers each producing `workers`
    /// buckets, a node holds its whole share of the shuffle in anonymous memory that no
    /// reservation covers and the kernel cannot reclaim. That is the classic shuffle OOM.
    ///
    /// Largest-first, because the point is to get back under the cap in the fewest reads
    /// later: one big bucket costs one re-read, many small ones cost many. Spilling is
    /// result-preserving — the same batches come back through the Arrow IPC round-trip.
    ///
    /// **The disk write happens outside the map lock.** The lock is taken to choose a victim
    /// and again to commit it, never across the write: holding the write guard through a
    /// multi-megabyte write stalled every fetch on the worker, including the ones that would
    /// have drained the store. Returns the last write error if a write failed.
    async fn spill_resident_down_to(&self, floor: usize) -> std::io::Result<usize> {
        let mut freed = 0usize;
        let mut last_err = None;
        while self.retained.load(Ordering::Relaxed) > floor {
            let claimed = Self::claim_victim(&mut *self.partitions.write().await);
            let Some((ticket, batches)) = claimed else {
                break;
            };
            let written = self.write_spill(&ticket, &batches);
            let mut guard = self.partitions.write().await;
            match written {
                Ok(path) => freed += self.commit_spill(&mut guard, &ticket, &batches, path),
                Err(e) => {
                    Self::release_claim(&mut guard, &ticket, &batches);
                    last_err = Some(e);
                    break;
                }
            }
        }
        match last_err {
            Some(e) => Err(e),
            None => Ok(freed),
        }
    }

    /// Free at least `target` bytes on demand, returning what was actually freed.
    ///
    /// This is the store's half of a *cooperative* reservation: when an operator cannot get
    /// memory from `bc-resource`'s pool, the pool asks its registered consumers to yield
    /// some, largest first, before refusing. Published shuffle output is the ideal thing to
    /// ask — it is finished work sitting idle waiting to be collected, so spilling it costs
    /// a re-read and stalls nobody, where spilling a half-built hash table costs the
    /// operator that is actively using it.
    ///
    /// **Synchronous and non-blocking on purpose.** The pool calls this from whatever
    /// thread lost a reservation, which may be a tokio worker; blocking on an async lock
    /// there would deadlock the runtime serving the very fetches that would drain this
    /// store. `try_write` instead: if the map is busy, this stops, which the `Spillable`
    /// contract explicitly allows and the pool treats as "this consumer cannot help right
    /// now". The disk write itself runs with no lock held, as in the async path.
    ///
    /// Independent of `cap`: a store with no configured cap still answers, because the
    /// caller here is real memory pressure rather than a configured bound.
    pub(crate) fn try_spill_at_least(&self, target: usize) -> usize {
        if target == 0 {
            return 0;
        }
        let floor = self.retained.load(Ordering::Relaxed).saturating_sub(target);
        let mut freed = 0usize;
        while self.retained.load(Ordering::Relaxed) > floor {
            let Ok(mut guard) = self.partitions.try_write() else {
                break;
            };
            let Some((ticket, batches)) = Self::claim_victim(&mut guard) else {
                break;
            };
            drop(guard);
            let written = self.write_spill(&ticket, &batches);
            // Committing needs the lock again. Readers hold it only to clone an `Arc`, so a
            // short spin acquires it; a claim must not be left set, or no spiller could ever
            // choose that bucket again.
            let mut guard = loop {
                if let Ok(guard) = self.partitions.try_write() {
                    break guard;
                }
                std::thread::yield_now();
            };
            let got = match written {
                Ok(path) => self.commit_spill(&mut guard, &ticket, &batches, path),
                Err(_) => {
                    Self::release_claim(&mut guard, &ticket, &batches);
                    0
                }
            };
            if got == 0 {
                break; // a failed or superseded write: stop rather than retry the same victim
            }
            freed += got;
        }
        freed
    }

    /// Mark the largest resident, unclaimed bucket as being spilled, and hand back its batches.
    fn claim_victim(
        guard: &mut HashMap<String, Partition>,
    ) -> Option<(String, Arc<Vec<RecordBatch>>)> {
        let (ticket, p) = guard
            .iter_mut()
            .filter(|(_, p)| p.in_memory() && !p.spilling)
            .max_by_key(|(_, p)| p.nbytes)?;
        let Body::Memory(batches) = &p.body else {
            return None;
        };
        p.spilling = true;
        Some((ticket.clone(), batches.clone()))
    }

    /// Clear a claim whose write failed, if the same bucket is still registered.
    fn release_claim(
        guard: &mut HashMap<String, Partition>,
        ticket: &str,
        batches: &Arc<Vec<RecordBatch>>,
    ) {
        if let Some(p) = guard.get_mut(ticket) {
            if matches!(&p.body, Body::Memory(b) if Arc::ptr_eq(b, batches)) {
                p.spilling = false;
            }
        }
    }

    /// Swap a claimed bucket's body to its spill file, if it is still the bucket that was
    /// written. A bucket removed or re-registered during the write keeps its new state and
    /// the orphaned file is deleted. Returns the bytes freed.
    fn commit_spill(
        &self,
        guard: &mut HashMap<String, Partition>,
        ticket: &str,
        batches: &Arc<Vec<RecordBatch>>,
        path: PathBuf,
    ) -> usize {
        match guard.get_mut(ticket) {
            Some(p) if matches!(&p.body, Body::Memory(b) if Arc::ptr_eq(b, batches)) => {
                p.body = Body::Spilled(path);
                p.spilling = false;
                self.retained.fetch_sub(p.nbytes, Ordering::Relaxed);
                p.nbytes
            }
            _ => {
                let _ = std::fs::remove_file(&path);
                0
            }
        }
    }

    /// Write `batches` to a fresh spill file for `ticket`. No lock is held by the caller.
    fn write_spill(&self, ticket: &str, batches: &[RecordBatch]) -> std::io::Result<PathBuf> {
        let dir = self.spill_dir().ok_or_else(|| {
            std::io::Error::other("the shuffle spill directory could not be created")
        })?;
        let seq = self.spill_seq.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("{}_{seq}.arrow", sanitize_ticket(ticket)));
        crate::shared::write_ipc_file(&path, batches)?;
        Ok(path)
    }

    /// This store's spill directory, created once on first use.
    ///
    /// Creating it first removes the directories of stores whose process is gone. A store
    /// removes its own spill files as their tickets are dropped, but a worker that is killed
    /// -- a fleet teardown, a lost node -- never gets to, and each such worker left its
    /// spilled buckets behind: on the SF1000 TPC-H suite they reached 72 GB on one node and,
    /// with other orphaned scratch, filled a 145 GB disk until no task could start there.
    fn spill_dir(&self) -> Option<&PathBuf> {
        self.spill_dir
            .get_or_init(|| {
                let root = self.spill_root.join("batcher_shuffle_spill");
                sweep_orphaned_spill_dirs(&root);
                let dir = root.join(format!("{}_{:p}", std::process::id(), self));
                crate::shared::create_private_dir(&dir).ok().map(|()| dir)
            })
            .as_ref()
    }

    /// Bytes currently held by registered partitions.
    ///
    /// The shuffle's resident footprint, which is anonymous memory the kernel cannot
    /// reclaim and which no reservation accounts for.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained.load(Ordering::Relaxed)
    }

    pub(crate) async fn get(&self, ticket: &str) -> Lookup<Arc<Vec<RecordBatch>>> {
        match self.partitions.read().await.get(ticket) {
            None => Ok(None),
            Some(p) => p.batches(ticket).map(Some),
        }
    }

    /// Fetch both the batches and the in-flight gauge for an exchange.
    pub(crate) async fn get_with_gauge(
        &self,
        ticket: &str,
    ) -> Lookup<(Arc<Vec<RecordBatch>>, Arc<InflightGauge>)> {
        let guard = self.partitions.read().await;
        let Some(p) = guard.get(ticket) else {
            return Ok(None);
        };
        Ok(Some((p.batches(ticket)?, p.gauge.clone())))
    }

    /// The in-flight gauge for a ticket (for tests/observability).
    pub(crate) async fn gauge(&self, ticket: &str) -> Option<Arc<InflightGauge>> {
        self.partitions
            .read()
            .await
            .get(ticket)
            .map(|p| p.gauge.clone())
    }

    /// Drop one published partition once its reducers have fetched it, freeing its
    /// batches. The store is otherwise append-only, so without this a long-lived
    /// worker accumulates every partition of every stage/epoch until it dies (OOM).
    pub(crate) async fn remove(&self, ticket: &str) {
        if let Some(p) = self.partitions.write().await.remove(ticket) {
            if p.in_memory() {
                self.retained.fetch_sub(p.nbytes, Ordering::Relaxed);
            }
            p.discard_spill_file();
        }
    }

    /// Drop every partition whose ticket begins with `prefix` (e.g. `"{plan_id}/"`
    /// to evict a whole finished plan, or `"{plan_id}/{stage}/"` one stage).
    pub(crate) async fn remove_prefix(&self, prefix: &str) {
        let mut freed = 0usize;
        self.partitions.write().await.retain(|ticket, p| {
            let keep = !ticket.starts_with(prefix);
            if !keep {
                if p.in_memory() {
                    freed += p.nbytes;
                }
                p.discard_spill_file();
            }
            keep
        });
        self.retained.fetch_sub(freed, Ordering::Relaxed);
    }

    /// Drop every published partition. Called at plan teardown to return the
    /// worker's shuffle memory to the OS without tearing down the actor.
    pub(crate) async fn clear(&self) {
        let mut guard = self.partitions.write().await;
        for p in guard.values() {
            p.discard_spill_file();
        }
        guard.clear();
        self.retained.store(0, Ordering::Relaxed);
    }

    /// Number of partitions currently retained (telemetry / leak tests).
    pub(crate) async fn len(&self) -> usize {
        self.partitions.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store whose process was killed never removes its spill directory; the next store
    /// on the node must, and must leave a live process's alone.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_new_store_sweeps_only_dead_owners_spill_dirs() {
        let root = std::env::temp_dir().join(format!("bc_sweep_test_{}", std::process::id()));
        let dead = root.join("4294967294_0xdead");
        let live = root.join(format!("{}_0xbeef", std::process::id()));
        let other = root.join("not-a-store");
        for d in [&dead, &live, &other] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("bucket.arrow"), b"x").unwrap();
        }
        super::sweep_orphaned_spill_dirs(&root);
        assert!(!dead.exists(), "a dead process's spill dir must be removed");
        assert!(live.exists(), "a live process's spill dir must survive");
        assert!(
            other.exists(),
            "a directory that is not a store's must survive"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};

    fn one_batch(v: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![v]))]).unwrap()
    }

    #[tokio::test]
    async fn register_then_get_returns_batches() {
        let store = PartitionStore::default();
        store
            .register("7/0/0/0".into(), vec![one_batch(1), one_batch(2)])
            .await
            .unwrap();
        let got = store.get("7/0/0/0").await.unwrap().expect("registered");
        assert_eq!(got.len(), 2);
        assert!(store.get("7/0/0/9").await.unwrap().is_none()); // unregistered ticket
    }

    #[tokio::test]
    async fn gauge_tracks_inflight_high_water() {
        let store = PartitionStore::default();
        store
            .register("p/0/0/0".into(), vec![one_batch(1)])
            .await
            .unwrap();
        let (_b, gauge) = store.get_with_gauge("p/0/0/0").await.unwrap().unwrap();
        // Two sends in flight, then one ack: current drops but the max is sticky.
        gauge.on_send();
        gauge.on_send();
        gauge.on_ack();
        assert_eq!(gauge.max(), 2, "high-water mark must not be under-reported");
        assert!(store.gauge("p/0/0/0").await.is_some());
    }

    #[tokio::test]
    async fn remove_and_clear_free_partitions() {
        let store = PartitionStore::default();
        store
            .register("9/0/0/0".into(), vec![one_batch(1)])
            .await
            .unwrap();
        store
            .register("9/0/0/1".into(), vec![one_batch(2)])
            .await
            .unwrap();
        assert_eq!(store.len().await, 2);
        store.remove("9/0/0/0").await;
        assert_eq!(store.len().await, 1);
        assert!(store.get("9/0/0/0").await.unwrap().is_none());
        store.clear().await;
        assert_eq!(store.len().await, 0);
    }

    #[tokio::test]
    async fn retained_bytes_tracks_registration_and_eviction() {
        let store = PartitionStore::default();
        assert_eq!(store.retained_bytes(), 0);

        store
            .register("b/0/0/0".into(), vec![one_batch(1)])
            .await
            .unwrap();
        let one = store.retained_bytes();
        assert!(
            one > 0,
            "a registered partition holds memory nothing accounts for"
        );

        store
            .register("b/0/0/1".into(), vec![one_batch(2)])
            .await
            .unwrap();
        assert_eq!(store.retained_bytes(), one * 2);

        store.remove("b/0/0/0").await;
        assert_eq!(store.retained_bytes(), one);
        store.remove("b/0/0/0").await; // already gone — must not double-credit
        assert_eq!(store.retained_bytes(), one);

        store.clear().await;
        assert_eq!(store.retained_bytes(), 0);
    }

    #[tokio::test]
    async fn re_registering_a_ticket_does_not_drift_the_total() {
        // A recompute republishes under the same ticket. Charging the new bytes without
        // crediting the old ones makes the total rise forever and read as a leak.
        let store = PartitionStore::default();
        store
            .register("r/0/0/0".into(), vec![one_batch(1)])
            .await
            .unwrap();
        let one = store.retained_bytes();
        store
            .register("r/0/0/0".into(), vec![one_batch(2), one_batch(3)])
            .await
            .unwrap();
        assert_eq!(store.len().await, 1);
        assert_eq!(
            store.retained_bytes(),
            one * 2,
            "the superseded bytes were not credited back"
        );
    }

    #[tokio::test]
    async fn remove_prefix_credits_back_every_partition_it_evicts() {
        let store = PartitionStore::default();
        store
            .register("p9/0/0/0".into(), vec![one_batch(1)])
            .await
            .unwrap();
        store
            .register("p9/1/0/0".into(), vec![one_batch(2)])
            .await
            .unwrap();
        store
            .register("p8/0/0/0".into(), vec![one_batch(3)])
            .await
            .unwrap();
        let all = store.retained_bytes();

        store.remove_prefix("p9/").await;
        assert_eq!(
            store.retained_bytes(),
            all / 3,
            "evicted bytes stayed on the books"
        );
    }

    #[tokio::test]
    async fn remove_prefix_evicts_matching_stage() {
        let store = PartitionStore::default();
        store
            .register("9/0/0/0".into(), vec![one_batch(1)])
            .await
            .unwrap(); // plan 9, stage 0
        store
            .register("9/1/0/0".into(), vec![one_batch(2)])
            .await
            .unwrap(); // plan 9, stage 1
        store
            .register("8/0/0/0".into(), vec![one_batch(3)])
            .await
            .unwrap(); // plan 8
        store.remove_prefix("9/0/").await; // evict only plan 9, stage 0
        assert!(store.get("9/0/0/0").await.unwrap().is_none());
        assert!(store.get("9/1/0/0").await.unwrap().is_some());
        assert!(store.get("8/0/0/0").await.unwrap().is_some());
        // A whole-plan prefix evicts every stage of that plan.
        store.remove_prefix("9/").await;
        assert!(store.get("9/1/0/0").await.unwrap().is_none());
        assert_eq!(store.len().await, 1);
    }

    fn wide_batch(v: i64, n: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let vals: Vec<i64> = (0..n as i64).map(|i| i + v).collect();
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals))]).unwrap()
    }

    /// A published bucket is never *reserved* — a mapper hands it over and it stays
    /// resident until a reducer fetches it — so with `workers` mappers each producing
    /// `workers` buckets a node holds its whole share of the shuffle in memory that no
    /// reservation covers. This is the bound for that.
    #[tokio::test]
    async fn the_store_spills_to_disk_when_it_exceeds_its_cap() {
        let one = batch_bytes(&[wide_batch(0, 4096)]);
        let store = PartitionStore::with_cap(one * 2);

        for i in 0..6 {
            store
                .register(format!("50/0/{i}/0/0"), vec![wide_batch(i * 1000, 4096)])
                .await
                .unwrap();
        }

        assert_eq!(store.len().await, 6, "spilling must not lose partitions");
        assert!(
            store.retained_bytes() <= one * 2,
            "resident bytes {} stayed above the cap {}",
            store.retained_bytes(),
            one * 2,
        );
    }

    /// Spilling is a memory strategy, not a semantics: every bucket must read back
    /// byte-identical, whichever side of the cap it ended up on.
    #[tokio::test]
    async fn every_spilled_bucket_reads_back_identically() {
        let store = PartitionStore::with_cap(1); // a one-byte cap: everything spills

        let expected: Vec<Vec<RecordBatch>> = (0..4)
            .map(|i| vec![wide_batch(i * 100, 512), wide_batch(i * 100 + 7, 256)])
            .collect();
        for (i, batches) in expected.iter().enumerate() {
            store
                .register(format!("51/0/{i}/0/0"), batches.clone())
                .await
                .unwrap();
        }

        for (i, want) in expected.iter().enumerate() {
            let got = store
                .get(&format!("51/0/{i}/0/0"))
                .await
                .unwrap()
                .unwrap_or_else(|| panic!("bucket {i} vanished after spilling"));
            assert_eq!(got.len(), want.len(), "bucket {i}: batch count");
            for (a, b) in got.iter().zip(want.iter()) {
                assert_eq!(
                    a, b,
                    "bucket {i}: a spilled batch changed on the round trip"
                );
            }
        }
    }

    /// A store whose spill directory cannot be written: the disk-full / read-only case.
    fn store_with_dead_disk(cap: usize) -> PartitionStore {
        let store = PartitionStore::with_cap(cap);
        let dead = std::env::temp_dir().join(format!("bc_no_such_dir_{}/x/y", std::process::id()));
        store.spill_dir.set(Some(dead)).unwrap();
        store
    }

    /// BT-005: when the cap is set and spilling fails, a publish that would take the store
    /// over its cap is refused -- the store does not silently keep it resident.
    #[tokio::test]
    async fn a_failed_spill_refuses_the_publish_instead_of_overrunning_the_cap() {
        let one = batch_bytes(&[wide_batch(0, 4096)]);
        let store = store_with_dead_disk(one * 2);
        for i in 0..2 {
            store
                .register(format!("60/0/{i}/0/0"), vec![wide_batch(i, 4096)])
                .await
                .expect("under the cap, no disk is needed");
        }
        let err = store
            .register("60/0/2/0/0".into(), vec![wide_batch(9, 4096)])
            .await
            .expect_err("over the cap with a dead disk must refuse");
        assert_eq!(err.ticket, "60/0/2/0/0");
        assert!(err.to_string().contains("spill"), "{err}");
        assert!(
            store.retained_bytes() <= one * 2,
            "a refused publish must hand its reservation back: {} > {}",
            store.retained_bytes(),
            one * 2
        );
        assert!(
            store.get("60/0/2/0/0").await.unwrap().is_none(),
            "refused means unpublished"
        );
        assert_eq!(store.len().await, 2);
        // An empty bucket costs nothing and must always publish, dead disk or not.
        store.register("60/0/3/0/0".into(), vec![]).await.unwrap();
    }

    /// BT-022: the bytes are reserved before the bucket is visible, so concurrent publishers
    /// cannot each see a total that excludes the others and leave the store over its cap.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_publishes_end_within_the_cap() {
        let one = batch_bytes(&[wide_batch(0, 4096)]);
        let store = Arc::new(PartitionStore::with_cap(one * 3));
        let mut tasks = Vec::new();
        for i in 0..32 {
            let store = store.clone();
            tasks.push(tokio::spawn(async move {
                store
                    .register(format!("61/0/{i}/0/0"), vec![wide_batch(i, 4096)])
                    .await
                    .unwrap();
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(store.len().await, 32);
        assert!(
            store.retained_bytes() <= one * 3,
            "{} resident against a cap of {}",
            store.retained_bytes(),
            one * 3
        );
        for i in 0..32 {
            let got = store.get(&format!("61/0/{i}/0/0")).await.unwrap().unwrap();
            assert_eq!(
                got[0],
                wide_batch(i, 4096),
                "bucket {i} changed through the spill"
            );
        }
    }

    /// BT-044: the spill write runs without the map lock, so the bucket can be removed or
    /// replaced mid-write. The commit must notice, keep the new state, and delete the file.
    #[tokio::test]
    async fn a_bucket_replaced_during_its_spill_write_keeps_the_new_bytes() {
        let store = PartitionStore::with_cap(0);
        store
            .register("62/0/0/0/0".into(), vec![wide_batch(1, 512)])
            .await
            .unwrap();
        let (ticket, batches) =
            PartitionStore::claim_victim(&mut *store.partitions.write().await).unwrap();
        // Claimed buckets are skipped by other spillers: no double write of one bucket.
        assert!(PartitionStore::claim_victim(&mut *store.partitions.write().await).is_none());
        let path = store.write_spill(&ticket, &batches).unwrap();

        // Meanwhile a retried map task republishes the ticket.
        store
            .register("62/0/0/0/0".into(), vec![wide_batch(2, 512)])
            .await
            .unwrap();
        let before = store.retained_bytes();
        let freed = store.commit_spill(
            &mut *store.partitions.write().await,
            &ticket,
            &batches,
            path.clone(),
        );
        assert_eq!(freed, 0, "a superseded write frees nothing");
        assert!(!path.exists(), "the orphaned spill file must be deleted");
        assert_eq!(store.retained_bytes(), before);
        let got = store.get("62/0/0/0/0").await.unwrap().unwrap();
        assert_eq!(got[0], wide_batch(2, 512), "the republished bytes must win");
    }

    /// The cooperative path spills outside the lock too, and still frees what it is asked for.
    #[tokio::test]
    async fn cooperative_spill_frees_without_holding_the_map() {
        let store = PartitionStore::with_cap(0);
        for i in 0..4 {
            store
                .register(format!("63/0/{i}/0/0"), vec![wide_batch(i, 4096)])
                .await
                .unwrap();
        }
        let held = store.retained_bytes();
        let freed = store.try_spill_at_least(held / 2);
        assert!(freed >= held / 2, "freed {freed} of the {} asked", held / 2);
        assert_eq!(store.retained_bytes(), held - freed);
        for i in 0..4 {
            let got = store.get(&format!("63/0/{i}/0/0")).await.unwrap().unwrap();
            assert_eq!(got[0], wide_batch(i, 4096));
        }
    }

    /// The path a spilled bucket was written to.
    async fn spill_path(store: &PartitionStore, ticket: &str) -> PathBuf {
        match &store
            .partitions
            .read()
            .await
            .get(ticket)
            .expect("registered")
            .body
        {
            Body::Spilled(path) => path.clone(),
            Body::Memory(_) => panic!("{ticket} was expected to be spilled"),
        }
    }

    /// A spilled bucket whose file is truncated, deleted, or unreadable is *lost*, and the
    /// store must say so rather than report the ticket as unknown -- which every consumer
    /// used to read as an empty bucket, turning a disk fault into a short result.
    #[tokio::test]
    async fn an_unreadable_spill_file_is_an_error_not_an_empty_bucket() {
        let store = PartitionStore::with_cap(1); // everything spills
        for i in 0..3 {
            store
                .register(format!("54/0/{i}/0/0"), vec![wide_batch(i, 512)])
                .await
                .unwrap();
        }

        // Truncated: half the file survives a crash mid-copy or a full disk.
        let truncated = spill_path(&store, "54/0/0/0/0").await;
        let bytes = std::fs::read(&truncated).unwrap();
        std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
        // Deleted: a scratch sweeper or an operator removed it.
        std::fs::remove_file(spill_path(&store, "54/0/1/0/0").await).unwrap();

        for ticket in ["54/0/0/0/0", "54/0/1/0/0"] {
            let err = store
                .get(ticket)
                .await
                .expect_err("an unreadable spill must not read as absent or empty");
            assert_eq!(err.ticket, ticket, "the error must name the lost bucket");
            assert!(
                store.get_with_gauge(ticket).await.is_err(),
                "do_exchange's lookup must see the same error"
            );
        }
        // The untouched bucket still reads back, and a never-published ticket is still absent.
        assert_eq!(store.get("54/0/2/0/0").await.unwrap().unwrap().len(), 1);
        assert!(store.get("54/0/9/0/0").await.unwrap().is_none());
    }

    /// Permission loss on a spill file (a remounted read-only-for-others scratch, a changed
    /// owner) is the third way the only copy becomes unreadable.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_permission_denied_spill_file_is_an_error() {
        use std::os::unix::fs::PermissionsExt;
        let store = PartitionStore::with_cap(1);
        store
            .register("55/0/0/0/0".into(), vec![wide_batch(3, 512)])
            .await
            .unwrap();
        let path = spill_path(&store, "55/0/0/0/0").await;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        // root ignores file modes; the assertion only means something when the open fails.
        if std::fs::File::open(&path).is_err() {
            let err = store
                .get("55/0/0/0/0")
                .await
                .expect_err("denied read must error");
            assert_eq!(err.cause.kind(), std::io::ErrorKind::PermissionDenied);
        }
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }

    /// The gauge and the fetch path must work for a spilled bucket exactly as for a
    /// resident one — `do_exchange` reads through `get_with_gauge`.
    #[tokio::test]
    async fn a_spilled_bucket_still_serves_its_gauge() {
        let store = PartitionStore::with_cap(1);
        store
            .register("52/0/0/0/0".into(), vec![wide_batch(1, 512)])
            .await
            .unwrap();

        let (batches, gauge) = store
            .get_with_gauge("52/0/0/0/0")
            .await
            .unwrap()
            .expect("spilled bucket");
        assert_eq!(batches.len(), 1);
        gauge.on_send();
        assert_eq!(gauge.max(), 1);
    }

    /// A configured spill root is where buckets spill, and where the orphan sweep runs.
    ///
    /// The store used to spill under `temp_dir()` whatever the operator configured, so on a
    /// container whose `/tmp` is a small tmpfs the spill took the RAM it was meant to free.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn buckets_spill_under_the_configured_root() {
        let scratch =
            std::env::temp_dir().join(format!("bc_spill_root_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&scratch);
        let orphan = scratch
            .join("batcher_shuffle_spill")
            .join("4294967294_0xdead");
        std::fs::create_dir_all(&orphan).unwrap();

        let store = PartitionStore::with_cap_in(1, scratch.clone());
        store
            .register("56/0/0/0/0".into(), vec![wide_batch(1, 512)])
            .await
            .unwrap();
        store
            .register("56/1/0/0/0".into(), vec![wide_batch(2, 512)])
            .await
            .unwrap();

        let dir = store.spill_dir().expect("a spill dir").clone();
        let spilled = std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
        let read_back = store.get("56/0/0/0/0").await.unwrap().expect("bucket");
        let orphan_swept = !orphan.exists();
        store.clear().await;
        let _ = std::fs::remove_dir_all(&scratch);

        assert!(
            dir.starts_with(&scratch),
            "spilled to {dir:?}, not under the configured root"
        );
        assert!(spilled >= 1, "nothing spilled under the configured root");
        assert_eq!(read_back.len(), 1);
        assert!(
            orphan_swept,
            "the orphan sweep did not run under the configured root"
        );
    }

    /// The process tunable is what a default store captures, and clearing it restores the
    /// temp-dir default. The root is a writable path rather than a bogus one because the
    /// tunable is process-wide: a store another test builds in the window captures it too,
    /// and must still be able to spill there.
    #[test]
    fn a_default_store_captures_the_process_spill_root() {
        let root = std::env::temp_dir().join("bc_spill_root_capture");
        crate::set_shuffle_spill_root(Some(root.clone()));
        let captured = PartitionStore::with_cap(0).spill_root;
        crate::set_shuffle_spill_root(None);
        assert_eq!(captured, root);
        assert_eq!(crate::shuffle_spill_root(), std::env::temp_dir());
    }

    /// A spilled bucket's file must go when the bucket does, on every eviction path —
    /// otherwise the memory bound is bought with an unbounded disk leak.
    #[tokio::test]
    async fn eviction_removes_a_spilled_buckets_file() {
        let store = PartitionStore::with_cap(1);
        store
            .register("53/0/0/0/0".into(), vec![wide_batch(1, 512)])
            .await
            .unwrap();
        store
            .register("53/1/0/0/0".into(), vec![wide_batch(2, 512)])
            .await
            .unwrap();
        store
            .register("54/0/0/0/0".into(), vec![wide_batch(3, 512)])
            .await
            .unwrap();

        let dir = store.spill_dir().expect("a spill dir").clone();
        let count = || std::fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0);
        assert_eq!(count(), 3, "each spilled bucket should have a file");

        store.remove("53/0/0/0/0").await;
        assert_eq!(count(), 2, "remove left the spill file behind");

        store.remove_prefix("53/").await;
        assert_eq!(count(), 1, "remove_prefix left spill files behind");

        store.clear().await;
        assert_eq!(count(), 0, "clear left spill files behind");
    }

    /// With no cap configured — the default — nothing spills and the store behaves
    /// exactly as it did.
    #[tokio::test]
    async fn an_unbounded_store_never_spills() {
        let store = PartitionStore::with_cap(0);
        for i in 0..8 {
            store
                .register(format!("55/0/{i}/0/0"), vec![wide_batch(i, 4096)])
                .await
                .unwrap();
        }
        let resident = store.retained_bytes();
        assert_eq!(resident, batch_bytes(&[wide_batch(0, 4096)]) * 8);
        assert!(
            store.spill_dir.get().is_none(),
            "an unbounded store created a spill dir"
        );
    }

    /// The counter going down is not the point — the *memory* has to go down. Publishes
    /// far more than the cap and checks the process's own resident set, so a bug that
    /// merely stopped counting (rather than stopped holding) fails here.
    #[cfg(target_os = "linux")]
    /// The cooperative-reservation entry point frees what it is asked for.
    #[tokio::test]
    async fn an_on_demand_spill_frees_at_least_what_was_asked() {
        let store = PartitionStore::with_cap(0); // no cap: pressure, not a bound, drives this
        for i in 0..6 {
            store
                .register(format!("70/0/{i}/0/0"), vec![wide_batch(i, 20_000)])
                .await
                .unwrap();
        }
        let held = store.retained_bytes();
        assert!(held > 0);

        let want = held / 2;
        let freed = store.try_spill_at_least(want);

        assert!(freed >= want, "freed {freed}, asked {want}");
        assert_eq!(store.retained_bytes(), held - freed);
    }

    /// It must not spill the whole store to satisfy a small request.
    ///
    /// The pool asks for a deficit, not for everything. Over-spilling turns one tight
    /// reservation into a re-read of every bucket on the node, which is how a memory
    /// mechanism becomes a throughput problem.
    #[tokio::test]
    async fn an_on_demand_spill_stops_once_the_target_is_met() {
        let store = PartitionStore::with_cap(0);
        for i in 0..8 {
            store
                .register(format!("71/0/{i}/0/0"), vec![wide_batch(i, 20_000)])
                .await
                .unwrap();
        }
        let held = store.retained_bytes();
        store.try_spill_at_least(1); // one byte: the smallest possible ask
        assert!(
            store.retained_bytes() > 0,
            "a one-byte request emptied the whole store"
        );
        assert!(store.retained_bytes() < held, "nothing was spilled at all");
    }

    /// Spilled-on-demand buckets must still serve the identical rows.
    ///
    /// This is the property that makes the whole mechanism admissible: a memory strategy
    /// that changed an answer would be a correctness bug wearing a performance costume.
    #[tokio::test]
    async fn an_on_demand_spilled_bucket_reads_back_identically() {
        let store = PartitionStore::with_cap(0);
        let mut expected = Vec::new();
        for i in 0..4 {
            let batch = wide_batch(i, 20_000);
            expected.push(batch.clone());
            store
                .register(format!("72/0/{i}/0/0"), vec![batch])
                .await
                .unwrap();
        }
        store.try_spill_at_least(store.retained_bytes()); // spill everything

        for (i, want) in expected.iter().enumerate() {
            let got = store
                .get(&format!("72/0/{i}/0/0"))
                .await
                .unwrap()
                .expect("bucket");
            assert_eq!(got.len(), 1);
            assert_eq!(&got[0], want, "bucket {i} changed across the spill");
        }
    }

    /// A zero request is a no-op, and an empty store answers zero rather than churning.
    #[tokio::test]
    async fn an_on_demand_spill_declines_when_there_is_nothing_to_do() {
        let store = PartitionStore::with_cap(0);
        assert_eq!(store.try_spill_at_least(1 << 20), 0, "empty store spilled");

        store
            .register("73/0/0/0/0".to_string(), vec![wide_batch(0, 20_000)])
            .await
            .unwrap();
        assert_eq!(store.try_spill_at_least(0), 0, "zero-byte request spilled");
        assert!(store.retained_bytes() > 0);
    }

    /// It yields rather than waits when the map is busy.
    ///
    /// The pool may call this from a tokio worker thread. Blocking there on the store's
    /// async lock would deadlock the runtime serving the very fetches that would drain the
    /// store, so a contended call must return `0` and let the pool move on. `0` is an
    /// explicitly permitted answer under the `Spillable` contract.
    #[tokio::test]
    async fn a_busy_store_declines_instead_of_blocking() {
        let store = PartitionStore::with_cap(0);
        store
            .register("74/0/0/0/0".to_string(), vec![wide_batch(0, 20_000)])
            .await
            .unwrap();
        let held = store.retained_bytes();

        let guard = store.partitions.write().await;
        let freed = store.try_spill_at_least(held);
        drop(guard);

        assert_eq!(freed, 0, "it took the slow path while the map was held");
        assert_eq!(store.retained_bytes(), held);
        // And it recovers once the lock is free — a decline is not a latch.
        assert!(store.try_spill_at_least(held) > 0);
    }

    /// Spilling must return memory to the *process*, not merely to a counter.
    ///
    /// Measured on the resident set, because that is the claim — a store that dropped its
    /// references but whose pages were still mapped would satisfy `retained_bytes` and help
    /// nobody. Measured on this test's own data: ~16.5 MB of growth bounded against ~52.8 MB
    /// unbounded, for 40 buckets of ~1.6 MB.
    ///
    /// **Best of several trials**, and that is not a weakened assertion. `/proc/self/statm`
    /// is process-wide, and `cargo test` runs this binary's tests in parallel threads, so a
    /// sibling test allocating during the measurement lands in this delta. That noise can
    /// only ever *inflate* the bounded figure and make the test fail — it cannot fabricate a
    /// pass — so a run that observes the expected shape once has observed it. Before this,
    /// the test passed alone and failed inside a full-workspace run, which is the worst
    /// failure mode a test can have: it teaches people to ignore it.
    #[tokio::test]
    async fn spilling_actually_returns_memory_to_the_process() {
        fn rss_bytes() -> usize {
            let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
            let pages: usize = statm
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            pages * 4096
        }

        const BUCKETS: i64 = 40;
        const ROWS: usize = 200_000; // ~1.6 MB per bucket, ~64 MB total
        const TRIALS: usize = 3;

        let mut best: Option<(usize, usize)> = None;
        for trial in 0..TRIALS {
            let tag = 60 + trial as i64 * 2;

            let bounded = PartitionStore::with_cap(8 << 20); // 8 MiB
            let base = rss_bytes();
            for i in 0..BUCKETS {
                bounded
                    .register(format!("{tag}/0/{i}/0/0"), vec![wide_batch(i, ROWS)])
                    .await
                    .unwrap();
            }
            let bounded_growth = rss_bytes().saturating_sub(base);
            assert!(
                bounded.retained_bytes() <= 8 << 20,
                "bounded store stayed at {} resident bytes",
                bounded.retained_bytes(),
            );
            bounded.clear().await;

            let unbounded = PartitionStore::with_cap(0);
            let base2 = rss_bytes();
            for i in 0..BUCKETS {
                unbounded
                    .register(format!("{}/0/{i}/0/0", tag + 1), vec![wide_batch(i, ROWS)])
                    .await
                    .unwrap();
            }
            let unbounded_growth = rss_bytes().saturating_sub(base2);
            assert!(
                unbounded.retained_bytes() > 8 << 20,
                "the unbounded control did not exceed the cap, so this proves nothing",
            );
            unbounded.clear().await;

            // A generous ratio (not the measured 3.2x) so the assertion is about the
            // mechanism working, not about an allocator's exact behaviour on one machine.
            if bounded_growth * 2 < unbounded_growth {
                return;
            }
            best = Some(match best {
                Some((b, u)) if b <= bounded_growth => (b, u),
                _ => (bounded_growth, unbounded_growth),
            });
        }

        let (b, u) = best.expect("at least one trial ran");
        panic!(
            "spilling freed no real memory in {TRIALS} trials: best was {b} bytes of bounded \
             growth against {u} unbounded"
        );
    }
}
