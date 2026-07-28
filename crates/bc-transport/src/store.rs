//! Internal partition store: the in-memory registry mapping a ticket string to
//! the batches served under it, plus the per-exchange in-flight gauge used to
//! prove the credit bound.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
        let _ = self
            .current
            .fetch_update(Relaxed, Relaxed, |c| Some(c.saturating_sub(1).max(0)));
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
}

impl Partition {
    /// The batches, reading them back from disk if this partition was spilled.
    ///
    /// A spilled read deliberately does **not** re-populate the heap copy. The store spilled
    /// it because memory was short; silently restoring it on the first fetch would undo the
    /// bound exactly when it is being relied on, and a bucket is typically fetched once.
    fn batches(&self) -> Option<Arc<Vec<RecordBatch>>> {
        match &self.body {
            Body::Memory(b) => Some(b.clone()),
            Body::Spilled(path) => crate::shared::read_ipc_file(path)
                .ok()
                .flatten()
                .map(Arc::new),
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
    /// Scratch directory for spilled buckets, created on first spill.
    spill_dir: std::sync::OnceLock<Option<PathBuf>>,
}

impl Default for PartitionStore {
    fn default() -> Self {
        Self::with_cap(crate::shuffle_store_cap())
    }
}

impl PartitionStore {
    /// A store bounded at `cap` bytes of resident buckets (`0` = unbounded).
    pub(crate) fn with_cap(cap: usize) -> Self {
        Self {
            partitions: RwLock::new(HashMap::new()),
            retained: AtomicUsize::new(0),
            cap,
            spill_dir: std::sync::OnceLock::new(),
        }
    }

    pub(crate) async fn register(&self, ticket: String, batches: Vec<RecordBatch>) {
        let nbytes = batch_bytes(&batches);
        let previous = self.partitions.write().await.insert(
            ticket,
            Partition {
                body: Body::Memory(Arc::new(batches)),
                gauge: Arc::new(InflightGauge::default()),
                nbytes,
            },
        );
        // Re-registering a ticket (a recompute republishing under a bumped epoch, or a
        // retried map task) replaces the entry. Charging the new bytes without crediting
        // back the old ones makes the total drift up forever, and a monotonically rising
        // "retained bytes" that never falls is worse than no number at all: it reads as a
        // leak in the one place someone would look to find one.
        self.retained.fetch_add(nbytes, Ordering::Relaxed);
        if let Some(old) = previous {
            if old.in_memory() {
                self.retained.fetch_sub(old.nbytes, Ordering::Relaxed);
            }
            old.discard_spill_file();
        }
        self.enforce_cap().await;
    }

    /// Spill published buckets to disk until the store is back under its byte cap.
    ///
    /// **The gap this closes.** Everything else Carbonite bounds is *reserved* memory — an
    /// operator asks the pool before it allocates, and spills when refused. A published
    /// shuffle bucket is never asked for: a mapper hands it to this store and it stays
    /// resident until a reducer fetches it. With `workers` mappers each producing `workers`
    /// buckets, a node holds its whole share of the shuffle in anonymous memory that no
    /// reservation covers and the kernel cannot reclaim. That is the classic shuffle OOM,
    /// and `PressureMonitor` can only see it indirectly, as unexplained RSS.
    ///
    /// Largest-first, because the point is to get back under the cap in the fewest reads
    /// later: one big bucket costs one re-read, many small ones cost many. Spilling is
    /// result-preserving — the same batches come back through the Arrow IPC round-trip —
    /// so this trades a re-read for a memory bound and can never change an answer.
    ///
    /// Off unless a cap is configured (`set_shuffle_store_cap`), so the default path is
    /// byte-for-byte what it was.
    async fn enforce_cap(&self) {
        let cap = self.cap;
        if cap == 0 || self.retained.load(Ordering::Relaxed) <= cap {
            return;
        }
        let Some(dir) = self.spill_dir() else { return };
        let mut guard = self.partitions.write().await;
        self.spill_down_to(&mut guard, dir, cap);
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
    /// store. `try_write` instead: if the map is busy, this returns `0`, which the
    /// `Spillable` contract explicitly allows and the pool treats as "this consumer cannot
    /// help right now". A missed opportunity is recoverable; a hung shuffle server is not.
    ///
    /// Independent of `cap`: a store with no configured cap still answers, because the
    /// caller here is real memory pressure rather than a configured bound.
    pub(crate) fn try_spill_at_least(&self, target: usize) -> usize {
        if target == 0 {
            return 0;
        }
        let held = self.retained.load(Ordering::Relaxed);
        if held == 0 {
            return 0;
        }
        let Some(dir) = self.spill_dir() else {
            return 0;
        };
        let Ok(mut guard) = self.partitions.try_write() else {
            return 0;
        };
        let floor = held.saturating_sub(target);
        self.spill_down_to(&mut guard, dir, floor)
    }

    /// Spill resident buckets, largest first, until `retained <= floor`. Returns bytes freed.
    ///
    /// Largest-first, because the point is to get back under the bound in the fewest
    /// re-reads later: one big bucket costs one, many small ones cost many. Spilling is
    /// result-preserving — the same batches return through the Arrow IPC round trip — so
    /// this trades a re-read for a memory bound and can never change an answer.
    ///
    /// Caller holds the map's write guard, which is what lets the async cap enforcement and
    /// the synchronous cooperative path share one implementation of the choice of victim.
    fn spill_down_to(
        &self,
        guard: &mut HashMap<String, Partition>,
        dir: &Path,
        floor: usize,
    ) -> usize {
        let mut freed = 0usize;
        loop {
            if self.retained.load(Ordering::Relaxed) <= floor {
                break;
            }
            // The largest still-resident bucket.
            let victim = guard
                .iter()
                .filter(|(_, p)| p.in_memory())
                .max_by_key(|(_, p)| p.nbytes)
                .map(|(t, _)| t.clone());
            let Some(ticket) = victim else { break };
            let Some(p) = guard.get_mut(&ticket) else {
                break;
            };
            let Body::Memory(batches) = &p.body else {
                break;
            };
            let path = dir.join(format!("{}.arrow", sanitize_ticket(&ticket)));
            if crate::shared::write_ipc_file(&path, batches).is_err() {
                // Nowhere to put it: stop rather than spin. Staying over the bound is worse
                // than it was, but failing the publish would be worse still — the bucket
                // is already produced and a reducer is waiting for it.
                break;
            }
            p.body = Body::Spilled(path);
            self.retained.fetch_sub(p.nbytes, Ordering::Relaxed);
            freed += p.nbytes;
        }
        freed
    }

    /// This store's spill directory, created once on first use.
    fn spill_dir(&self) -> Option<&PathBuf> {
        self.spill_dir
            .get_or_init(|| {
                let dir = std::env::temp_dir().join(format!(
                    "batcher_shuffle_spill/{}_{:p}",
                    std::process::id(),
                    self
                ));
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

    pub(crate) async fn get(&self, ticket: &str) -> Option<Arc<Vec<RecordBatch>>> {
        self.partitions.read().await.get(ticket)?.batches()
    }

    /// Fetch both the batches and the in-flight gauge for an exchange.
    pub(crate) async fn get_with_gauge(
        &self,
        ticket: &str,
    ) -> Option<(Arc<Vec<RecordBatch>>, Arc<InflightGauge>)> {
        let guard = self.partitions.read().await;
        let p = guard.get(ticket)?;
        Some((p.batches()?, p.gauge.clone()))
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
            .await;
        let got = store.get("7/0/0/0").await.expect("registered");
        assert_eq!(got.len(), 2);
        assert!(store.get("7/0/0/9").await.is_none()); // unregistered ticket
    }

    #[tokio::test]
    async fn gauge_tracks_inflight_high_water() {
        let store = PartitionStore::default();
        store.register("p/0/0/0".into(), vec![one_batch(1)]).await;
        let (_b, gauge) = store.get_with_gauge("p/0/0/0").await.unwrap();
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
        store.register("9/0/0/0".into(), vec![one_batch(1)]).await;
        store.register("9/0/0/1".into(), vec![one_batch(2)]).await;
        assert_eq!(store.len().await, 2);
        store.remove("9/0/0/0").await;
        assert_eq!(store.len().await, 1);
        assert!(store.get("9/0/0/0").await.is_none());
        store.clear().await;
        assert_eq!(store.len().await, 0);
    }

    #[tokio::test]
    async fn retained_bytes_tracks_registration_and_eviction() {
        let store = PartitionStore::default();
        assert_eq!(store.retained_bytes(), 0);

        store.register("b/0/0/0".into(), vec![one_batch(1)]).await;
        let one = store.retained_bytes();
        assert!(
            one > 0,
            "a registered partition holds memory nothing accounts for"
        );

        store.register("b/0/0/1".into(), vec![one_batch(2)]).await;
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
        store.register("r/0/0/0".into(), vec![one_batch(1)]).await;
        let one = store.retained_bytes();
        store
            .register("r/0/0/0".into(), vec![one_batch(2), one_batch(3)])
            .await;
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
        store.register("p9/0/0/0".into(), vec![one_batch(1)]).await;
        store.register("p9/1/0/0".into(), vec![one_batch(2)]).await;
        store.register("p8/0/0/0".into(), vec![one_batch(3)]).await;
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
        store.register("9/0/0/0".into(), vec![one_batch(1)]).await; // plan 9, stage 0
        store.register("9/1/0/0".into(), vec![one_batch(2)]).await; // plan 9, stage 1
        store.register("8/0/0/0".into(), vec![one_batch(3)]).await; // plan 8
        store.remove_prefix("9/0/").await; // evict only plan 9, stage 0
        assert!(store.get("9/0/0/0").await.is_none());
        assert!(store.get("9/1/0/0").await.is_some());
        assert!(store.get("8/0/0/0").await.is_some());
        // A whole-plan prefix evicts every stage of that plan.
        store.remove_prefix("9/").await;
        assert!(store.get("9/1/0/0").await.is_none());
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
                .await;
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
                .await;
        }

        for (i, want) in expected.iter().enumerate() {
            let got = store
                .get(&format!("51/0/{i}/0/0"))
                .await
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

    /// The gauge and the fetch path must work for a spilled bucket exactly as for a
    /// resident one — `do_exchange` reads through `get_with_gauge`.
    #[tokio::test]
    async fn a_spilled_bucket_still_serves_its_gauge() {
        let store = PartitionStore::with_cap(1);
        store
            .register("52/0/0/0/0".into(), vec![wide_batch(1, 512)])
            .await;

        let (batches, gauge) = store
            .get_with_gauge("52/0/0/0/0")
            .await
            .expect("spilled bucket");
        assert_eq!(batches.len(), 1);
        gauge.on_send();
        assert_eq!(gauge.max(), 1);
    }

    /// A spilled bucket's file must go when the bucket does, on every eviction path —
    /// otherwise the memory bound is bought with an unbounded disk leak.
    #[tokio::test]
    async fn eviction_removes_a_spilled_buckets_file() {
        let store = PartitionStore::with_cap(1);
        store
            .register("53/0/0/0/0".into(), vec![wide_batch(1, 512)])
            .await;
        store
            .register("53/1/0/0/0".into(), vec![wide_batch(2, 512)])
            .await;
        store
            .register("54/0/0/0/0".into(), vec![wide_batch(3, 512)])
            .await;

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
                .await;
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
                .await;
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
                .await;
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
            store.register(format!("72/0/{i}/0/0"), vec![batch]).await;
        }
        store.try_spill_at_least(store.retained_bytes()); // spill everything

        for (i, want) in expected.iter().enumerate() {
            let got = store.get(&format!("72/0/{i}/0/0")).await.expect("bucket");
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
            .await;
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
            .await;
        let held = store.retained_bytes();

        let guard = store.partitions.write().await;
        let freed = store.try_spill_at_least(held);
        drop(guard);

        assert_eq!(freed, 0, "it took the slow path while the map was held");
        assert_eq!(store.retained_bytes(), held);
        // And it recovers once the lock is free — a decline is not a latch.
        assert!(store.try_spill_at_least(held) > 0);
    }

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

        let bounded = PartitionStore::with_cap(8 << 20); // 8 MiB
        let base = rss_bytes();
        for i in 0..BUCKETS {
            bounded
                .register(format!("60/0/{i}/0/0"), vec![wide_batch(i, ROWS)])
                .await;
        }
        let bounded_growth = rss_bytes().saturating_sub(base);
        bounded.clear().await;

        let unbounded = PartitionStore::with_cap(0);
        let base2 = rss_bytes();
        for i in 0..BUCKETS {
            unbounded
                .register(format!("61/0/{i}/0/0"), vec![wide_batch(i, ROWS)])
                .await;
        }
        let unbounded_growth = rss_bytes().saturating_sub(base2);

        assert!(
            bounded.retained_bytes() <= 8 << 20,
            "bounded store stayed at {} resident bytes",
            bounded.retained_bytes(),
        );
        assert!(
            unbounded.retained_bytes() > 8 << 20,
            "the unbounded control did not exceed the cap, so this proves nothing",
        );
        // The counter is not the claim; the resident set is. Measured on this test's own
        // data: ~16.5 MB of growth bounded against ~52.8 MB unbounded, for 40 buckets of
        // ~1.6 MB. A generous ratio here (not the measured 3.2x) so the assertion is about
        // the mechanism working, not about an allocator's exact behaviour on one machine.
        assert!(
            bounded_growth * 2 < unbounded_growth,
            "spilling freed no real memory: bounded grew {bounded_growth} bytes, \
             unbounded {unbounded_growth}",
        );
        unbounded.clear().await;
    }
}
