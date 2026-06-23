//! Internal partition store: the in-memory registry mapping a ticket string to
//! the batches served under it, plus the per-exchange in-flight gauge used to
//! prove the credit bound.

use std::collections::HashMap;
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
}

impl InflightGauge {
    /// Producer is about to send one more batch: bump in-flight and the max.
    ///
    /// The high-water update uses `AcqRel` (C58): the gauge is read to *prove* the
    /// credit bound was honored, so the max must not be under-reported on a weak
    /// memory model. It is off the per-batch data path, so the stronger ordering is
    /// negligible.
    pub(crate) fn on_send(&self) {
        use std::sync::atomic::Ordering::{AcqRel, Relaxed};
        let now = self.current.fetch_add(1, Relaxed) + 1;
        self.max.fetch_max(now, AcqRel);
    }

    /// Consumer acked one batch (a top-up credit arrived): drop in-flight.
    pub(crate) fn on_ack(&self) {
        self.current
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// High-water mark of simultaneously in-flight batches.
    pub(crate) fn max(&self) -> i64 {
        self.max.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// One registered partition: the batches plus its in-flight gauge.
pub(crate) struct Partition {
    batches: Arc<Vec<RecordBatch>>,
    gauge: Arc<InflightGauge>,
}

/// In-memory registry mapping a ticket string to the batches served under it.
#[derive(Default)]
pub(crate) struct PartitionStore {
    partitions: RwLock<HashMap<String, Partition>>,
}

impl PartitionStore {
    pub(crate) async fn register(&self, ticket: String, batches: Vec<RecordBatch>) {
        self.partitions.write().await.insert(
            ticket,
            Partition {
                batches: Arc::new(batches),
                gauge: Arc::new(InflightGauge::default()),
            },
        );
    }

    pub(crate) async fn get(&self, ticket: &str) -> Option<Arc<Vec<RecordBatch>>> {
        self.partitions
            .read()
            .await
            .get(ticket)
            .map(|p| p.batches.clone())
    }

    /// Fetch both the batches and the in-flight gauge for an exchange.
    pub(crate) async fn get_with_gauge(
        &self,
        ticket: &str,
    ) -> Option<(Arc<Vec<RecordBatch>>, Arc<InflightGauge>)> {
        self.partitions
            .read()
            .await
            .get(ticket)
            .map(|p| (p.batches.clone(), p.gauge.clone()))
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
        self.partitions.write().await.remove(ticket);
    }

    /// Drop every partition whose ticket begins with `prefix` (e.g. `"{plan_id}/"`
    /// to evict a whole finished plan, or `"{plan_id}/{stage}/"` one stage).
    pub(crate) async fn remove_prefix(&self, prefix: &str) {
        self.partitions
            .write()
            .await
            .retain(|ticket, _| !ticket.starts_with(prefix));
    }

    /// Drop every published partition. Called at plan teardown to return the
    /// worker's shuffle memory to the OS without tearing down the actor.
    pub(crate) async fn clear(&self) {
        self.partitions.write().await.clear();
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
}
