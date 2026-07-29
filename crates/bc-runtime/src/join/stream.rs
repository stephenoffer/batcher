//! Streaming broadcast probe — build the hash table once, probe one morsel at a time.
//!
//! The broadcast join replicates a small build side and probes a large one. Until now the
//! executor had to concatenate the *whole* probe relation into a single `RecordBatch`
//! first, because [`super::broadcast_hash_join_indices`] slices it by absolute row range.
//! On a 60M-row `lineitem` that copy is gigabytes of pure overhead, it is the largest
//! single allocation in the query, and it is exactly what the engine's own performance rule
//! forbids: *"don't introduce whole-relation materialization where a streaming / morsel path
//! is possible."* It also puts every `Utf8` column at risk of Arrow's 2 GiB 32-bit-offset
//! ceiling.
//!
//! It is possible. [`JoinTable::build`] reads only the build side, so the table can be built
//! once and then probed with a *different* left side per call — one morsel at a time, across
//! cores, with each morsel's output gathered from that morsel alone. The emitted left indices
//! are morsel-local, which is precisely what the caller's `take`-based gather wants.
//!
//! **Identical to the materialized path.** Morsels are contiguous, in-order row ranges of the
//! probe relation, so probing them in order emits the same rows in the same order as slicing
//! the concatenated batch by row range. The build table, hash state, and probe loop are the
//! same code.
//!
//! Restricted to what is provably safe to do per morsel:
//!
//! * **Left-driven join types only** (`Inner`/`Left`/`Semi`/`Anti`). Each probe row lands in
//!   exactly one morsel and no build-side-unmatched rows are emitted, so morsels are
//!   independent. `Right`/`Full` must reconcile unmatched build rows across every morsel and
//!   keep the materialized path.
//! * **Integer keys** (one or two `Int64` columns — the analytical join shape after the FFI
//!   boundary normalizes narrow integers). A row-encoded key would need its `RowConverter`
//!   shared across morsels; until that is threaded through, those joins fall back.
//!
//! [`BroadcastProbe::new`] returns `None` for anything else, and the caller keeps its old
//! path. Nothing silently changes shape.

use arrow::array::ArrayRef;

use super::{
    null_mask, use_probe_bloom_with, I64Keys, I64x2Keys, JoinIndices, JoinTable, JoinType,
};

/// The build-side key shape a [`BroadcastProbe`] was built for. Each morsel's probe keys
/// must present the same shape, which they do — both sides come from the same plan node.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KeyShape {
    /// One `Int64` column.
    I64,
    /// Two `Int64` columns.
    I64x2,
}

impl KeyShape {
    /// The shape of `keys`, or `None` if it is not an integer fast-path shape.
    fn of(keys: &[ArrayRef]) -> Option<Self> {
        use arrow::datatypes::DataType;
        if keys.iter().any(|k| k.data_type() != &DataType::Int64) {
            return None;
        }
        match keys.len() {
            1 => Some(Self::I64),
            2 => Some(Self::I64x2),
            _ => None,
        }
    }
}

/// Whether a join type emits each probe row independently of the others, so a morsel can be
/// probed on its own. `Right`/`Full` must emit build rows nothing matched, which is only
/// knowable after every morsel has been seen.
fn is_probe_driven(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Inner | JoinType::Left | JoinType::Semi | JoinType::Anti
    )
}

/// Whether [`BroadcastProbe`] can serve this join — answerable from the build side's *schema
/// and row count alone*, before anything is concatenated.
///
/// [`BroadcastProbe::new`] re-checks all of this and returns `None` if it disagrees; this
/// exists so a caller weighing the streaming path against a shuffle can find out **without
/// first paying to materialize the build side**. The conditions are the module's: a
/// probe-driven join type, one or two `Int64` key columns, and a build that stays under the
/// cache-radix cliff.
pub fn streaming_supported(
    join_type: JoinType,
    key_types: &[&arrow::datatypes::DataType],
    build_rows: usize,
) -> bool {
    use arrow::datatypes::DataType;
    is_probe_driven(join_type)
        && build_rows <= super::RADIX_MIN_BUILD_ROWS_BROADCAST
        && matches!(key_types.len(), 1 | 2)
        && key_types.iter().all(|t| *t == &DataType::Int64)
}

/// A hash table over a broadcast build side, ready to be probed morsel by morsel.
pub struct BroadcastProbe {
    table: JoinTable,
    build_keys: Vec<ArrayRef>,
    shape: KeyShape,
    join_type: JoinType,
}

impl BroadcastProbe {
    /// Build the table over `build_keys`, or `None` when this join cannot be streamed
    /// (see the module docs) and the caller should keep its materialized path.
    ///
    /// `probe_rows` is the probe relation's *total* row count across all its morsels; it
    /// decides whether the probe-side bloom pre-filter pays for itself, exactly as the
    /// materialized path decides it.
    pub fn new(
        build_keys: &[ArrayRef],
        join_type: JoinType,
        probe_rows: usize,
        bloom_fp_rate: f64,
        bloom_min_build_rows: usize,
    ) -> Option<Self> {
        let build_rows = build_keys.first().map_or(0, |a| a.len());
        // A build past the cache-radix floor belongs on the partitioned path, whose
        // per-partition table stays cache-resident. Streaming would probe one flat table
        // that no longer fits L3 and pay a miss per probe row — the very cliff
        // `RADIX_MIN_BUILD_ROWS_BROADCAST` exists to avoid. Those joins keep the
        // materialized radix path, whose probe-side concatenation the partition pass needs
        // anyway (it addresses probe rows by absolute index).
        if build_rows > super::RADIX_MIN_BUILD_ROWS_BROADCAST {
            return None;
        }
        Self::over_any_build(
            build_keys,
            join_type,
            probe_rows,
            bloom_fp_rate,
            bloom_min_build_rows,
        )
    }

    /// [`BroadcastProbe::new`] **without** the cache ceiling on the build side.
    ///
    /// The ceiling in `new` compares a flat probe against the *partitioned* join and is right
    /// for that comparison: past L3, one shared table pays a cache miss per probe row where the
    /// radix path keeps each partition resident.
    ///
    /// A caller that is **fusing an aggregate onto the probe** is not making that comparison.
    /// Its alternative is not a partitioned probe, it is materializing the join's entire output
    /// and taking a second pass over it to group — and that output is the largest thing in the
    /// query. On the sf10 `lineitem ⋈ orders` group-by, declining here produced a 2.0 GB
    /// intermediate and a separate 60M-row grouping pass; the cache misses a flat probe pays are
    /// bounded by the build side, which is smaller than the output it avoids writing.
    ///
    /// So this is not "ignore the ceiling", it is "the ceiling answers a question this caller is
    /// not asking". Everything else `new` checks — a probe-driven join type, a supported key
    /// shape — still applies, and the table, the probe loop and the emitted rows are identical.
    pub fn over_any_build(
        build_keys: &[ArrayRef],
        join_type: JoinType,
        probe_rows: usize,
        bloom_fp_rate: f64,
        bloom_min_build_rows: usize,
    ) -> Option<Self> {
        if !is_probe_driven(join_type) {
            return None;
        }
        let shape = KeyShape::of(build_keys)?;
        let build_rows = build_keys.first().map_or(0, |a| a.len());
        let build_null = null_mask(build_keys, build_rows);
        let use_bloom = use_probe_bloom_with(build_rows, probe_rows, bloom_min_build_rows);
        // The build loop reads only the right side of a `JoinKeys`, so the left may be the
        // build itself: no probe rows exist yet.
        let table = match shape {
            KeyShape::I64 => {
                let keys = I64Keys::try_new(build_keys, build_keys)?;
                JoinTable::build(&keys, build_rows, &build_null, use_bloom, bloom_fp_rate)
            }
            KeyShape::I64x2 => {
                let keys = I64x2Keys::try_new(build_keys, build_keys)?;
                JoinTable::build(&keys, build_rows, &build_null, use_bloom, bloom_fp_rate)
            }
        };
        Some(Self {
            table,
            build_keys: build_keys.to_vec(),
            shape,
            join_type,
        })
    }

    /// Whether `probe_keys` present the same key shape the table was built for.
    ///
    /// Every morsel of a relation shares one schema, so the caller checks this once against
    /// the first morsel and can then treat [`Self::probe`] as infallible for the rest.
    pub fn accepts(&self, probe_keys: &[ArrayRef]) -> bool {
        KeyShape::of(probe_keys) == Some(self.shape)
    }

    /// Probe one morsel. `left` indices are **local to `probe_keys`**; `right` indices are
    /// absolute into the build side. `None` when the morsel's key shape disagrees with the
    /// build's — check once with [`Self::accepts`].
    pub fn probe(&self, probe_keys: &[ArrayRef]) -> Option<JoinIndices> {
        if KeyShape::of(probe_keys)? != self.shape {
            return None;
        }
        let rows = probe_keys.first().map_or(0, |a| a.len());
        // Build the null mask only when a probe key actually carries nulls. This runs once per
        // morsel — hundreds of times per join — and a foreign-key probe (`l_orderkey`, never
        // null) hits the `None` arm, skipping a 16 KB `vec![false; 16384]` allocate-and-zero
        // that `probe_range` would only ever read as `false`.
        let probe_null =
            (probe_keys.iter().any(|k| k.null_count() != 0)).then(|| null_mask(probe_keys, rows));
        let mut left = super::IndexBuf::with_capacity(rows);
        let mut right = super::IndexBuf::with_capacity(rows);
        match self.shape {
            KeyShape::I64 => {
                let keys = I64Keys::try_new(probe_keys, &self.build_keys)?;
                self.table.probe_range(
                    &keys,
                    0..rows,
                    probe_null.as_deref(),
                    self.join_type,
                    &mut left,
                    &mut right,
                    None,
                );
            }
            KeyShape::I64x2 => {
                let keys = I64x2Keys::try_new(probe_keys, &self.build_keys)?;
                self.table.probe_range(
                    &keys,
                    0..rows,
                    probe_null.as_deref(),
                    self.join_type,
                    &mut left,
                    &mut right,
                    None,
                );
            }
        }
        Some(JoinIndices {
            left: left.finish(),
            right: right.finish(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use std::sync::Arc;

    fn i64s(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    /// Probing the morsels of a relation, in order, reproduces exactly what probing the
    /// whole concatenated relation does — same pairs, same order. This is the invariant the
    /// streaming path rests on.
    #[test]
    fn morsel_by_morsel_matches_the_whole_relation() {
        let build = vec![i64s(&[1, 2, 3])];
        let whole = vec![i64s(&[3, 1, 9, 2, 1])];
        let morsels = [vec![i64s(&[3, 1])], vec![i64s(&[9, 2, 1])]];

        let probe = BroadcastProbe::new(&build, JoinType::Inner, 5, 0.01, 1 << 16).unwrap();
        let all = probe.probe(&whole).unwrap();

        let mut left: Vec<Option<u32>> = Vec::new();
        let mut right: Vec<Option<u32>> = Vec::new();
        let mut base = 0u32;
        for morsel in &morsels {
            let part = probe.probe(morsel).unwrap();
            // A morsel's left indices are local; shift them to compare against the whole.
            left.extend(part.left.iter().map(|l| l.map(|v| v + base)));
            right.extend(part.right.iter());
            base += morsel[0].len() as u32;
        }
        assert_eq!(left, all.left.iter().collect::<Vec<_>>());
        assert_eq!(right, all.right.iter().collect::<Vec<_>>());
    }

    /// A two-column integer key streams too (TPC-H Q9's `partsupp ⋈ lineitem` shape).
    #[test]
    fn composite_integer_keys_stream() {
        let build = vec![i64s(&[1, 1, 2]), i64s(&[7, 8, 7])];
        let whole = vec![i64s(&[1, 2, 1]), i64s(&[8, 7, 9])];
        let probe = BroadcastProbe::new(&build, JoinType::Inner, 3, 0.01, 1 << 16).unwrap();
        let idx = probe.probe(&whole).unwrap();
        // (1,8) matches build row 1; (2,7) matches build row 2; (1,9) matches nothing.
        assert_eq!(idx.left.iter().collect::<Vec<_>>(), vec![Some(0), Some(1)]);
        assert_eq!(idx.right.iter().collect::<Vec<_>>(), vec![Some(1), Some(2)]);
    }

    /// `Right`/`Full` must see every morsel before emitting unmatched build rows, so they
    /// are refused and the caller keeps the materialized path.
    #[test]
    fn build_driven_join_types_are_refused() {
        let build = vec![i64s(&[1])];
        for jt in [JoinType::Right, JoinType::Full] {
            assert!(BroadcastProbe::new(&build, jt, 1, 0.01, 1 << 16).is_none());
        }
    }

    /// A build too large for one cache-resident table stays on the radix path.
    #[test]
    fn an_oversized_build_is_refused() {
        let big = vec![i64s(&vec![
            1i64;
            super::super::RADIX_MIN_BUILD_ROWS_BROADCAST + 1
        ])];
        assert!(BroadcastProbe::new(&big, JoinType::Inner, 1, 0.01, 1 << 16).is_none());
        let ok = vec![i64s(&vec![1i64; 1024])];
        assert!(BroadcastProbe::new(&ok, JoinType::Inner, 1, 0.01, 1 << 16).is_some());
    }

    /// A non-integer or wide key falls back rather than silently taking a different path.
    #[test]
    fn unsupported_key_shapes_are_refused() {
        use arrow::array::StringArray;
        let strings: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
        assert!(BroadcastProbe::new(&[strings], JoinType::Inner, 1, 0.01, 1 << 16).is_none());
        let three = vec![i64s(&[1]), i64s(&[1]), i64s(&[1])];
        assert!(BroadcastProbe::new(&three, JoinType::Inner, 1, 0.01, 1 << 16).is_none());
    }

    /// An empty morsel probes to nothing without panicking (the tail of a relation whose
    /// row count is an exact multiple of the morsel size).
    #[test]
    fn an_empty_morsel_yields_no_pairs() {
        let build = vec![i64s(&[1, 2])];
        let probe = BroadcastProbe::new(&build, JoinType::Inner, 0, 0.01, 1 << 16).unwrap();
        let idx = probe.probe(&[i64s(&[])]).unwrap();
        assert_eq!(idx.left.len(), 0);
        assert_eq!(idx.right.len(), 0);
    }

    /// `accepts` gates the shape once so the per-morsel probe cannot fail.
    #[test]
    fn accepts_matches_the_built_shape() {
        let build = vec![i64s(&[1])];
        let probe = BroadcastProbe::new(&build, JoinType::Inner, 1, 0.01, 1 << 16).unwrap();
        assert!(probe.accepts(&[i64s(&[1, 2])]));
        assert!(!probe.accepts(&[i64s(&[1]), i64s(&[2])])); // wrong arity
        assert!(probe.probe(&[i64s(&[1]), i64s(&[2])]).is_none());
    }

    /// A `Left` join emits every probe row, matched or not — per morsel, as it must.
    #[test]
    fn left_join_emits_unmatched_probe_rows_per_morsel() {
        let build = vec![i64s(&[2])];
        let probe = BroadcastProbe::new(&build, JoinType::Left, 2, 0.01, 1 << 16).unwrap();
        let idx = probe.probe(&[i64s(&[1, 2])]).unwrap();
        assert_eq!(idx.left.iter().collect::<Vec<_>>(), vec![Some(0), Some(1)]);
        assert_eq!(idx.right.iter().collect::<Vec<_>>(), vec![None, Some(0)]);
    }
}
