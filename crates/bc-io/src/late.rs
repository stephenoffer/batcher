//! Late materialization: decode a row group's predicate columns first, evaluate the caller's
//! own filter on them, and decode every other column only for the rows that survive.
//!
//! [`crate::row_filter`] pushes the *control plane's* predicate translation into the decode,
//! and only where it can prove arrow's comparison kernels agree with the engine's. That leaves
//! out most of what a `WHERE` clause holds — `LIKE`, a date range, anything the translation
//! cannot spell — and it never installs on a read of one row group, which is how the engine's
//! own workers read (`read_parquet_row_group`). ClickBench's
//! `SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10` therefore decoded
//! all 105 columns of all 10,000,000 rows to keep 646 of them.
//!
//! Here the predicate is opaque: a [`RowPredicate`] is the columns it reads and a function
//! from a batch of them to a keep mask. `bc-io` cannot evaluate the engine's expressions (the
//! crate DAG points the other way), so the caller supplies that function — `bc-py` hands in the
//! engine's own `Filter` predicate, evaluated by the same evaluator over the same normalized
//! columns the `Filter` above the scan sees. That is what makes removing rows here sound
//! without any of `row_filter`'s superset reasoning: the mask *is* the engine's mask, and the
//! engine still applies its `Filter` to whatever is returned.
//!
//! # When it pays, decided by measuring both
//!
//! Late materialization trades one straight decode of every column for a decode of the
//! predicate columns plus a selective decode of the rest. Whether that wins depends on things
//! no footer records: how many rows survive, how *scattered* they are (a fragmented selection
//! decodes more slowly per row than a straight one, and a page with one survivor is still
//! decompressed whole), and how wide the deferred columns are against the predicate's.
//! `row_filter` measured the same trade crossing over below 10 % selected on one shape, and
//! nothing about that figure carries to another.
//!
//! So the scan measures it. Until it has decided, it alternates — even-numbered row groups are
//! read with the filter, odd ones without — and times the decode per row each way. Once each
//! side has [`SAMPLES`] row groups it keeps the faster for the rest of the scan; a filter seen
//! keeping over [`CLEAR_LOSS`] of its rows is dropped without waiting. A wrong verdict costs
//! speed, never rows.
//!
//! The kept fraction cannot settle a win the same way, which is the measurement that argued
//! for timing both sides. Over ClickBench's DuckDB-written `hits`, a filter keeping *nothing*
//! still read slower than none on most shapes — 512 against 296 ns per row for
//! `URL LIKE '%google%' AND SearchPhrase <> ''`, 54 against 42 for `SearchPhrase <> ''` with
//! `ClientIP` and three integers deferred — because the selective decode still decompresses
//! every page holding a survivor, and a scattered selection leaves few pages without one. It
//! won, by 1.9x, where the deferred columns dwarf the predicate's: `SELECT *` behind a `LIKE`.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, RecordBatch};
use parquet::arrow::arrow_reader::{ArrowPredicate, ArrowPredicateFn, RowFilter};
use parquet::schema::types::SchemaDescriptor;

/// A keep mask over a batch of a predicate's columns. Must return exactly one non-null value
/// per row; `true` keeps the row.
pub type MaskFn = dyn Fn(&RecordBatch) -> BooleanArray + Send + Sync;

/// One stage of a late-materialized read: the columns it decodes, and the mask it computes.
///
/// Stages run in order, each over only the rows every earlier stage kept, so a caller splitting
/// a conjunction should put the cheapest conjunct first — and must split only a conjunction
/// whose conjuncts cannot raise on the rows an earlier one removes.
#[derive(Clone)]
pub struct RowPredicate {
    /// The top-level column names the mask reads.
    pub columns: Vec<String>,
    /// The mask itself.
    pub mask: Arc<MaskFn>,
}

/// Row groups each way the scan times before keeping the faster.
pub const SAMPLES: u64 = 3;

/// A kept fraction this high settles the question without sampling the unfiltered side.
/// Keeping most rows, the filter can save at most a fraction of the deferred columns' decode
/// while making all of it fragmented.
pub const CLEAR_LOSS: f64 = 0.5;

/// Rows the filtered side must have read before [`CLEAR_LOSS`] is trusted.
const CLEAR_LOSS_ROWS: u64 = 100_000;

const UNDECIDED: u8 = 0;
const ON: u8 = 1;
const OFF: u8 = 2;

/// What the reads taken one way have cost.
#[derive(Default)]
struct Side {
    units: AtomicU64,
    rows: AtomicU64,
    kept: AtomicU64,
    nanos: AtomicU64,
}

impl Side {
    fn add(&self, rows: u64, kept: u64, nanos: u64) {
        self.units.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(rows, Ordering::Relaxed);
        self.kept.fetch_add(kept, Ordering::Relaxed);
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// Nanoseconds per row read, or `None` before any row has been.
    fn per_row(&self) -> Option<f64> {
        let rows = self.rows.load(Ordering::Relaxed);
        (rows > 0).then(|| self.nanos.load(Ordering::Relaxed) as f64 / rows as f64)
    }
}

/// A scan's late-materialization filter, shared by every worker reading its row groups.
pub struct LateFilter {
    predicates: Vec<RowPredicate>,
    started: AtomicU64,
    on: Side,
    off: Side,
    state: AtomicU8,
}

impl LateFilter {
    /// A filter running `predicates` in order. `None` when there is nothing to run, or when
    /// late materialization is disabled (`BATCHER_PARQUET_LATE_FILTER=0`).
    #[must_use]
    pub fn new(predicates: Vec<RowPredicate>) -> Option<Self> {
        if !enabled() || predicates.is_empty() || predicates.iter().any(|p| p.columns.is_empty()) {
            return None;
        }
        Some(Self {
            predicates,
            started: AtomicU64::new(0),
            on: Side::default(),
            off: Side::default(),
            state: AtomicU8::new(UNDECIDED),
        })
    }

    /// Whether the next row group should be read with the filter installed.
    pub(crate) fn install(&self) -> bool {
        match self.state.load(Ordering::Relaxed) {
            ON => true,
            OFF => false,
            _ => self.started.fetch_add(1, Ordering::Relaxed).is_multiple_of(2),
        }
    }

    /// Record a row group of `rows` rows read in `nanos`, `installed` or not, keeping `kept`.
    pub(crate) fn record(&self, installed: bool, rows: u64, kept: u64, nanos: u64) {
        if self.state.load(Ordering::Relaxed) != UNDECIDED {
            return;
        }
        let side = if installed { &self.on } else { &self.off };
        side.add(rows, kept, nanos);
        if let Some(verdict) = self.verdict() {
            // Only the first verdict counts; a racing worker's later one is discarded.
            let _ = self.state.compare_exchange(
                UNDECIDED,
                verdict,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    }

    /// The verdict the samples so far support, if any.
    fn verdict(&self) -> Option<u8> {
        let on_rows = self.on.rows.load(Ordering::Relaxed);
        let kept = self.on.kept.load(Ordering::Relaxed);
        if on_rows >= CLEAR_LOSS_ROWS && (kept as f64) > CLEAR_LOSS * on_rows as f64 {
            return Some(OFF);
        }
        if self.on.units.load(Ordering::Relaxed) < SAMPLES
            || self.off.units.load(Ordering::Relaxed) < SAMPLES
        {
            return None;
        }
        let (on, off) = (self.on.per_row()?, self.off.per_row()?);
        Some(if on <= off { ON } else { OFF })
    }

    /// Whether `rg` holds enough to defer for the filter to be worth installing on it at all.
    ///
    /// The filter decodes its first stage's columns in full and saves only on the rest, so the
    /// rest must be at least as large: deferring a narrow column behind a wide predicate (a
    /// `LIKE` over two text columns, say, to save an integer key) can only cost. Measured in
    /// the footer's uncompressed bytes, which is what the decode handles. `columns` is the
    /// read's projection, `None` for every column.
    pub(crate) fn worth_deferring(
        &self,
        rg: &parquet::file::metadata::RowGroupMetaData,
        columns: Option<&[String]>,
    ) -> bool {
        let first = &self.predicates[0].columns;
        let (mut decoded, mut deferred) = (0i64, 0i64);
        for chunk in rg.columns() {
            let Some(root) = chunk.column_path().parts().first() else {
                continue;
            };
            let size = chunk.uncompressed_size();
            if first.iter().any(|c| c == root) {
                decoded += size;
            } else if columns.is_none_or(|cols| cols.iter().any(|c| c == root)) {
                deferred += size;
            }
        }
        deferred > 0 && deferred >= decoded
    }

    /// The parquet `RowFilter` running every stage, against the file's schema.
    pub(crate) fn row_filter(&self, descr: &SchemaDescriptor) -> RowFilter {
        let stages: Vec<Box<dyn ArrowPredicate>> = self
            .predicates
            .iter()
            .map(|p| {
                let projection =
                    crate::projection::exact_columns(descr, p.columns.iter().map(String::as_str));
                let mask = p.mask.clone();
                Box::new(ArrowPredicateFn::new(
                    projection,
                    move |batch: RecordBatch| {
                        let m = mask(&batch);
                        // A mask that does not fit its batch is a caller bug; keeping every row is
                        // the answer that cannot lose one.
                        Ok(if m.len() == batch.num_rows() && m.null_count() == 0 {
                            m
                        } else {
                            BooleanArray::from(vec![true; batch.num_rows()])
                        })
                    },
                )) as Box<dyn ArrowPredicate>
            })
            .collect();
        RowFilter::new(stages)
    }
}

/// Whether late materialization is enabled (`BATCHER_PARQUET_LATE_FILTER=0` disables it).
///
/// The A/B switch it was measured with, in the shape `row_filter_enabled` already uses: one
/// binary run both ways separates the effect from a shared machine's noise where two builds
/// cannot.
fn enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("BATCHER_PARQUET_LATE_FILTER").as_deref() != Ok("0"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int64Array, StringArray};
    use arrow::compute::filter_record_batch;
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use super::*;

    /// A file of `rows` rows in row groups of `per_group`: `a` = 0..rows (every 11th null),
    /// `s` a wide text column (every 5th null), `b` = a * 3. Small pages, so a selective read
    /// can skip some.
    fn write(path: &std::path::Path, rows: i64, per_group: usize, dictionary: bool) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("b", DataType::Int64, false),
        ]));
        let a = Int64Array::from(
            (0..rows)
                .map(|i| (i % 11 != 0).then_some(i))
                .collect::<Vec<_>>(),
        );
        let s = StringArray::from(
            (0..rows)
                .map(|i| (i % 5 != 0).then(|| format!("row-{i}-{}", "x".repeat((i % 40) as usize))))
                .collect::<Vec<_>>(),
        );
        let b = Int64Array::from((0..rows).map(|i| i * 3).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(a) as ArrayRef, Arc::new(s), Arc::new(b)],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(per_group))
            .set_data_page_row_count_limit(500)
            .set_write_batch_size(500)
            .set_dictionary_enabled(dictionary)
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    /// `a % 7 == 3`, null-free, over a batch holding `a`.
    fn mask(batch: &RecordBatch) -> BooleanArray {
        let a = batch
            .column_by_name("a")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        a.iter()
            .map(|v| Some(v.is_some_and(|v| v % 7 == 3)))
            .collect()
    }

    fn filter_of(columns: &[&str]) -> LateFilter {
        LateFilter::new(vec![RowPredicate {
            columns: columns.iter().map(|c| (*c).to_string()).collect(),
            mask: Arc::new(mask),
        }])
        .unwrap()
    }

    /// Row group `rg` of `path`, read by arrow-rs's own synchronous reader and filtered by
    /// `mask` — the oracle a late-materialized read must equal, row for row and in order.
    fn oracle(path: &std::path::Path, rg: usize) -> RecordBatch {
        let file = std::fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .with_row_groups(vec![rg])
            .build()
            .unwrap();
        let batches: Vec<RecordBatch> = reader.map(Result::unwrap).collect();
        let all = concat(&batches);
        filter_record_batch(&all, &mask(&all)).unwrap()
    }

    fn concat(batches: &[RecordBatch]) -> RecordBatch {
        arrow::compute::concat_batches(&batches[0].schema(), batches).unwrap()
    }

    fn read(path: &std::path::Path, rg: usize, late: &Arc<LateFilter>) -> RecordBatch {
        let uri = path.to_str().unwrap();
        concat(&crate::read_parquet_row_group_late(uri, rg, None, 1024, None, Some(late)).unwrap())
    }

    /// With the filter installed, a row group comes back as exactly the rows the mask keeps —
    /// every column, nulls and all, in file order — for plain and dictionary encodings alike.
    #[test]
    fn an_installed_filter_returns_exactly_the_masked_rows() {
        let dir = std::env::temp_dir().join(format!("bcio_late_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for dictionary in [false, true] {
            let path = dir.join(format!("t{dictionary}.parquet"));
            write(&path, 20_000, 6_000, dictionary);
            let late = Arc::new(filter_of(&["a"]));
            late.state.store(ON, Ordering::Relaxed);
            for rg in 0..4 {
                let want = oracle(&path, rg);
                assert!(want.num_rows() > 0);
                assert_eq!(
                    read(&path, rg, &late),
                    want,
                    "rg {rg}, dictionary {dictionary}"
                );
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Off, a row group comes back whole — a superset the caller's `Filter` reduces to the
    /// same rows.
    #[test]
    fn an_uninstalled_filter_returns_every_row() {
        let dir = std::env::temp_dir().join(format!("bcio_late_off_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.parquet");
        write(&path, 9_000, 3_000, true);
        let late = Arc::new(filter_of(&["a"]));
        late.state.store(OFF, Ordering::Relaxed);
        let got = read(&path, 1, &late);
        assert_eq!(got.num_rows(), 3_000);
        assert_eq!(
            filter_record_batch(&got, &mask(&got)).unwrap(),
            oracle(&path, 1)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A projection the first stage already covers has nothing to defer, so the filter is not
    /// worth installing; one that adds a wide column is.
    #[test]
    fn deferring_needs_a_column_the_first_stage_does_not_read() {
        use parquet::file::reader::FileReader;
        let dir = std::env::temp_dir().join(format!("bcio_late_defer_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.parquet");
        write(&path, 6_000, 6_000, false);
        let file = std::fs::File::open(&path).unwrap();
        let reader = parquet::file::serialized_reader::SerializedFileReader::new(file).unwrap();
        let rg = reader.metadata().row_group(0);
        let cols = |names: &[&str]| names.iter().map(|c| (*c).to_string()).collect::<Vec<_>>();
        let late = filter_of(&["a"]);
        assert!(!late.worth_deferring(rg, Some(&cols(&["a"]))));
        assert!(late.worth_deferring(rg, Some(&cols(&["a", "s"]))));
        assert!(late.worth_deferring(rg, None));
        // A wide predicate column guarding a narrow deferred one is not worth it.
        let wide = filter_of(&["s"]);
        assert!(!wide.worth_deferring(rg, Some(&cols(&["s", "b"]))));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The verdict keeps whichever side read faster per row once both have their samples, and
    /// drops a filter that keeps most of its rows without sampling the other side at all.
    #[test]
    fn the_verdict_follows_the_measurement() {
        let faster_filtered = filter_of(&["a"]);
        for _ in 0..SAMPLES {
            assert_eq!(faster_filtered.state.load(Ordering::Relaxed), UNDECIDED);
            faster_filtered.record(true, 100_000, 1_000, 1_000_000);
            faster_filtered.record(false, 100_000, 100_000, 2_000_000);
        }
        assert_eq!(faster_filtered.state.load(Ordering::Relaxed), ON);
        assert!(faster_filtered.install());

        let slower_filtered = filter_of(&["a"]);
        for _ in 0..SAMPLES {
            slower_filtered.record(true, 100_000, 1_000, 3_000_000);
            slower_filtered.record(false, 100_000, 100_000, 2_000_000);
        }
        assert_eq!(slower_filtered.state.load(Ordering::Relaxed), OFF);
        assert!(!slower_filtered.install());

        let permissive = filter_of(&["a"]);
        permissive.record(true, 200_000, 150_000, 1_000);
        assert_eq!(permissive.state.load(Ordering::Relaxed), OFF);

        // Undecided, it alternates, starting with the filter on.
        let fresh = filter_of(&["a"]);
        assert!(fresh.install());
        assert!(!fresh.install());
        assert!(fresh.install());
    }

    /// A mask that does not fit its batch keeps every row rather than losing any.
    #[test]
    fn a_misfit_mask_keeps_every_row() {
        let dir = std::env::temp_dir().join(format!("bcio_late_misfit_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.parquet");
        write(&path, 3_000, 3_000, false);
        let late = Arc::new(
            LateFilter::new(vec![RowPredicate {
                columns: vec!["a".to_string()],
                mask: Arc::new(|_: &RecordBatch| BooleanArray::from(vec![false; 3])),
            }])
            .unwrap(),
        );
        late.state.store(ON, Ordering::Relaxed);
        assert_eq!(read(&path, 0, &late).num_rows(), 3_000);
        std::fs::remove_dir_all(&dir).ok();
    }
}
