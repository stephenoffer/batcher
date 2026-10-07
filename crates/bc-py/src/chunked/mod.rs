//! The FFI entry point for streaming one source into the engine chunk by chunk.
//!
//! `execute_plan` takes every source resident, which is what forces the control plane to decode
//! a whole table before the engine may start (see `bc_interp::stream::chunked`). Here the
//! driving source is a Python iterator of batch lists instead. The engine pulls the next chunk
//! only between its own parallel steps, on the calling thread, reacquiring the GIL for exactly
//! as long as it takes to advance the iterator and take ownership of the batches — so a Python
//! producer that decodes on a background thread (the control plane's prefetching Parquet reader)
//! keeps decoding while the engine computes.

use std::sync::Arc;

use arrow::array::{Array, RecordBatch};
use arrow::datatypes::Schema;
use arrow_pyarrow::PyArrowType;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;

use crate::normalize::{narrow_output, normalize_batch, rebase_nested_offsets};

mod late;
mod resident;
use crate::{errors, prepare_exec, ExecSetup};

/// Whether `plan_json` can run with source `driving` streamed in chunks.
#[pyfunction]
pub(crate) fn plan_chunkable(plan_json: &str, driving: usize) -> PyResult<bool> {
    let plan = bc_ir::RelOp::from_json(plan_json).map_err(errors::ir_to_pyerr)?;
    Ok(bc_interp::chunkable(&plan, driving))
}

/// Execute `plan_json` with `sources[driving]` read from `chunks`, an iterator of batch lists.
///
/// `sources[driving]` carries only the driving relation's schema (a zero-row batch), and
/// `memory_budget` bounds what the path may hold (the tighter of it and the engine config's).
/// Raises `MemoryBudgetExceededError` when that is exceeded, which the caller
/// answers by taking an out-of-core path; the rows returned otherwise are exactly those
/// `execute_plan` returns over the concatenated chunks.
#[pyfunction]
#[pyo3(signature = (plan_json, sources, driving, chunks, engine_config="", query_id=None, memory_budget=0))]
pub(crate) fn execute_plan_chunked(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    driving: usize,
    chunks: Bound<'_, PyAny>,
    engine_config: &str,
    query_id: Option<&str>,
    memory_budget: usize,
) -> PyResult<Vec<PyArrowType<RecordBatch>>> {
    let args = (plan_json, engine_config, query_id, memory_budget);
    chunked_call(py, args, sources, driving, chunks, false).map(|(out, _)| out)
}

/// [`execute_plan_chunked`], also returning the per-operator metrics document
/// `execute_plan_metered` returns, summed over every chunk (`bc_interp::execute_chunked_metered`).
#[pyfunction]
#[pyo3(signature = (plan_json, sources, driving, chunks, engine_config="", query_id=None, memory_budget=0))]
pub(crate) fn execute_plan_chunked_metered(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    driving: usize,
    chunks: Bound<'_, PyAny>,
    engine_config: &str,
    query_id: Option<&str>,
    memory_budget: usize,
) -> PyResult<(Vec<PyArrowType<RecordBatch>>, String)> {
    let args = (plan_json, engine_config, query_id, memory_budget);
    chunked_call(py, args, sources, driving, chunks, true)
        .map(|(out, m)| (out, m.unwrap_or_default()))
}

/// The body of both chunked entry points; `metered` adds the metrics document.
type ChunkedOut = (Vec<PyArrowType<RecordBatch>>, Option<String>);
fn chunked_call(
    py: Python<'_>,
    (plan_json, engine_config, query_id, memory_budget): (&str, &str, Option<&str>, usize),
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    driving: usize,
    chunks: Bound<'_, PyAny>,
    metered: bool,
) -> PyResult<ChunkedOut> {
    let ExecSetup {
        plan,
        sources,
        opts,
        narrow,
        budget,
        ..
    } = prepare_exec(plan_json, sources, engine_config, query_id)?;
    // What the chunked path holds — aggregate state, or a spine's collected output — must stay
    // inside an envelope even when the engine config names none, because the chunked path's only
    // fallback is the caller's out-of-core one: over budget, it must say so rather than grow.
    let budget = match (budget, memory_budget) {
        (0, m) => m,
        (b, 0) => b,
        (b, m) => b.min(m),
    };
    let iterator: Py<PyAny> = chunks.try_iter()?.into_any().unbind();
    // `auto_width` caps the width by the resident sources' size, and the driving one is not
    // resident here: honour an explicit width, and otherwise give the chunks the operator cores
    // a resident run over them would get.
    let workers = if opts.parallelism > 0 {
        opts.parallelism
    } else {
        bc_arrow::operator_cores().max(1)
    };
    // A Python exception raised by the chunk iterator (a strict-schema `SchemaError`, say) is
    // kept here and re-raised as itself: the engine only sees that the producer failed, and the
    // caller must get the same exception the resident read would have raised.
    let raised: std::sync::Mutex<Option<PyErr>> = std::sync::Mutex::new(None);
    let out = py.detach(|| {
        let mut next = || {
            Python::attach(|py| next_chunk(py, &iterator))
                .map_err(|e| {
                    let message = e.to_string();
                    if let Ok(mut slot) = raised.lock() {
                        slot.get_or_insert(e);
                    }
                    bc_interp::InterpError::ChunkSource(message)
                })
                .transpose()
        };
        if metered {
            bc_interp::execute_chunked_metered(
                &plan, &sources, driving, &mut next, workers, budget, &opts,
            )
            .map(|(out, metrics)| (out, Some(metrics.to_json())))
        } else {
            bc_interp::execute_chunked(&plan, &sources, driving, &mut next, workers, budget, &opts)
                .map(|out| (out, None))
        }
    });
    if let Some(err) = raised.into_inner().ok().flatten() {
        return Err(err);
    }
    let (out, metrics) = out.map_err(errors::interp_to_pyerr)?;
    let out = bc_interp::coalesce_small_batches(rebase_nested_offsets(narrow_output(out, &narrow)));
    Ok((out.into_iter().map(PyArrowType).collect(), metrics))
}

/// The iterator's next chunk, normalized exactly as `prepare_exec` normalizes a resident source.
fn next_chunk(py: Python<'_>, iterator: &Py<PyAny>) -> PyResult<Option<Vec<RecordBatch>>> {
    let item = match iterator.bind(py).call_method0("__next__") {
        Ok(item) => item,
        Err(e) if e.is_instance_of::<PyStopIteration>(py) => return Ok(None),
        Err(e) => return Err(e),
    };
    let batches: Vec<PyArrowType<RecordBatch>> = item.extract()?;
    batches
        .iter()
        .map(|b| normalize_batch(&b.0))
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

/// The row groups of a list of Parquet files, read one at a time by the engine's own workers.
///
/// Each read decodes on the worker that asked for it (`bc_io::read_parquet_row_group`), which then
/// computes over the rows and frees them on the same thread. Reading ahead onto the I/O pool was
/// measured and dropped: with the decode already on every worker it only added threads competing
/// for the same cores (TPC-H sf10 q6, 158 ms without it, 182 ms with).
///
/// What *is* read ahead is a remote unit's **bytes** ([`bc_io::prefetch_row_group`]), which costs
/// no core: one GET at a time per worker left a node latency-bound at a fifth of its link
/// (TPC-H sf1000 on three 16-core nodes, ~250 MB/s received with the CPUs 12-24% busy). A local
/// file reads from the page cache and is never prefetched.
struct ParquetUnits<'a> {
    uris: &'a [String],
    columns: Option<&'a [String]>,
    predicate: Option<&'a str>,
    batch_size: usize,
    units: Vec<(usize, usize)>,
    /// Rows across every unit, from the footers `parquet_row_groups` already read.
    rows: usize,
    /// The plan's own `Filter` over this scan, decoded first and shared by every unit read.
    late: Option<Arc<bc_io::LateFilter>>,
    /// That `Filter` as stages, for a keyed read to run after its key stages.
    plan_stages: Vec<bc_io::RowPredicate>,
    /// The columns every read decodes, which a keyed read's key column must be among.
    read_columns: Vec<String>,
    /// The filter a keyed read installs ([`late::keyed_late`]), built on the first one: every
    /// unit of one execution is handed the same keys, and sharing one filter is what lets
    /// `bc_io::LateFilter` time the keyed read across units and keep the faster way.
    keyed: std::sync::OnceLock<Option<Arc<bc_io::LateFilter>>>,
    /// Which units a read-ahead was started for, so each is requested once.
    prefetched: Vec<std::sync::atomic::AtomicBool>,
}

/// Units ahead of the one being read whose bytes are fetched in the background
/// (`BATCHER_UNIT_PREFETCH`, `0` to turn it off).
fn unit_prefetch_depth() -> usize {
    static D: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *D.get_or_init(|| {
        std::env::var("BATCHER_UNIT_PREFETCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(12)
    })
}

impl<'a> ParquetUnits<'a> {
    /// The driving scan over `units` of `uris`, with the plan's `Filter` over source `driving`
    /// as its late filter. `carrier` is that source's schema-carrying batches.
    #[allow(clippy::too_many_arguments)]
    fn new(
        uris: &'a [String],
        columns: Option<&'a [String]>,
        predicate: Option<&'a str>,
        batch_size: usize,
        units: Vec<(usize, usize)>,
        rows: usize,
        plan: &bc_ir::RelOp,
        driving: usize,
        carrier: &[RecordBatch],
    ) -> Self {
        let plan_stages = late::plan_stages(plan, driving, carrier);
        let read_columns = late::read_columns(carrier, columns);
        Self {
            uris,
            columns,
            predicate,
            batch_size: batch_size.max(1),
            rows,
            late: late::late_of(plan_stages.clone(), &read_columns),
            plan_stages,
            read_columns,
            keyed: std::sync::OnceLock::new(),
            prefetched: (0..units.len())
                .map(|_| std::sync::atomic::AtomicBool::new(false))
                .collect(),
            units,
        }
    }

    /// Start the byte reads of the units after `unit`, which this worker reads next: its units
    /// are a contiguous range read in order (`bc_interp`'s `unit_ranges`).
    fn read_ahead(&self, unit: usize) {
        use std::sync::atomic::Ordering;
        let depth = unit_prefetch_depth();
        for next in (unit + 1)..(unit + 1 + depth).min(self.units.len()) {
            if self.prefetched[next].swap(true, Ordering::Relaxed) {
                continue;
            }
            let (file, rg) = self.units[next];
            bc_io::prefetch_row_group(&self.uris[file], rg, self.columns);
        }
    }

    /// Unit `unit`, decoded with `late` installed.
    fn read_with(
        &self,
        unit: usize,
        late: Option<&Arc<bc_io::LateFilter>>,
    ) -> Result<Vec<RecordBatch>, bc_interp::InterpError> {
        let source = |e: String| bc_interp::InterpError::ChunkSource(e);
        self.read_ahead(unit);
        let (file, rg) = self.units[unit];
        bc_io::read_parquet_row_group_late(
            &self.uris[file],
            rg,
            self.columns,
            self.batch_size,
            self.predicate,
            late,
            false,
        )
        .map_err(|e| source(e.to_string()))?
        .iter()
        .map(|b| normalize_batch(b).map_err(|e| source(e.to_string())))
        .collect()
    }
}

impl bc_interp::UnitSource for ParquetUnits<'_> {
    fn units(&self) -> usize {
        self.units.len()
    }

    fn rows(&self) -> Option<usize> {
        Some(self.rows)
    }

    fn read(&self, unit: usize) -> Result<Vec<RecordBatch>, bc_interp::InterpError> {
        self.read_with(unit, self.late.as_ref())
    }

    /// Unit `unit` with the runtime join filters on this scan tested during the decode, so the
    /// columns no key stage reads are decoded only for the rows the keys do not refute.
    ///
    /// TPC-H q18 is the shape it is for: the outer `lineitem` scan probes a join whose build side
    /// is 624 orders, and decoding it whole materialized every column of 60M rows (sf10) to keep
    /// 4,368. Falls back to [`Self::read`] when no key is among the scan's columns, or when the
    /// keys would leave nothing to defer.
    fn read_keyed(
        &self,
        unit: usize,
        keys: &[bc_interp::ScanKeyFilter],
    ) -> Result<Vec<RecordBatch>, bc_interp::InterpError> {
        if keys.is_empty() {
            return self.read(unit);
        }
        let keyed = self
            .keyed
            .get_or_init(|| late::keyed_late(keys, &self.plan_stages, &self.read_columns));
        match keyed {
            Some(late) => self.read_with(unit, Some(late)),
            None => self.read(unit),
        }
    }

    /// The relation as `columns` plus each row's locator, when the rest is worth deferring.
    ///
    /// Worth it when the columns left for the fetch are at least as large, uncompressed, as the
    /// ones read now — the same yardstick `bc_io::LateFilter` defers by. A locator is the file's
    /// index in the high 24 bits and the row's position in it in the low 40.
    fn narrowed(&self, columns: &[String]) -> Option<Box<dyn bc_interp::UnitSource + '_>> {
        if self.uris.len() >= 1 << LOCATOR_FILE_SHIFT_BITS {
            return None;
        }
        let bytes = bc_io::parquet_column_bytes(self.uris).ok()?;
        let projected = |name: &str| {
            self.columns
                .is_none_or(|cols| cols.iter().any(|c| c == name))
        };
        let (mut read, mut deferred) = (0u64, 0u64);
        for (name, size) in &bytes {
            if columns.iter().any(|c| c == name) {
                read += size;
            } else if projected(name) {
                deferred += size;
            }
        }
        (deferred > 0 && deferred >= read).then(|| {
            Box::new(NarrowUnits {
                base: self,
                columns: columns.to_vec(),
            }) as Box<dyn bc_interp::UnitSource + '_>
        })
    }

    fn fetch(&self, locators: &[u64]) -> Result<Vec<RecordBatch>, bc_interp::InterpError> {
        fetch_located(self, locators)
    }
}

/// Bits of a locator holding the row's position in its file; the file index sits above them.
const LOCATOR_ROW_BITS: u32 = 40;

/// Bits left for the file index.
const LOCATOR_FILE_SHIFT_BITS: u32 = 64 - LOCATOR_ROW_BITS;

/// [`ParquetUnits`] reading only some columns, each row tagged with its locator.
struct NarrowUnits<'s, 'a> {
    base: &'s ParquetUnits<'a>,
    columns: Vec<String>,
}

impl bc_interp::UnitSource for NarrowUnits<'_, '_> {
    fn units(&self) -> usize {
        self.base.units()
    }

    fn rows(&self) -> Option<usize> {
        self.base.rows()
    }

    fn read(&self, unit: usize) -> Result<Vec<RecordBatch>, bc_interp::InterpError> {
        let source = |e: String| bc_interp::InterpError::ChunkSource(e);
        let (file, rg) = self.base.units[unit];
        let out = bc_io::read_parquet_row_group_late(
            &self.base.uris[file],
            rg,
            Some(&self.columns),
            self.base.batch_size,
            self.base.predicate,
            self.base.late.as_ref(),
            true,
        )
        .map_err(|e| source(e.to_string()))?;
        out.iter()
            .map(|b| {
                located(
                    &normalize_batch(b).map_err(|e| source(e.to_string()))?,
                    &self.columns,
                    file,
                )
            })
            .collect()
    }
}

/// `batch`'s `columns` in that order, then its locators, from the reader's row numbers.
fn located(
    batch: &RecordBatch,
    columns: &[String],
    file: usize,
) -> Result<RecordBatch, bc_interp::InterpError> {
    let source = |e: String| bc_interp::InterpError::ChunkSource(e);
    let rows = batch
        .column_by_name(bc_io::ROW_NUMBER)
        .and_then(|c| c.as_any().downcast_ref::<arrow::array::Int64Array>())
        .ok_or_else(|| source("a locating read returned no row numbers".into()))?;
    let high = (file as u64) << LOCATOR_ROW_BITS;
    let mut locators = Vec::with_capacity(rows.len());
    for row in rows.values() {
        let row = u64::try_from(*row).map_err(|e| source(e.to_string()))?;
        if row >> LOCATOR_ROW_BITS != 0 {
            return Err(source(format!("row {row} is past the locator's range")));
        }
        locators.push(high | row);
    }
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(columns.len() + 1);
    let mut arrays = Vec::with_capacity(columns.len() + 1);
    for name in columns {
        let i = schema.index_of(name).map_err(|e| source(e.to_string()))?;
        fields.push(schema.field(i).clone());
        arrays.push(batch.column(i).clone());
    }
    fields.push(arrow::datatypes::Field::new(
        bc_interp::LOCATOR,
        arrow::datatypes::DataType::UInt64,
        false,
    ));
    arrays.push(Arc::new(arrow::array::UInt64Array::from(locators)));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).map_err(|e| source(e.to_string()))
}

/// The rows `locators` name, every projected column, in `locators`' order.
///
/// Each file is asked once, for its rows in position order (`bc_io::read_parquet_rows`), and
/// the answers are put back in the order asked for.
fn fetch_located(
    units: &ParquetUnits<'_>,
    locators: &[u64],
) -> Result<Vec<RecordBatch>, bc_interp::InterpError> {
    let source = |e: String| bc_interp::InterpError::ChunkSource(e);
    let mask = (1u64 << LOCATOR_ROW_BITS) - 1;
    let mut by_file: std::collections::BTreeMap<usize, Vec<(u64, usize)>> =
        std::collections::BTreeMap::new();
    for (at, loc) in locators.iter().enumerate() {
        let file = usize::try_from(loc >> LOCATOR_ROW_BITS).map_err(|e| source(e.to_string()))?;
        by_file.entry(file).or_default().push((loc & mask, at));
    }
    let mut pieces: Vec<RecordBatch> = Vec::new();
    // For each requested position, where its row landed in the concatenated pieces.
    let mut slot = vec![0u32; locators.len()];
    let mut base = 0usize;
    for (file, mut rows) in by_file {
        let uri = units
            .uris
            .get(file)
            .ok_or_else(|| source(format!("no file {file} behind a locator")))?;
        rows.sort_unstable();
        rows.dedup_by_key(|(row, _)| *row);
        let positions: Vec<u64> = rows.iter().map(|(row, _)| *row).collect();
        let got = bc_io::read_parquet_rows(uri, &positions, units.columns, units.batch_size)
            .map_err(|e| source(e.to_string()))?;
        let got: Vec<RecordBatch> = got
            .iter()
            .map(|b| normalize_batch(b).map_err(|e| source(e.to_string())))
            .collect::<Result<_, _>>()?;
        let n: usize = got.iter().map(RecordBatch::num_rows).sum();
        if n != positions.len() {
            return Err(source(format!("fetched {n} rows of {}", positions.len())));
        }
        pieces.extend(got);
        let index: std::collections::HashMap<u64, usize> = positions
            .iter()
            .enumerate()
            .map(|(i, p)| (*p, base + i))
            .collect();
        for (at, loc) in locators.iter().enumerate() {
            if usize::try_from(loc >> LOCATOR_ROW_BITS).ok() == Some(file) {
                slot[at] =
                    u32::try_from(index[&(loc & mask)]).map_err(|e| source(e.to_string()))?;
            }
        }
        base += n;
    }
    let Some(first) = pieces.first() else {
        return Ok(Vec::new());
    };
    let all = arrow::compute::concat_batches(&first.schema(), &pieces)
        .map_err(|e| source(e.to_string()))?;
    let order = arrow::array::UInt32Array::from(slot);
    let columns = all
        .columns()
        .iter()
        .map(|c| arrow::compute::take(c.as_ref(), &order, None))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| source(e.to_string()))?;
    let out = RecordBatch::try_new(all.schema(), columns).map_err(|e| source(e.to_string()))?;
    Ok(vec![out])
}

/// Execute `plan_json` with `sources[driving]` read from Parquet `uris` by the engine's workers.
///
/// Each worker decodes a contiguous range of the files' row groups one at a time and pushes each
/// straight through its pipeline (`bc_interp::execute_units`), so decoding and computing overlap
/// across the pool and no worker holds more than one decoded row group of the driving relation.
/// `columns` and `predicate` are the scan's pushed projection and native predicate, exactly as
/// the resident read takes them; `sources[driving]` is the zero-row schema carrier. Budget and
/// errors as [`execute_plan_chunked`]; a plan that cannot run this way raises the engine's
/// not-chunkable error, and the caller reads the source resident instead.
///
/// Returns the rows and the per-operator metrics document `execute_plan_metered` returns: every
/// driving row passes through exactly once here, so the counts are the query's own and the
/// learning loop records them (`bc_interp::execute_units_metered`).
#[pyfunction]
#[pyo3(signature = (plan_json, sources, driving, uris, columns=None, predicate=None, batch_size=65536, engine_config="", query_id=None, memory_budget=0, resident=Vec::new()))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_plan_parquet(
    py: Python<'_>,
    plan_json: &str,
    sources: Vec<Vec<PyArrowType<RecordBatch>>>,
    driving: usize,
    uris: Vec<String>,
    columns: Option<Vec<String>>,
    predicate: Option<String>,
    batch_size: usize,
    engine_config: &str,
    query_id: Option<&str>,
    memory_budget: usize,
    resident: Vec<resident::ResidentRead>,
) -> PyResult<(Vec<PyArrowType<RecordBatch>>, String)> {
    let ExecSetup {
        plan,
        mut sources,
        opts,
        narrow,
        budget,
        ..
    } = prepare_exec(plan_json, sources, engine_config, query_id)?;
    let budget = match (budget, memory_budget) {
        (0, m) => m,
        (b, 0) => b,
        (b, m) => b.min(m),
    };
    let query_watch = bc_interp::QueryStopwatch::start();
    // Every usable core, not `operator_cores`' SMT-discounted width: each worker here spends
    // most of its time decoding Parquet (Snappy, then value copies), which is the work `bc-io`
    // already sizes its own pool by `usable_cores` for. At the operator width a 48-thread host
    // decoded on 30 threads and left the rest idle. An explicit width still wins, which is how
    // a concurrency grant or a user's `parallelism` bounds it.
    let workers = if opts.parallelism > 0 {
        opts.parallelism
    } else {
        bc_arrow::usable_cores().max(1)
    };
    let out = py.detach(|| {
        // The plan's other Parquet scans, which the control plane handed over as schema
        // carriers plus how to read them (`resident`).
        resident::read_into(&resident, &mut sources, workers)?;
        // Pruned before the units are ranged across the workers, so a clustered predicate's
        // survivors are spread over them rather than left in the few ranges they fall in.
        let groups = bc_io::parquet_row_groups_surviving(&uris, predicate.as_deref())
            .map_err(|e| bc_interp::InterpError::ChunkSource(e.to_string()))?;
        let rows = groups.iter().map(|&(_, _, n)| n).sum();
        let src = ParquetUnits::new(
            &uris,
            columns.as_deref(),
            predicate.as_deref(),
            batch_size,
            groups.into_iter().map(|(file, rg, _)| (file, rg)).collect(),
            rows,
            &plan,
            driving,
            &sources[driving],
        );
        bc_interp::execute_units_metered(&plan, &sources, driving, &src, workers, budget, &opts)
    });
    let (out, metrics) = out.map_err(errors::interp_to_pyerr)?;
    let metrics = metrics.with_query(query_watch);
    let out = bc_interp::coalesce_small_batches(rebase_nested_offsets(narrow_output(out, &narrow)));
    Ok((
        out.into_iter().map(PyArrowType).collect(),
        metrics.to_json(),
    ))
}

/// The map side of a distributed aggregate over Parquet row groups, read by the engine's workers.
///
/// `units` are the partition's `(uri, row group)` pairs; `map_json` is the breaker-free map
/// prefix over source 0, and `carrier` its zero-row schema batch. Each worker reads a contiguous
/// range of the units one at a time, runs the prefix and folds the rows into its own partial
/// (`bc_interp::partial_aggregate_units`), so decoding overlaps computing across the pool and
/// no mapped row crosses into Python. Returns the partial-state batch `partial_aggregate`
/// returns for the same rows -- the reducers cannot tell the paths apart -- and the map
/// prefix's per-operator metrics document, numbered as the map plan is.
///
/// Raises the engine's not-chunkable error for a prefix the unit executor cannot stream; the
/// caller then reads the partition itself.
#[pyfunction]
#[pyo3(signature = (map_json, group_keys_json, aggregates_json, carrier, units, columns=None, predicate=None, batch_size=65536, engine_config=""))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn partial_aggregate_parquet(
    py: Python<'_>,
    map_json: &str,
    group_keys_json: &str,
    aggregates_json: &str,
    carrier: Vec<PyArrowType<RecordBatch>>,
    units: Vec<(String, usize)>,
    columns: Option<Vec<String>>,
    predicate: Option<String>,
    batch_size: usize,
    engine_config: &str,
) -> PyResult<(PyArrowType<RecordBatch>, String)> {
    let group_keys = crate::normalize::parse_group_keys(group_keys_json)?;
    let aggregates = crate::normalize::parse_aggregates(aggregates_json)?;
    let ExecSetup {
        plan,
        sources,
        opts,
        ..
    } = prepare_exec(map_json, vec![carrier], engine_config, None)?;
    // As `execute_plan_parquet`: every usable core, since each worker mostly decodes.
    let workers = if opts.parallelism > 0 {
        opts.parallelism
    } else {
        bc_arrow::usable_cores().max(1)
    };
    // The files once each, in first-seen order, and each unit as (file index, row group).
    let mut uris: Vec<String> = Vec::new();
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let pairs: Vec<(usize, usize)> = units
        .into_iter()
        .map(|(uri, rg)| {
            let file = *index.entry(uri.clone()).or_insert_with(|| {
                uris.push(uri);
                uris.len() - 1
            });
            (file, rg)
        })
        .collect();
    let out = py.detach(|| {
        // Footer row counts (cached by the reader), for the fold's size hints.
        let counts: std::collections::HashMap<(usize, usize), usize> =
            bc_io::parquet_row_groups(&uris)
                .map_err(|e| bc_interp::InterpError::ChunkSource(e.to_string()))?
                .into_iter()
                .map(|(file, rg, n)| ((file, rg), n))
                .collect();
        // The units the footers prove empty, dropped before the fold ranges them across its
        // workers (see `execute_plan_parquet`). Only those: a unit the footers do not list at
        // all stays, so a bad index still fails its read rather than vanishing.
        let surviving: std::collections::HashSet<(usize, usize)> =
            bc_io::parquet_row_groups_surviving(&uris, predicate.as_deref())
                .map_err(|e| bc_interp::InterpError::ChunkSource(e.to_string()))?
                .into_iter()
                .map(|(file, rg, _)| (file, rg))
                .collect();
        let mut pairs = pairs;
        pairs.retain(|u| surviving.contains(u) || !counts.contains_key(u));
        let rows = pairs.iter().filter_map(|u| counts.get(u)).sum();
        let src = ParquetUnits::new(
            &uris,
            columns.as_deref(),
            predicate.as_deref(),
            batch_size,
            pairs,
            rows,
            &plan,
            0,
            &sources[0],
        );
        bc_interp::partial_aggregate_units(
            &plan,
            &group_keys,
            &aggregates,
            &sources,
            0,
            &src,
            workers,
            &opts,
        )
    });
    let (batch, metrics) = out.map_err(errors::interp_to_pyerr)?;
    Ok((
        PyArrowType(crate::normalize::rebase_batch(batch)),
        metrics.to_json(),
    ))
}

/// Register this module's entry points on the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(execute_plan_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan_chunked_metered, m)?)?;
    m.add_function(wrap_pyfunction!(plan_chunkable, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(partial_aggregate_parquet, m)?)?;
    Ok(())
}
