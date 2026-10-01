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

use arrow::array::{Array, BooleanArray, DictionaryArray, RecordBatch};
use arrow::datatypes::{DataType, Int32Type, Schema};
use arrow_pyarrow::PyArrowType;
use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;

use crate::normalize::{narrow_output, normalize_batch, rebase_nested_offsets};
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
        bc_interp::execute_chunked(&plan, &sources, driving, &mut next, workers, budget, &opts)
    });
    if let Some(err) = raised.into_inner().ok().flatten() {
        return Err(err);
    }
    let out = out.map_err(errors::interp_to_pyerr)?;
    let out = bc_interp::coalesce_small_batches(rebase_nested_offsets(narrow_output(out, &narrow)));
    Ok(out.into_iter().map(PyArrowType).collect())
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
}

impl bc_interp::UnitSource for ParquetUnits<'_> {
    fn units(&self) -> usize {
        self.units.len()
    }

    fn rows(&self) -> Option<usize> {
        Some(self.rows)
    }

    fn read(&self, unit: usize) -> Result<Vec<RecordBatch>, bc_interp::InterpError> {
        let source = |e: String| bc_interp::InterpError::ChunkSource(e);
        let (file, rg) = self.units[unit];
        bc_io::read_parquet_row_group_late(
            &self.uris[file],
            rg,
            self.columns,
            self.batch_size,
            self.predicate,
            self.late.as_ref(),
            false,
        )
        .map_err(|e| source(e.to_string()))?
        .iter()
        .map(|b| normalize_batch(b).map_err(|e| source(e.to_string())))
        .collect()
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
#[pyo3(signature = (plan_json, sources, driving, uris, columns=None, predicate=None, batch_size=65536, engine_config="", query_id=None, memory_budget=0))]
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
) -> PyResult<(Vec<PyArrowType<RecordBatch>>, String)> {
    let ExecSetup {
        plan,
        sources,
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
        // Pruned before the units are ranged across the workers, so a clustered predicate's
        // survivors are spread over them rather than left in the few ranges they fall in.
        let groups = bc_io::parquet_row_groups_surviving(&uris, predicate.as_deref())
            .map_err(|e| bc_interp::InterpError::ChunkSource(e.to_string()))?;
        let rows = groups.iter().map(|&(_, _, n)| n).sum();
        let src = ParquetUnits {
            uris: &uris,
            columns: columns.as_deref(),
            predicate: predicate.as_deref(),
            batch_size: batch_size.max(1),
            units: groups.into_iter().map(|(file, rg, _)| (file, rg)).collect(),
            rows,
            late: late_filter(&plan, driving, &sources[driving], columns.as_deref()),
        };
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
        let src = ParquetUnits {
            uris: &uris,
            columns: columns.as_deref(),
            predicate: predicate.as_deref(),
            batch_size: batch_size.max(1),
            units: pairs,
            rows,
            late: late_filter(&plan, 0, &sources[0], columns.as_deref()),
        };
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

/// The predicate of the plan's `Filter` directly over `Scan(driving)`, if it has one.
fn scan_filter(plan: &bc_ir::RelOp, driving: usize) -> Option<&bc_expr::Expr> {
    if let bc_ir::RelOp::Filter { input, predicate } = plan {
        if matches!(**input, bc_ir::RelOp::Scan { source_id } if source_id == driving) {
            return Some(predicate);
        }
    }
    plan.children()
        .into_iter()
        .find_map(|child| scan_filter(child, driving))
}

/// The plan's `Filter` over the driving scan, as a late-materialization filter for its reads.
///
/// The decode evaluates exactly what the `Filter` evaluates — the same expression, by the same
/// evaluator, over the same columns normalized the same way ([`filter_mask`]) — so the rows it
/// keeps are the rows that `Filter` keeps, and the `Filter` still runs over them.
///
/// A conjunction is split into stages, cheapest first, so a wide column is decoded only for the
/// rows the narrow ones kept. That evaluates a later conjunct over fewer rows, which is what the
/// engine's own `Filter` already does (`Expr::short_circuit_filter_mask`), and is sound on the
/// same condition: every conjunct is [infallible](bc_expr::Expr::is_infallible_predicate), so a
/// row skipped can never have been the one that raised. Its one remaining hazard is a
/// schema-driven error that only fires on a non-empty input, which a stage after one that kept
/// nothing would never see; so each conjunct is first evaluated over one all-null row of the
/// scan's schema, and a conjunction any of whose conjuncts fails that is pushed whole.
///
/// `None` when the stages would decode every column the scan reads: there is then nothing left
/// to defer, and the filter could only add its own evaluation to the read.
fn late_filter(
    plan: &bc_ir::RelOp,
    driving: usize,
    carrier: &[RecordBatch],
    columns: Option<&[String]>,
) -> Option<Arc<bc_io::LateFilter>> {
    let predicate = scan_filter(plan, driving)?;
    let schema = carrier.first()?.schema();
    let conjuncts = predicate.and_conjuncts();
    let split = conjuncts.len() > 1
        && conjuncts
            .iter()
            .all(|c| c.is_infallible_predicate(&schema) && evaluates_on_a_null_row(c, &schema));
    let stages = if split {
        let mut ordered = conjuncts;
        ordered.sort_by_key(|c| c.eval_cost());
        ordered.into_iter().map(|c| stage(c, &schema)).collect()
    } else {
        vec![stage(predicate, &schema)]
    };
    let staged = |name: &String| stages.iter().any(|s| s.columns.contains(name));
    let defers = match columns {
        Some(cols) => cols.iter().any(|c| !staged(c)),
        None => schema.fields().iter().any(|f| !staged(f.name())),
    };
    if !defers {
        return None;
    }
    bc_io::LateFilter::new(stages).map(Arc::new)
}

/// One late-materialization stage evaluating `expr`.
///
/// A stage over one string column that cannot raise accepts that column dictionary-encoded
/// (`dictionary`): its mask is then a pure function of each row's value, so it is computed once
/// per distinct value and looked up by key ([`dictionary_mask`]).
fn stage(expr: &bc_expr::Expr, schema: &Schema) -> bc_io::RowPredicate {
    let mut names: Vec<&str> = Vec::new();
    expr.collect_columns(&mut names);
    names.sort_unstable();
    names.dedup();
    let dictionary = matches!(names.as_slice(), [only] if schema
        .field_with_name(only)
        .is_ok_and(|f| matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8)))
        && expr.is_infallible_predicate(schema);
    let owned = expr.clone();
    bc_io::RowPredicate {
        columns: names.into_iter().map(str::to_string).collect(),
        mask: Arc::new(move |batch: &RecordBatch| {
            dictionary_mask(&owned, batch).unwrap_or_else(|| filter_mask(&owned, batch))
        }),
        dictionary,
    }
}

/// [`filter_mask`] over a batch whose one column is a `Dictionary`, computed per distinct value.
///
/// The predicate is evaluated over the dictionary's values, plus one null standing for the rows
/// whose key is null, and each row takes its key's answer. That is the mask the per-row
/// evaluation computes because the stage's predicate reads only this column and cannot raise
/// (see [`stage`]), so its answer for a row is a function of the row's value alone. `None` for
/// any other batch, and for a dictionary larger than the batch, where evaluating every value
/// would cost more than evaluating every row.
fn dictionary_mask(predicate: &bc_expr::Expr, batch: &RecordBatch) -> Option<BooleanArray> {
    let [column] = batch.columns() else {
        return None;
    };
    let dict = column
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()?;
    let values = dict.values();
    if values.len() > batch.num_rows() {
        return None;
    }
    let keys = dict.keys();
    let null_slot = values.len();
    let candidates = if keys.null_count() > 0 {
        let null = arrow::array::new_null_array(values.data_type(), 1);
        arrow::compute::concat(&[values.as_ref(), null.as_ref()]).ok()?
    } else {
        values.clone()
    };
    let field = batch
        .schema()
        .field(0)
        .clone()
        .with_data_type(values.data_type().clone())
        .with_nullable(true);
    let candidates =
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![candidates]).ok()?;
    let answers = filter_mask(predicate, &candidates);
    if answers.len() != candidates.num_rows() {
        return None;
    }
    let (codes, nulls) = (keys.values(), keys.nulls());
    let mask = arrow::buffer::BooleanBuffer::collect_bool(codes.len(), |i| {
        let slot = if nulls.is_some_and(|n| n.is_null(i)) {
            null_slot
        } else {
            usize::try_from(codes[i]).unwrap_or(null_slot)
        };
        answers.value(slot)
    });
    Some(BooleanArray::new(mask, None))
}

/// Whether `expr` evaluates to a boolean over one all-null row of `schema`'s columns.
fn evaluates_on_a_null_row(expr: &bc_expr::Expr, schema: &Schema) -> bool {
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone().with_nullable(true))
        .collect();
    let columns = fields
        .iter()
        .map(|f| arrow::array::new_null_array(f.data_type(), 1))
        .collect();
    let Ok(row) = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns) else {
        return false;
    };
    expr.eval(&row)
        .is_ok_and(|a| a.as_any().downcast_ref::<BooleanArray>().is_some())
}

/// The keep mask the engine's `Filter` computes for `predicate` over `batch`: null-free, and
/// every row kept when it cannot be computed, so the `Filter` above raises whatever it raises.
fn filter_mask(predicate: &bc_expr::Expr, batch: &RecordBatch) -> BooleanArray {
    let keep_all = || BooleanArray::from(vec![true; batch.num_rows()]);
    // The columns as the engine sees them: decoded straight from Parquet they may still be
    // narrow or dictionary-encoded, which `normalize_batch` undoes on the way to the `Filter`.
    let Ok(batch) = normalize_batch(batch) else {
        return keep_all();
    };
    let mask = match predicate.short_circuit_filter_mask(&batch) {
        Ok(Some(mask)) => mask,
        _ => match predicate.eval(&batch) {
            Ok(array) => match array.as_any().downcast_ref::<BooleanArray>() {
                Some(mask) => mask.clone(),
                None => return keep_all(),
            },
            Err(_) => return keep_all(),
        },
    };
    if mask.null_count() > 0 {
        arrow::compute::prep_null_mask_filter(&mask)
    } else {
        mask
    }
}

/// Register this module's entry points on the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(execute_plan_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(plan_chunkable, m)?)?;
    m.add_function(wrap_pyfunction!(execute_plan_parquet, m)?)?;
    m.add_function(wrap_pyfunction!(partial_aggregate_parquet, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::Field;

    fn expr(json: &str) -> bc_expr::Expr {
        serde_json::from_str(json).unwrap()
    }

    /// A one-column batch `m` holding `values` indexed by `keys` (a `None` key is a null row).
    fn dict_batch(values: &[Option<&str>], keys: &[Option<i32>]) -> RecordBatch {
        let dict = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(keys.to_vec()),
            Arc::new(StringArray::from(values.to_vec())),
        )
        .unwrap();
        let field = Field::new("m", dict.data_type().clone(), true);
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![Arc::new(dict)]).unwrap()
    }

    /// Per distinct value, the mask is the one the per-row evaluation of the decoded column
    /// computes — for membership, equality, a pattern and a null test, with null keys and a
    /// null dictionary value both present.
    #[test]
    fn a_dictionary_mask_equals_the_decoded_mask() {
        let values = [
            Some("MAIL"),
            Some("SHIP"),
            Some("AIR"),
            None,
            Some("AIR REG"),
        ];
        let keys: Vec<Option<i32>> = (0..64).map(|i| (i % 9 != 0).then_some(i * 7 % 5)).collect();
        let batch = dict_batch(&values, &keys);
        let decoded = normalize_batch(&batch).unwrap();
        assert_eq!(decoded.schema().field(0).data_type(), &DataType::Utf8);
        let col = r#"{"e":"col","name":"m"}"#;
        let air = r#"{"e":"lit","value":{"str":"AIR"}}"#;
        for predicate in [
            format!(r#"{{"e":"in_list","input":{col},"set":[{{"str":"MAIL"}},{{"str":"SHIP"}}]}}"#),
            format!(r#"{{"e":"binary","op":"eq","left":{col},"right":{air}}}"#),
            format!(r#"{{"e":"str","fn":"like","input":{col},"pattern":"AIR%"}}"#),
            format!(r#"{{"e":"is_null","input":{col}}}"#),
        ] {
            let e = expr(&predicate);
            let got = dictionary_mask(&e, &batch).expect("a dictionary batch takes this path");
            assert_eq!(got, filter_mask(&e, &decoded), "{predicate}");
        }
    }

    /// A dictionary with more values than the batch has rows, and a plain column, take the
    /// per-row path.
    #[test]
    fn a_large_dictionary_or_a_plain_column_is_declined() {
        let e = expr(r#"{"e":"is_null","input":{"e":"col","name":"m"}}"#);
        let values = [Some("a"), Some("b"), Some("c")];
        assert!(dictionary_mask(&e, &dict_batch(&values, &[Some(0), Some(1)])).is_none());
        let plain = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("m", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec![Some("a"), None]))],
        )
        .unwrap();
        assert!(dictionary_mask(&e, &plain).is_none());
    }
}
