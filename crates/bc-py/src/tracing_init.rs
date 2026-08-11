//! Rust data-plane `tracing` → Python `logging` bridge.
//!
//! `init_tracing` installs a global subscriber whose only layer forwards each event to
//! the Python `batcher.engine` logger, so data-plane spans/events land in the *same*
//! configured logging hierarchy (console + rotating file) the control plane uses — there
//! is one place to read engine logs, not two. The filter is set from the Python
//! `ObservabilityConfig.log_level`; at the default `WARNING` the `tracing` macros compile
//! to a cheap level check, so an ordinary query pays essentially nothing.
//!
//! The engine's *own* events are emitted at operator/stage granularity, never per row, so
//! the per-event GIL acquisition here is bounded — a handful per query. **Third-party
//! events are not**, and that difference is load-bearing rather than cosmetic. Several
//! crates the media path links (`symphonia`, and any decoder reachable from `bc-expr`) log
//! through the `log` facade, which `tracing-subscriber` bridges into `tracing` by default.
//! A decoder emits one record per payload it cannot identify, so on a mixed unstructured
//! corpus that is one *per row* — each one acquiring the GIL, serializing the rayon fan-out
//! that `eval::media::map_rows` exists to parallelize. Measured on 2,000 non-audio blobs
//! through `.audio.to_waveform()`, forwarding them cost **123 µs/row against 3.1 µs/row**
//! with the bridge quiet: a 40x penalty on the *failure* path, which is 33x the cost of a
//! row that decodes successfully.
//!
//! The severity was wrong as well, and that is the worse half. `symphonia` logs a payload
//! it cannot parse at ERROR, but for a media expression that is the documented *expected*
//! outcome — the media convention reports an undecodable payload as a null row, not a
//! failure. So a perfectly healthy job over a corpus of mixed blobs filled the engine log
//! with thousands of ERROR lines, which is alarm fatigue that hides the real ones.
//!
//! `is_foreign` therefore splits the two populations, and `init_tracing` drops the foreign
//! one unless the level explicitly asks for debug detail. It is applied as part of the
//! per-layer `Filter` rather than as a check inside `on_event` on purpose: a `Filter` whose
//! verdict depends only on metadata reports `Interest::never()` for the callsite, so the
//! record is never turned into an event at all. Checking inside `on_event` would still pay
//! to construct every one.
//!
//! Hiding a whole population is only safe because of what that population contains here,
//! and that was checked rather than assumed. Exactly thirteen crates in the lock file
//! depend on `log`: the six `symphonia` crates (the per-row decoders above), the three
//! Cranelift crates plus `regalloc2` (JIT compilation internals), `reqwest`/`rustls`
//! (HTTP and TLS internals), and `iana-time-zone`. None of them reports something an
//! operator would act on at `WARNING`. In particular `object_store`, which is where a
//! genuinely actionable storage warning would come from (a retried S3 request, a throttled
//! bucket), is **not** in that list — it does not use the `log` facade, so its diagnostics
//! never travelled this path and are not affected. Re-check this list when adding a
//! dependency: one that logs actionable warnings through `log` would need a narrower
//! filter than "foreign", not a wider level.

use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Metadata, Subscriber};
use tracing_subscriber::filter::{filter_fn, FilterExt, LevelFilter};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

static INIT: OnceLock<()> = OnceLock::new();

/// The field `tracing-log` stamps on every event it synthesizes from a `log` record.
///
/// Its presence is how a record from a third-party crate using the `log` facade is told
/// apart from an event an engine crate emitted through `tracing` directly. The field set
/// is static per callsite, so this test is answered once per callsite and cached, not
/// re-evaluated per event.
const LOG_ORIGIN_FIELD: &str = "log.target";

/// Whether this callsite is a third-party crate logging through the `log` facade.
fn is_foreign(metadata: &Metadata<'_>) -> bool {
    metadata.fields().field(LOG_ORIGIN_FIELD).is_some()
}

/// Install the data-plane tracing subscriber once, forwarding to Python `logging`.
///
/// `level` is a Python level name (`"DEBUG"`/`"INFO"`/`"WARNING"`/`"ERROR"`); `json` is
/// accepted for API symmetry but ignored — the Python logging formatter owns the record
/// layout. A global subscriber can be set only once per process, so the first call wins
/// and later calls (e.g. a level change) are no-ops.
///
/// Records from third-party crates are forwarded only when `level` is `DEBUG` or finer.
/// They are per-row on the media paths and describe expected outcomes, so at ordinary
/// levels they are noise that costs more than the work they describe (see the module
/// docs); at `DEBUG` the caller has asked to see everything, and they reappear.
#[pyfunction]
#[pyo3(signature = (level="WARNING", json=false))]
pub fn init_tracing(level: &str, json: bool) -> PyResult<()> {
    let _ = json;
    let filter = level_filter(level);
    // `DEBUG` and `TRACE` are the levels at which a caller is asking for third-party
    // detail; every level above it is asking about the engine.
    let verbose = filter >= LevelFilter::DEBUG;
    let engine_only = filter_fn(move |metadata| verbose || !is_foreign(metadata));
    INIT.get_or_init(|| {
        // `try_init` returns Err if another subscriber is already global; ignore it so a
        // host that installed its own tracing stack is respected rather than panicked on.
        //
        // `engine_only` is the left operand because `And::callsite_enabled` short-circuits
        // on a `never` from the left — which is exactly the outcome that keeps a foreign
        // callsite from ever building an event.
        let _ = tracing_subscriber::registry()
            .with(PyLogBridge.with_filter(engine_only.and(filter)))
            .try_init();
    });
    Ok(())
}

fn level_filter(level: &str) -> LevelFilter {
    match level.to_ascii_uppercase().as_str() {
        "TRACE" => LevelFilter::TRACE,
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "WARN" | "WARNING" => LevelFilter::WARN,
        "ERROR" | "CRITICAL" => LevelFilter::ERROR,
        // `observability.verbosity="silent"` asks for no data-plane tracing at all. Without
        // this arm "OFF" fell through to the catch-all and enabled WARN — so the quietest
        // setting still paid to emit and filter events, and "silent" was not silent here.
        "OFF" | "NONE" => LevelFilter::OFF,
        _ => LevelFilter::WARN,
    }
}

/// A `tracing` layer that forwards each event to the Python `batcher.engine` logger.
struct PyLogBridge;

impl<S: Subscriber> Layer<S> for PyLogBridge {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        // A third-party crate's idea of an error is not the engine's. `symphonia` logs at
        // ERROR for a payload it cannot identify, which on a media path is the expected
        // outcome for every non-audio row rather than a fault. Foreign records only reach
        // here at DEBUG in the first place, so report them at the level that asked for
        // them and keep ERROR meaning "the engine has a problem".
        let level = if is_foreign(metadata) {
            Level::DEBUG
        } else {
            *metadata.level()
        };
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        Python::with_gil(|py| {
            let _ = forward(py, level, &visitor.message);
        });
    }
}

/// Collects the event's `message` field (and any other fields, appended) into one string.
#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else if !self.message.is_empty() {
            self.message
                .push_str(&format!(" {}={value:?}", field.name()));
        } else {
            self.message = format!("{}={value:?}", field.name());
        }
    }
}

/// The `batcher.engine` logger, resolved once per process.
///
/// `logging.getLogger` returns the *same* object for a given name for the life of the
/// process, so holding the handle is behaviour-preserving rather than a snapshot: handlers
/// and levels are mutated on that object, and a reconfiguration after this point is still
/// seen. What it avoids is re-importing `logging` and re-walking the logger dictionary on
/// every event, which the `DEBUG` path pays per record.
static LOGGER: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

fn logger<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let cached = LOGGER.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py
            .import("logging")?
            .getattr("getLogger")?
            .call1(("batcher.engine",))?
            .unbind())
    })?;
    Ok(cached.bind(py).clone())
}

fn forward(py: Python<'_>, level: Level, message: &str) -> PyResult<()> {
    let logger = logger(py)?;
    let method = match level {
        Level::ERROR => "error",
        Level::WARN => "warning",
        Level::INFO => "info",
        Level::DEBUG | Level::TRACE => "debug",
    };
    logger.call_method1(method, (message,))?;
    Ok(())
}
