//! Rust data-plane `tracing` → Python `logging` bridge.
//!
//! `init_tracing` installs a global subscriber whose only layer forwards each event to
//! the Python `batcher.engine` logger, so data-plane spans/events land in the *same*
//! configured logging hierarchy (console + rotating file) the control plane uses — there
//! is one place to read engine logs, not two. The filter is set from the Python
//! `ObservabilityConfig.log_level`; at the default `WARNING` the `tracing` macros compile
//! to a cheap level check, so an ordinary query pays essentially nothing.
//!
//! Events are emitted only at operator/stage granularity (never per row), so the
//! per-event GIL acquisition here is bounded — a handful per query.

use std::sync::OnceLock;

use pyo3::prelude::*;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

static INIT: OnceLock<()> = OnceLock::new();

/// Install the data-plane tracing subscriber once, forwarding to Python `logging`.
///
/// `level` is a Python level name (`"DEBUG"`/`"INFO"`/`"WARNING"`/`"ERROR"`); `json` is
/// accepted for API symmetry but ignored — the Python logging formatter owns the record
/// layout. A global subscriber can be set only once per process, so the first call wins
/// and later calls (e.g. a level change) are no-ops.
#[pyfunction]
#[pyo3(signature = (level="WARNING", json=false))]
pub fn init_tracing(level: &str, json: bool) -> PyResult<()> {
    let _ = json;
    let filter = level_filter(level);
    INIT.get_or_init(|| {
        // `try_init` returns Err if another subscriber is already global; ignore it so a
        // host that installed its own tracing stack is respected rather than panicked on.
        let _ = tracing_subscriber::registry()
            .with(PyLogBridge.with_filter(filter))
            .try_init();
    });
    Ok(())
}

fn level_filter(level: &str) -> LevelFilter {
    match level.to_ascii_uppercase().as_str() {
        "TRACE" => LevelFilter::TRACE,
        "DEBUG" => LevelFilter::DEBUG,
        "INFO" => LevelFilter::INFO,
        "ERROR" | "CRITICAL" => LevelFilter::ERROR,
        _ => LevelFilter::WARN,
    }
}

/// A `tracing` layer that forwards each event to the Python `batcher.engine` logger.
struct PyLogBridge;

impl<S: Subscriber> Layer<S> for PyLogBridge {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let level = *event.metadata().level();
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

fn forward(py: Python<'_>, level: Level, message: &str) -> PyResult<()> {
    let logging = py.import("logging")?;
    let logger = logging.getattr("getLogger")?.call1(("batcher.engine",))?;
    let method = match level {
        Level::ERROR => "error",
        Level::WARN => "warning",
        Level::INFO => "info",
        Level::DEBUG | Level::TRACE => "debug",
    };
    logger.call_method1(method, (message,))?;
    Ok(())
}
