//! Cooperative cancellation: a flag the executor polls, and the registry that finds it.
//!
//! # Why cooperative, and not a kill
//!
//! `bc_py::execute_plan` wraps the whole of execution in `Python::allow_threads`, which
//! releases the GIL. That is what lets other Python threads run while a query executes —
//! and it is also why a `SIGINT` arriving during a ten-minute `collect()` cannot be
//! delivered until the native call returns. Python's signal handler only runs between
//! bytecodes, and there are no bytecodes to run between. The interpreter is not hung; it
//! is simply not scheduled.
//!
//! There is no safe way to interrupt the native call from outside. Killing a thread
//! mid-operator leaks its Arrow buffers, its memory-pool reservation, and any spill file
//! it has open. So cancellation is *cooperative*: something sets a flag, and the executor
//! checks it at points where unwinding is already safe — between morsels, between
//! operators, between spill runs. Those are the points where the executor is holding only
//! values it can drop.
//!
//! # The cost
//!
//! [`CancelToken::is_cancelled`] is a `Relaxed` load of an `AtomicBool`. `Relaxed` is
//! deliberate and sufficient: cancellation needs no happens-before relationship with
//! anything, only *eventual* visibility, and it costs a plain load with no fence on every
//! architecture the engine targets. Polled once per 16,384-row morsel it is unmeasurable
//! against the morsel's own work — but "unmeasurable" is what everyone says about their
//! own added check, so it is benchmarked rather than asserted.
//!
//! # Ownership
//!
//! The registry is process-global because the thing it indexes — a running query in this
//! process — is. It holds only `Arc`s to flags: cancelling a query that already finished
//! is a no-op rather than an error, because the race between "the user hit Ctrl-C" and
//! "the query returned" has no correct loser.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// A shared flag one side sets and the executor polls.
///
/// Cloning shares the flag — every clone observes a `cancel()` on any other. That is the
/// point: the token is handed to the executor while the canceller keeps its own handle.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// A fresh, uncancelled token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Idempotent; safe from any thread.
    pub fn cancel(&self) {
        // `Release` pairs with nothing in particular — the flag carries no data — but it
        // costs nothing on the *setting* side, which happens once per cancellation rather
        // than once per morsel, and it keeps the store from being sunk past a caller's
        // bookkeeping.
        self.flag.store(true, Ordering::Release);
    }

    /// Whether cancellation has been requested. The hot-path read.
    #[must_use]
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

/// Process-wide map of query id to its token.
fn registry() -> &'static Mutex<HashMap<String, CancelToken>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, CancelToken>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register `query_id` as running and return the token the executor should poll.
///
/// Re-registering an id that is already present replaces it and returns the new token.
/// That is the right behavior for a control plane that reuses an id after a crash, and it
/// cannot orphan a running query: the old token's holder still owns its `Arc`, so it keeps
/// polling a flag that simply nobody can reach any more.
#[must_use]
pub fn register(query_id: &str) -> CancelToken {
    let token = CancelToken::new();
    if let Ok(mut map) = registry().lock() {
        map.insert(query_id.to_string(), token.clone());
    }
    token
}

/// The token for `query_id`, or `None` if it is not registered.
///
/// This is how the executor gets the flag it polls. It deliberately does **not** register:
/// the control plane owns a query's lifetime, and it opens the registration *before* it
/// starts optimizing, so a cancel arriving while the plan is still being built lands on a
/// token that already exists. An executor that registered its own would drop exactly those
/// cancels on the floor, and the user pressing Ctrl-C during a slow optimize would see
/// nothing happen.
#[must_use]
pub fn token_for(query_id: &str) -> Option<CancelToken> {
    registry().lock().ok()?.get(query_id).cloned()
}

/// Drop `query_id` from the registry. Call when the query finishes, however it finished.
pub fn unregister(query_id: &str) {
    if let Ok(mut map) = registry().lock() {
        map.remove(query_id);
    }
}

/// Cancel `query_id`.
///
/// Returns whether a query with that id was registered. `false` means it already finished
/// or never started, which is information for the caller and not an error — the race
/// between a cancel and a completion has no correct loser.
pub fn cancel(query_id: &str) -> bool {
    let Ok(map) = registry().lock() else {
        return false;
    };
    match map.get(query_id) {
        Some(token) => {
            token.cancel();
            true
        }
        None => false,
    }
}

/// Ids of every query currently registered, in unspecified order.
#[must_use]
pub fn running() -> Vec<String> {
    registry()
        .lock()
        .map(|map| map.keys().cloned().collect())
        .unwrap_or_default()
}

/// Cancel every registered query. For process shutdown.
pub fn cancel_all() {
    if let Ok(map) = registry().lock() {
        for token in map.values() {
            token.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_token_is_not_cancelled() {
        assert!(!CancelToken::new().is_cancelled());
    }

    #[test]
    fn a_clone_observes_the_original_being_cancelled() {
        // The whole mechanism: the canceller and the executor hold different handles.
        let held_by_executor = CancelToken::new();
        let held_by_canceller = held_by_executor.clone();
        assert!(!held_by_executor.is_cancelled());
        held_by_canceller.cancel();
        assert!(held_by_executor.is_cancelled());
    }

    #[test]
    fn cancelling_twice_is_a_no_op() {
        let token = CancelToken::new();
        token.cancel();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn token_for_finds_a_registered_query_and_misses_an_unregistered_one() {
        let token = register("q-lookup");
        let found = token_for("q-lookup").expect("a registered query had no token");
        found.cancel();
        assert!(
            token.is_cancelled(),
            "the looked-up token was not the registered one"
        );
        unregister("q-lookup");
        assert!(token_for("q-lookup").is_none());
    }

    #[test]
    fn a_registered_query_is_cancellable_by_id() {
        let token = register("q-registered");
        assert!(cancel("q-registered"));
        assert!(token.is_cancelled());
        unregister("q-registered");
    }

    #[test]
    fn cancelling_an_unknown_id_reports_false_rather_than_failing() {
        // The completion/cancel race: the query finished first. Not an error.
        assert!(!cancel("q-never-existed"));
    }

    #[test]
    fn unregistering_makes_a_later_cancel_a_no_op() {
        let token = register("q-finished");
        unregister("q-finished");
        assert!(!cancel("q-finished"));
        assert!(
            !token.is_cancelled(),
            "a finished query was cancelled after the fact"
        );
    }

    #[test]
    fn running_lists_registered_ids() {
        let _a = register("q-list-a");
        let _b = register("q-list-b");
        let ids = running();
        assert!(ids.contains(&"q-list-a".to_string()));
        assert!(ids.contains(&"q-list-b".to_string()));
        unregister("q-list-a");
        unregister("q-list-b");
    }

    #[test]
    fn re_registering_an_id_does_not_orphan_the_first_token() {
        // The old holder keeps polling its own flag; it is merely unreachable by id.
        let first = register("q-reused");
        let second = register("q-reused");
        assert!(cancel("q-reused"));
        assert!(second.is_cancelled());
        assert!(
            !first.is_cancelled(),
            "cancelling by id reached a superseded token"
        );
        unregister("q-reused");
    }

    #[test]
    fn cancellation_is_visible_across_threads() {
        let token = CancelToken::new();
        let watcher = token.clone();
        let handle = std::thread::spawn(move || {
            while !watcher.is_cancelled() {
                std::hint::spin_loop();
            }
            true
        });
        token.cancel();
        assert!(handle.join().expect("watcher thread panicked"));
    }
}
