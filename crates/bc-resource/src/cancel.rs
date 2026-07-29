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

    /// Whether `other` is the *same* flag as this token, not merely an equal one.
    ///
    /// Identity, because that is the question the registry needs: two tokens for the same
    /// query id are indistinguishable by value (both are a bool that is false) and are
    /// completely different things.
    #[must_use]
    fn is(&self, other: &CancelToken) -> bool {
        Arc::ptr_eq(&self.flag, &other.flag)
    }
}

/// Process-wide map of query id to its token.
///
/// A poisoned lock is recovered rather than propagated (`into_inner`), matching the memory
/// pool. A panic in one query must not make the whole process uncancellable: the map holds
/// only `Arc`s to flags, so there is no invariant a panic could have left half-updated, and
/// silently returning "not registered" for every future query would turn one panic into a
/// process where Ctrl-C stops working.
fn registry() -> &'static Mutex<HashMap<String, CancelToken>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, CancelToken>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The registry map, recovering a poisoned lock.
fn locked() -> std::sync::MutexGuard<'static, HashMap<String, CancelToken>> {
    registry().lock().unwrap_or_else(|e| e.into_inner())
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
    locked().insert(query_id.to_string(), token.clone());
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
    locked().get(query_id).cloned()
}

/// Drop `query_id` from the registry. Call when the query finishes, however it finished.
///
/// Prefer [`unregister_token`] wherever the caller still holds the token it registered.
/// This unconditional form removes whatever entry is under the id, which is wrong in the
/// one case the id is not unique — see there.
pub fn unregister(query_id: &str) {
    locked().remove(query_id);
}

/// Drop `query_id` **only if** the registered token is still `token`.
///
/// The re-registration hazard, and the reason this exists. `register` documents that
/// reusing an id replaces the entry and cannot orphan the running query — which is true of
/// `register` and false of the cleanup that follows it. The first query finishes, calls
/// `unregister(id)`, and removes the *second* query's registration: that query is now
/// running with a token nothing can reach, so Ctrl-C on it does nothing at all and the
/// caller is told `false` as though it had already finished.
///
/// Comparing identity closes it: a stale holder finds a token that is not its own and
/// leaves the live registration alone.
///
/// Args are the id and the token `register` returned for it.
///
/// Returns whether an entry was removed.
pub fn unregister_token(query_id: &str, token: &CancelToken) -> bool {
    let mut map = locked();
    match map.get(query_id) {
        Some(current) if current.is(token) => {
            map.remove(query_id);
            true
        }
        _ => false,
    }
}

/// Cancel `query_id`.
///
/// Returns whether a query with that id was registered. `false` means it already finished
/// or never started, which is information for the caller and not an error — the race
/// between a cancel and a completion has no correct loser.
pub fn cancel(query_id: &str) -> bool {
    match locked().get(query_id) {
        Some(token) => {
            token.cancel();
            true
        }
        None => false,
    }
}

/// Ids of every query currently registered, sorted.
///
/// Sorted rather than "unspecified order": this is a diagnostic that a human reads and a
/// test asserts on, and `HashMap` iteration order varies run to run, which makes both
/// harder for no benefit at these sizes.
#[must_use]
pub fn running() -> Vec<String> {
    let mut ids: Vec<String> = locked().keys().cloned().collect();
    ids.sort();
    ids
}

/// Cancel every registered query. For process shutdown.
///
/// Returns how many were cancelled, so a shutdown path can say whether it interrupted
/// anything rather than guessing.
pub fn cancel_all() -> usize {
    let map = locked();
    for token in map.values() {
        token.cancel();
    }
    map.len()
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
    fn a_stale_unregister_cannot_orphan_the_live_registration() {
        // Two queries reuse one id (a control plane recycling after a crash). The first
        // finishes and cleans up; the second is still running and must stay cancellable.
        let first = register("q-recycled");
        let second = register("q-recycled");

        assert!(
            !unregister_token("q-recycled", &first),
            "a stale token removed an entry"
        );
        assert!(cancel("q-recycled"), "the live query became uncancellable");
        assert!(second.is_cancelled());
        assert!(!first.is_cancelled());

        assert!(unregister_token("q-recycled", &second));
        assert!(!cancel("q-recycled"));
    }

    #[test]
    fn unregister_token_reports_whether_it_removed_anything() {
        let token = register("q-report");
        assert!(unregister_token("q-report", &token));
        assert!(
            !unregister_token("q-report", &token),
            "removing twice reported a removal"
        );
        assert!(!unregister_token("q-never", &token));
    }

    #[test]
    fn running_is_sorted_so_a_reader_and_a_test_can_rely_on_it() {
        for id in ["q-sort-c", "q-sort-a", "q-sort-b"] {
            let _ = register(id);
        }
        let ids: Vec<String> = running()
            .into_iter()
            .filter(|s| s.starts_with("q-sort-"))
            .collect();
        assert_eq!(ids, vec!["q-sort-a", "q-sort-b", "q-sort-c"]);
        for id in ["q-sort-a", "q-sort-b", "q-sort-c"] {
            unregister(id);
        }
    }

    #[test]
    fn cancel_all_reports_how_many_it_interrupted() {
        let a = register("q-all-1");
        let b = register("q-all-2");
        assert!(cancel_all() >= 2);
        assert!(a.is_cancelled() && b.is_cancelled());
        unregister("q-all-1");
        unregister("q-all-2");
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
