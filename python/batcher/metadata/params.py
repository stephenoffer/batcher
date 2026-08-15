"""`LearnedParams` — the learned-parameter half of the store, and its parsed-read cache.

The Hub has two jobs that share only a backend. One is absorbing execution feedback and
maintaining the derived views over it (`hub.py`). The other is this: holding the *learned
parameters* each tuning loop reads at plan time and writes back after — cost coefficients,
cardinality corrections, converged credit windows, per-source throughputs, bandit posteriors.
They are separate responsibilities with separate failure modes, so they live in separate
modules and the Hub composes them.

Two storage shapes coexist, and both must keep working:

* **per key** — one backend entry per `(namespace, entry_key)`. This is the shape to write.
  A write touches one key, so two pipelines learning different shapes cannot lose each
  other's update, which the whole-blob read-modify-write below routinely did.
* **whole blob** — one entry per `(namespace,)` holding the entire `{key: value}` map. The
  legacy shape, still written by `save_params` for callers that genuinely replace a namespace
  wholesale. `load_keyed_params` merges it *underneath* the per-key entries, so a store
  written by an older build keeps answering and migrates a key at a time as each is rewritten.

Everything here is cached parsed, generation-stamped **per namespace**, and bounded. The
per-namespace part is not a detail: source statistics take one namespace per source path, so
a single shared generation counter meant that persisting one dataset's statistics invalidated
the parsed view of every unrelated namespace in the process.
"""

from __future__ import annotations

import json
from typing import Any

from batcher._internal.errors import ConfigError
from batcher.metadata.store import MetadataBackend

__all__ = ["LEARNED_PARAMS", "LearnedParams"]

#: The logical table learned parameters live in.
LEARNED_PARAMS = "learned_params"

# Sentinel for "the parsed view has no entry under this key" — distinct from a stored `None`,
# which `_unchanged` must be able to recognize as already-written.
_MISSING = object()

# Cap on the namespaces whose parsed views stay resident. Most namespaces are a fixed
# vocabulary (`kyber.calibration`, `carbonite.shuffle_window`, ...), but source statistics take
# one namespace *per source path*, so a session that reads thousands of distinct files would
# otherwise retain every one of their decoded blobs — bounds, blooms, and quantile grids
# included — for the life of the process. The cap sits well above the working set of any
# single query and evicts in insertion order, so what a served query re-reads stays warm.
_NAMESPACE_CACHE_MAX = 256


def check_namespace(namespace: str) -> None:
    """Reject a namespace that cannot address a stored entry.

    The store's keys are tuples the backends JSON-encode, so a non-string namespace
    writes under one spelling and reads back under another — a silent "the learning
    loop never persists anything", not an error.
    """
    if not isinstance(namespace, str) or not namespace:
        raise ConfigError(
            f"A learned-parameter namespace must be a non-empty string, but got "
            f"{type(namespace).__name__} {namespace!r}.",
            hint="Namespaces are dotted names, e.g. 'kyber.cardinality'.",
        )


def encoded(where: str, value: Any) -> bytes:
    """`value` as JSON bytes, or a typed error naming the entry that could not encode.

    `json.dumps` reports only the offending *type*, which in a map of learned stats is
    never enough to find the entry. Naming the namespace (and, for a dict, the key)
    turns a dead end into a one-line fix.
    """
    try:
        return json.dumps(value).encode()
    except TypeError as exc:
        culprit = ""
        if isinstance(value, dict):
            for key, item in value.items():
                try:
                    json.dumps(item)
                except TypeError:
                    culprit = f" (entry {key!r} has type {type(item).__name__})"
                    break
        raise ConfigError(
            f"Learned parameters for {where!r} are not JSON-serializable{culprit}: {exc}.",
            hint="Learned stats are stored as JSON, so use only str/int/float/bool/list/dict.",
        ) from exc


class LearnedParams:
    """Parsed, per-namespace-cached access to the learned-parameter table."""

    def __init__(self, backend: MetadataBackend) -> None:
        """Wrap an already-validated backend.

        Args:
            backend: Where learned state is stored. Validated by the Hub, which owns the
                only construction site.
        """
        self._backend = backend
        # Generation per namespace, bumped by a whole-blob write. Also the roster the
        # eviction walks: a namespace with no counter is invisible to the bound.
        self._generations: dict[str, int] = {}
        self._blob_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        self._keyed_cache: dict[str, tuple[int, dict[str, Any]]] = {}
        # Per namespace, the keys known to be backed by their own `(namespace, key)` backend
        # entry — as opposed to merged up from the legacy single-blob shape. `_unchanged`
        # only elides a redundant write for a key in here, so a legacy entry still migrates.
        self._keyed_stored: dict[str, set[str]] = {}
        # Every write that changed something, across every namespace. See `writes`.
        self._writes = 0

    def __repr__(self) -> str:
        """How many namespaces are resident — the question asked when a cache looks cold."""
        return f"LearnedParams(namespaces={len(self._generations)})"

    @property
    def writes(self) -> int:
        """Count of writes that changed the store, across every namespace.

        The per-namespace `_generations` counter cannot serve as a change signal for a
        *reader*, and neither can the identity of the dict `load_keyed` hands back:
        `put_keyed` deliberately patches that dict in place and leaves the generation alone,
        because the write *is* the new value of one entry and re-parsing the namespace on the
        next query would be pure waste. Both choices are right for the cache they serve and
        both make "has anything changed since I last looked?" unanswerable from outside.

        This answers it in one integer, monotonic and cheap, so a consumer that folds the
        whole store into a derived bundle can reuse the fold until a write actually lands.
        """
        return self._writes

    # --- whole-namespace blob ----------------------------------------------
    def load(self, namespace: str) -> dict[str, Any]:
        """Every learned parameter under `namespace`, or an empty dict.

        Args:
            namespace: The learning loop's name, e.g. ``"kyber.calibration"``.

        Returns:
            The stored parameters. This is the store's own parsed view, handed back without a
            copy so a repeat read costs a dict lookup rather than a backend round-trip and a
            `json.loads` of the whole blob. **Treat it as read-only**: mutating it edits what
            the next reader sees without writing anything, so the store and the view diverge.
            To change a stored value call `save` — or better `put_keyed`, which writes one
            entry and cannot lose a concurrent writer's update.

        Raises:
            ConfigError: If `namespace` is not a non-empty string.
        """
        check_namespace(namespace)
        generation = self._generation(namespace)
        cached = self._blob_cache.get(namespace)
        if cached is not None and cached[0] == generation:
            return cached[1]
        raw = self._backend.get(LEARNED_PARAMS, (namespace,))
        parsed = json.loads(raw) if raw else {}
        if not isinstance(parsed, dict):  # a foreign blob; never hand back a non-mapping
            parsed = {}
        self._blob_cache[namespace] = (generation, parsed)
        self._bound()
        return parsed

    def save(self, namespace: str, params: dict[str, Any]) -> None:
        """Replace every learned parameter under `namespace`.

        Args:
            namespace: The learning loop's name.
            params: The parameters to store. Must be JSON-serializable — the store
                holds opaque bytes so that any backend can serve it.

        Raises:
            ConfigError: If `namespace` is invalid, or `params` is not serializable.
                The offending key is named, because a `TypeError` reading "Object of
                type X is not JSON serializable" does not say *which* entry it was.
        """
        check_namespace(namespace)
        blob = encoded(namespace, params)
        self._backend.put(LEARNED_PARAMS, (namespace,), blob)
        # A whole-blob write is the one thing the per-key view cannot patch in place: it is
        # the *legacy* layer that view merges underneath its own entries. Only this
        # namespace's generation moves.
        generation = self._generation(namespace) + 1
        self._generations[namespace] = generation
        self._writes += 1
        # The blob view *can* be patched, because this write is its new whole value. Parsed
        # back rather than cached as `params`, so the view holds exactly what a reader of the
        # store would see and never aliases an object the caller still owns and may mutate.
        self._blob_cache[namespace] = (generation, json.loads(blob))
        self._bound()

    # --- per-key entries -----------------------------------------------------
    def load_keyed(self, namespace: str) -> dict[str, Any]:
        """The `{entry_key: value}` map for `namespace`, per-key entries over the legacy blob.

        Args:
            namespace: The learning loop's name.

        Returns:
            The store's parsed view. Read-only, for the reason `load` gives.
        """
        generation = self._generation(namespace)
        cached = self._keyed_cache.get(namespace)
        if cached is not None and cached[0] == generation:
            return cached[1]
        out: dict[str, Any] = {}
        legacy: dict[str, Any] = {}
        stored: set[str] = set()
        for key, value in self._backend.scan(LEARNED_PARAMS, (namespace,)):
            if len(key) >= 2:
                out[key[1]] = json.loads(value)
                stored.add(str(key[1]))
            elif len(key) == 1:
                legacy = json.loads(value)
        if isinstance(legacy, dict):
            for name, value in legacy.items():
                out.setdefault(name, value)  # per-key entries win over the legacy blob
        self._keyed_cache[namespace] = (generation, out)
        self._keyed_stored[namespace] = stored
        self._bound()
        return out

    def get_keyed(self, namespace: str, key: str) -> Any | None:
        """The learned value under `(namespace, key)`, or `None`.

        Served from the same parsed view `load_keyed` builds — in which a per-key entry
        already shadows the legacy blob's — so the reads the tuning loops issue several times
        per query cost a dict lookup rather than a store round-trip and a `json.loads` (and,
        on a miss, a second round-trip for the legacy blob).

        Args:
            namespace: The learning loop's name.
            key: The entry's name within the namespace.

        Returns:
            The stored value, or `None` when the namespace has no such entry.
        """
        return self.load_keyed(namespace).get(key)

    def put_keyed(self, namespace: str, key: str, value: Any) -> None:
        """Store one learned entry under `(namespace, key)`.

        Args:
            namespace: The learning loop's name.
            key: The entry's name within the namespace.
            value: A JSON-serializable value.

        Raises:
            ConfigError: If `namespace` or `key` is not a non-empty string, or `value`
                is not serializable. A non-string key would round-trip through JSON as
                a string and silently stop matching the key it was written under.
        """
        check_namespace(namespace)
        if not isinstance(key, str) or not key:
            raise ConfigError(
                f"A learned-parameter key must be a non-empty string, but got "
                f"{type(key).__name__} {key!r} in namespace {namespace!r}.",
                hint="Keys round-trip through JSON, so only strings survive unchanged.",
            )
        if self._unchanged(namespace, key, value):
            return
        blob = encoded(f"{namespace}.{key}", value)
        self._backend.put(LEARNED_PARAMS, (namespace, key), blob)
        self._writes += 1
        self._generation(namespace)  # register on the eviction roster before caching under it
        self._keyed_stored.setdefault(namespace, set()).add(key)
        # Patch the parsed view rather than invalidating it: this write *is* the new value of
        # exactly one entry, and the tuning loops read the namespace back on the very next
        # query. Invalidating instead would re-scan and re-parse the namespace each time. The
        # blob is parsed back rather than caching `value` itself, so the view holds exactly
        # what a reader of the store would see (a tuple written is a list read) and never
        # aliases an object the caller still owns.
        cached = self._keyed_cache.get(namespace)
        if cached is not None:
            cached[1][key] = json.loads(blob)
        self._bound()

    # --- cache mechanics -----------------------------------------------------
    def _generation(self, namespace: str) -> int:
        """This namespace's cache generation, registering it as resident on first touch.

        Registration is what lets `_bound` see a namespace that has only ever been *read*:
        the counter is the roster the eviction walks, so a namespace that entered a view
        without one would be invisible to the bound and retained forever.
        """
        return self._generations.setdefault(namespace, 0)

    def _bound(self) -> None:
        """Evict the oldest cached namespaces once more than `_NAMESPACE_CACHE_MAX` are held.

        A namespace is evicted from **all four** maps at once — its generation counter, its
        blob view, its keyed view, and its set of per-key-backed entries. That is the whole
        correctness argument: the views are validated against the generation counter, so
        dropping a counter while leaving a view behind would let the counter restart at zero
        and climb back to a number the stale view still carries, at which point the stale view
        reads as current. Evicting the group keeps that impossible, and an eviction costs one
        re-scan of a namespace nobody has touched in a while.
        """
        if len(self._generations) <= _NAMESPACE_CACHE_MAX:
            return
        excess = len(self._generations) - _NAMESPACE_CACHE_MAX
        for namespace in list(self._generations)[:excess]:
            self._generations.pop(namespace, None)
            self._blob_cache.pop(namespace, None)
            self._keyed_cache.pop(namespace, None)
            self._keyed_stored.pop(namespace, None)

    def _unchanged(self, namespace: str, key: str, value: Any) -> bool:
        """True when `(namespace, key)` already stores exactly `value` — so writing is a no-op.

        The learning loops **re-record what they already know on every query**: a query over
        the same source re-measures the same distinct counts, and merges them into the same
        map, and hands the same map back. Serving that write meant a `json.dumps` of the whole
        column map, a backend `put`, and a `json.loads` of the blob back — per column-stat
        table, per query — to arrive at the value already sitting in the parsed view. On the
        default in-process backend (a dict in this very process) the round-trip through JSON
        bytes was the *entire* cost. It was ~48% of a small query's control plane.

        The parsed view is by construction "what a reader of the store would see", so a value
        equal to it is a value already stored, and the write can be dropped. Two guards keep
        that inference honest:

        * the view must be current (its generation), and
        * the key must be backed by its own per-key backend entry (`_keyed_stored`) — an entry
          the view merged up from the *legacy* single-blob shape is readable but not yet
          migrated, and eliding its first write would defer that migration forever.

        Equality is `==` plus a top-level type check, which pins the int/float distinction
        JSON preserves. A nested int-vs-float drift under an equal value is not distinguished
        — it would require a deterministic producer to change a value's type while keeping it
        numerically equal, and every consumer of these learned stats does float arithmetic.
        """
        cached = self._keyed_cache.get(namespace)
        if cached is None or cached[0] != self._generations.get(namespace, 0):
            return False
        if key not in self._keyed_stored.get(namespace, ()):
            return False
        current = cached[1].get(key, _MISSING)
        return type(current) is type(value) and current == value
