"""Schedule the compatibility probe on the workers and enforce what it finds.

`ensure_workers_compatible` is called where Batcher is about to make a worker import its
package: at the end of `_ensure_ray` (before any operator task is submitted) and in
`probe_options` (before the planning-time hardware probes, whose bodies live in Batcher). It
runs `report.facts_on_this_node` once per node the session has not verified yet, compares the
answers with the driver, and then:

* **when the driver's build is being shipped** (the default `py_modules` upload, either
  job-level or per remote), raises `BackendError` naming each node and field that makes the
  build unloadable there, before any worker has tried to load it;
* **when the workers run their own build** (`distributed.trust_cluster_image`, or an explicit
  `distributed.runtime_env` the user manages), logs the same report as a warning and lets the
  query run. A trusted image is expected to carry a build for its own platform, so an
  architecture or C library that differs from the driver's is not evidence of a problem
  there, and refusing it would break exactly the x86-laptop-submits-to-Arm-cluster setup the
  setting exists for.

The probe is shipped **by value**, with no reference to this package, so evaluating it on an
incompatible worker cannot import the engine the check is protecting. It asks for no CPU.

Cost: one round trip per node per Ray session. Nodes the driver itself runs on are never
probed (they share the driver's interpreter), so a local cluster pays nothing at all. Between
re-reads of the node list the verdict is served from a per-session cache for `_RECHECK_S`, so
the several `_ensure_ray` calls a query makes do not each read the topology. A node that does
not answer within `_PROBE_TIMEOUT_S` is reported as unverified and not asked again this
session: a slow node must not stall every query, and the preflight is a guard against a known
failure, not a health check.
"""

from __future__ import annotations

import builtins
import contextlib
import functools
import pathlib
import re
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field

from batcher._internal.errors import BackendError
from batcher._internal.logging import get_logger, note_suppressed
from batcher._internal.paths import package_dir
from batcher.config import active_config

from ..scheduling import job_ships_batcher, ray_session_key
from .report import (
    ENGINE_DIST,
    REQUIRED_ON_WORKER,
    CompatibilityReport,
    Finding,
    PlatformFacts,
    compare,
    engine_glibc_floor,
    facts_on_this_node,
)

__all__ = [
    "ensure_workers_compatible",
    "last_compatibility_report",
    "probe_nodes",
    "reset_preflight_cache",
    "ships_driver_build",
]

#: How long a node may take to answer. Generous because the first task on a node unpacks the
#: job's `runtime_env` before its body runs, and that cost is paid once whoever pays it.
_PROBE_TIMEOUT_S = 60.0

#: How long a session's verdict is reused before the node list is read again to catch nodes
#: an autoscaler added.
_RECHECK_S = 30.0

#: Where the fix is described.
_DOC = "getting-started/install/clusters-and-servers"


@dataclass
class _SessionState:
    """What one Ray session has learned. Replaced wholesale when the session changes."""

    session: str | None
    seen: set[str] = field(default_factory=set)
    answered: dict[str, tuple[str, PlatformFacts]] = field(default_factory=dict)
    unanswered: list[str] = field(default_factory=list)
    checked_at: float = float("-inf")
    warned: set[Finding] = field(default_factory=set)
    warned_unanswered: int = 0


_STATE: _SessionState | None = None
_LAST: CompatibilityReport | None = None
_LOCK = threading.Lock()


def reset_preflight_cache() -> None:
    """Forget every verdict, so the next distributed call probes the workers again.

    For a test substituting a cluster, and for an operator who has just replaced the nodes a
    previous run refused.
    """
    global _STATE, _LAST
    with _LOCK:
        _STATE = None
        _LAST = None


def last_compatibility_report() -> CompatibilityReport | None:
    """The most recent report this process produced, or `None` before any preflight ran."""
    return _LAST


def ships_driver_build() -> bool:
    """Whether the workers will load the driver's own build rather than one of their own.

    True on both shipping paths: Batcher initialized Ray and set the job's `py_modules`, or
    another process did and Batcher attaches the package to each remote. False when the
    cluster image is trusted, or when the user passed an explicit `distributed.runtime_env`
    and therefore owns what the workers run.
    """
    dc = active_config().distributed
    if dc.trust_cluster_image:
        return False
    if not job_ships_batcher():
        return True  # `worker_runtime_env` ships the package on every remote
    return dc.runtime_env is None


@functools.cache
def _engine_floor() -> tuple[int, ...] | None:
    """The engine's glibc floor, read once per process from the installed extension."""
    so = sorted(pathlib.Path(package_dir()).glob("_native*.so"))
    return engine_glibc_floor(so[0]) if so else None


@functools.cache
def _probe_packages() -> tuple[str, ...]:
    """The distributions a worker needs installed: Batcher's unconditional requirements.

    Read from Batcher's own installed metadata so the list cannot drift from `pyproject.toml`.
    A source checkout run from `PYTHONPATH` has no metadata, and still asks for the two whose
    absence breaks `import batcher` outright.
    """
    from importlib import metadata

    try:
        reqs = metadata.requires(ENGINE_DIST) or []
    except metadata.PackageNotFoundError:
        reqs = []
    names = [re.split(r"[\s<>=!~\[;(]", r, maxsplit=1)[0] for r in reqs if ";" not in r]
    return (*dict.fromkeys([*REQUIRED_ON_WORKER, *names]), ENGINE_DIST)


@functools.cache
def _driver_facts() -> PlatformFacts:
    """The driver's own facts, gathered by the same body the workers run."""
    return PlatformFacts.from_probe(facts_on_this_node(_probe_packages()))


def _by_value(fn: Callable) -> Callable:
    """A copy of `fn` that pickles by value and carries no reference to Batcher.

    A module-level function is pickled *by reference*, which makes the worker import its
    module to find it: for a function in this package that is `import batcher`, and so the
    engine. Rebinding it to `__main__` with bare globals is what cloudpickle ships by value.
    """
    clone = types.FunctionType(fn.__code__, {"__builtins__": builtins}, fn.__name__)
    clone.__module__ = "__main__"
    clone.__qualname__ = fn.__name__
    clone.__doc__ = fn.__doc__
    return clone


def _node_label(node: dict) -> str:
    node_id = str(node.get("NodeID", ""))
    address = node.get("NodeManagerAddress") or node.get("NodeManagerHostname") or "?"
    return f"node {node_id[:12]} ({address})"


def probe_nodes(ray, nodes: list[dict], packages: tuple[str, ...]) -> tuple[dict, list[str]]:
    """Run the probe pinned to each node; return answers by node id and the silent nodes.

    Args:
        ray: The imported `ray` module.
        nodes: Ray node records to probe.
        packages: Distribution names each node reports versions for.

    Returns:
        `({node_id: raw facts}, [node ids that did not answer])`.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    remote = ray.remote(num_cpus=0, max_retries=0)(_by_value(facts_on_this_node))
    refs = {
        remote.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False)
        ).remote(packages): n["NodeID"]
        for n in nodes
    }
    ready, pending = ray.wait(list(refs), num_returns=len(refs), timeout=_PROBE_TIMEOUT_S)
    answers: dict[str, dict] = {}
    silent = [refs[r] for r in pending]
    for ref in ready:
        try:
            answers[refs[ref]] = ray.get(ref)
        except Exception as exc:  # the task itself failed: unverified, not incompatible
            note_suppressed("dist", "run the worker compatibility probe", exc)
            silent.append(refs[ref])
    for ref in pending:
        with contextlib.suppress(Exception):
            ray.cancel(ref, force=True)
    return answers, silent


def _probe_new_nodes(ray, state: _SessionState) -> None:
    """Probe the alive remote nodes `state` has not seen, recording what each answered."""
    here = ray.get_runtime_context().get_node_id()
    alive = [n for n in ray.nodes() if n.get("Alive", True) and n.get("NodeID")]
    fresh = [n for n in alive if n["NodeID"] != here and n["NodeID"] not in state.seen]
    state.checked_at = time.monotonic()
    if not fresh:
        return
    labels = {n["NodeID"]: _node_label(n) for n in fresh}
    answers, silent = probe_nodes(ray, fresh, _probe_packages())
    for nid, raw in answers.items():
        state.answered[nid] = (labels[nid], PlatformFacts.from_probe(raw))
    state.seen.update(labels)
    state.unanswered.extend(labels[nid] for nid in silent)


def _report(state: _SessionState, ships: bool) -> CompatibilityReport:
    """Judge what the session has gathered under the contract that applies *now*.

    Compared per call rather than stored as findings, so a change of `trust_cluster_image`
    between queries is honoured without re-probing anyone.
    """
    driver = _driver_facts()
    return CompatibilityReport(
        driver=driver,
        ships_driver_build=ships,
        findings=compare(
            driver, state.answered, ships_driver_build=ships, glibc_floor=_engine_floor()
        ),
        probed=tuple(label for label, _ in state.answered.values()),
        unanswered=tuple(state.unanswered),
    )


def ensure_workers_compatible(ray) -> CompatibilityReport | None:
    """Verify every remote node can load what the driver is about to send it.

    Best-effort in every direction but one: failing to *ask* (Ray down, a substituted `ray`
    module, an unreadable topology) never stops a query. Only a positive answer that a node
    cannot load the shipped build does, by raising.

    Args:
        ray: The imported `ray` module (passed so a test can substitute it).

    Returns:
        The report, or `None` when the preflight could not run.

    Raises:
        BackendError: The driver's build is being shipped and at least one node differs from
            the driver on a field that makes it unloadable there.
    """
    global _STATE, _LAST
    with _LOCK:
        try:
            if not ray.is_initialized():
                return None
            ships = ships_driver_build()
            session = ray_session_key()
            if _STATE is None or _STATE.session != session:
                _STATE = _SessionState(session)
            state = _STATE
            if time.monotonic() - state.checked_at >= _RECHECK_S:
                _probe_new_nodes(ray, state)
            report = _report(state, ships)
        except Exception as exc:
            note_suppressed("dist", "check worker binary compatibility", exc)
            return None
        _LAST = report
        _act_on(report, state)
        return report


def _act_on(report: CompatibilityReport, state: _SessionState) -> None:
    """Raise on a blocking finding; warn once per session about everything else."""
    if report.ships_driver_build and report.blocking:
        nodes = sorted({f.node for f in report.blocking})
        d = report.driver
        raise BackendError(
            f"{len(nodes)} Ray worker node(s) cannot load the Batcher build this driver ships "
            f"(driver: {d.os}/{d.machine}, {d.libc or 'no glibc'} {d.libc_version}, "
            f"Python {d.python}). Refused before any worker imported it:\n{report.render()}\n",
            hint=(
                "Run the driver on a machine matching the workers (or submit the job with "
                "`ray job submit` so it runs on the head node), or give every node an image "
                "with its own Batcher build and set distributed.trust_cluster_image=True."
            ),
            doc=_DOC,
        )
    new = [f for f in report.findings if f not in state.warned]
    if new or len(report.unanswered) > state.warned_unanswered:
        state.warned.update(report.findings)
        state.warned_unanswered = len(report.unanswered)
        get_logger("dist").warning(
            "Ray worker preflight: differences from the driver or unverified nodes (%s, not "
            "refused):\n%s",
            "advisory" if report.ships_driver_build else "workers run their own build",
            report.render(),
        )
