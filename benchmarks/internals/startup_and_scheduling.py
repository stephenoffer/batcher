#!/usr/bin/env python3
"""Driver-side fixed costs: process startup, and the scheduling phase against fleet size.

Neither of these is a query. They are what a process pays *before* it can do anything, and
they are the two costs that decide whether the engine scales to the fleets it targets —
a hundred thousand nodes and millions of worker processes, where a fixed cost per process
and a per-query cost per node are both multiplied by numbers the query never sees.

Two measurements, run independently::

    python benchmarks/internals/startup_and_scheduling.py startup
    python benchmarks/internals/startup_and_scheduling.py scheduling

**startup** times `import batcher` in a fresh interpreter, and reports what it pulled in.
The surface resolves lazily (`batcher._lazy`), so the figure to watch is the module count
as much as the milliseconds: a module-scope import creeping back into a façade shows up
there first and as a duration only on a cold page cache.

**scheduling** times the placement phase against a *synthetic* Ray whose `nodes()` returns
a pre-built list. That is deliberate and it is also the measurement's main limitation: it
prices the driver's own Python work and charges nothing for the GCS round trip or the
deserialization of a hundred thousand node records, which on a real cluster of that size
dominate. Read it as "what the control plane does with the node list once it has it",
which is the part this repository controls, and not as a whole-query projection.

Compare a change against the same script on the same machine — the absolute numbers move
with core count and Python version, the ratios do not. To measure a branch against `main`,
run it under `PYTHONPATH` pointing at a `git archive` sandbox of the baseline rather than
switching the working tree, which other sessions may be using.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envinfo import require_release_build

#: Fleet sizes to sweep. The top of the range is the engine's stated scaling target; the
#: bottom is an ordinary cluster, and is there so a change that helps only at scale is
#: visible as such rather than being reported as an unqualified win.
FLEET_SIZES = (100, 1_000, 10_000, 50_000, 100_000)

#: Calls the placement phase makes per distributed query, from the call sites themselves:
#: five node-class questions (the pack decision, the node-class selector, `placeable_workers`,
#: the zone selector, the pending-demand diagnosis), the fleet shape the optimizer plans
#: against, and one locality question per pipeline breaker.
CLASS_QUESTIONS = 5
BREAKERS = 10


def _fake_ray(node_count: int, zones: int = 3, gpu_every: int = 4) -> None:
    """Install a stub `ray` describing a `node_count`-node heterogeneous fleet."""
    nodes = [
        {
            "NodeID": f"{i:032x}",
            "Alive": True,
            "Resources": {
                "CPU": 64.0,
                "GPU": 8.0 if i % gpu_every == 0 else 0.0,
                "memory": 512e9,
                **({"node:__internal_head__": 1.0} if i == 0 else {}),
            },
            "Labels": {
                "ray.io/availability-zone": f"z{i % zones}",
                "ray.io/accelerator-type": "A100" if i % gpu_every == 0 else "",
            },
        }
        for i in range(node_count)
    ]
    ray = types.ModuleType("ray")
    ray.nodes = lambda: nodes
    ray.is_initialized = lambda: True
    ray.cluster_resources = lambda: {
        "CPU": 64.0 * node_count,
        "GPU": 8.0 * (node_count // gpu_every),
        "memory": 512e9 * node_count,
    }
    ray.available_resources = ray.cluster_resources
    private = types.ModuleType("ray._private")
    state = types.ModuleType("ray._private.state")
    state.available_resources_per_node = lambda: {n["NodeID"]: {"CPU": 60.0} for n in nodes}
    private.state = state
    ray._private = private
    for name, module in (
        ("ray", ray),
        ("ray._private", private),
        ("ray._private.state", state),
    ):
        sys.modules[name] = module


def _scheduling_phase(node_count: int) -> tuple[float, int]:
    """Milliseconds one query's placement phase costs, and the classes the fleet reduced to."""
    for name in list(sys.modules):
        if name.startswith(("batcher", "ray")):
            del sys.modules[name]
    _fake_ray(node_count)

    from batcher.dist.executors.ray_runtime import capacity, scaling
    from batcher.dist.executors.ray_runtime.fabric import shape

    # Falls back to the per-node view on a baseline that predates the census, so this script
    # can be pointed at a `git archive` sandbox of `main` and produce a comparable figure.
    ask_classes = getattr(scaling, "node_class_census", scaling.node_classes)

    scaling._reset_topology_cache()
    gc.collect()
    started = time.perf_counter()
    with scaling.topology_scope():
        for _ in range(CLASS_QUESTIONS):
            ask_classes()
        scaling.alive_node_count()
        scaling.cluster_node_count()
        capacity.placeable_workers(8.0, 0.0, memory_bytes=int(8e9))
        capacity.free_cpus_by_node()
        fleet = shape.cluster_shape()
        for _ in range(BREAKERS):
            fleet.locality_shares(max(1, node_count // 10))
            hash(fleet)  # the shape reaches the plan cache key
        classes = len(fleet.nodes)
    scaling._reset_topology_cache()
    return (time.perf_counter() - started) * 1000, classes


def run_scheduling() -> None:
    """Sweep the fleet sizes and print the phase cost against each."""
    print(f"{'nodes':>9}  {'placement phase':>16}  {'per node':>10}  {'classes':>8}")
    for size in FLEET_SIZES:
        elapsed, classes = _scheduling_phase(size)
        print(f"{size:>9}  {elapsed:>13.1f} ms  {elapsed * 1000 / size:>7.2f} us  {classes:>8}")


def run_startup(repeats: int = 5) -> None:
    """Time `import batcher` in a fresh interpreter, and report what it loaded."""
    probe = (
        "import time, sys\n"
        "start = time.perf_counter()\n"
        "import batcher\n"
        "elapsed = (time.perf_counter() - start) * 1000\n"
        "loaded = [m for m in sys.modules if m.startswith('batcher')]\n"
        "heavy = sorted({'pyarrow', 'numpy', 'pandas'} & set(sys.modules))\n"
        'print(f\'{elapsed:.2f} {len(loaded)} {",".join(heavy) or "-"}\')\n'
    )
    print(f"{'run':>4}  {'import batcher':>15}  {'modules':>8}  third-party")
    for attempt in range(1, repeats + 1):
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        elapsed, modules, heavy = out.stdout.split()
        print(f"{attempt:>4}  {float(elapsed):>12.2f} ms  {modules:>8}  {heavy}")


def main() -> None:
    """Parse the subcommand and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("startup", "scheduling", "both"), default="both")
    parser.add_argument(
        "--allow-debug-build",
        action="store_true",
        help="time a dev-profile engine anyway (see benchmarks/envinfo.py)",
    )
    args = parser.parse_args()
    # Neither measurement here runs an operator — the scheduling sweep never leaves Python and
    # the lazy surface does not even open the extension — so the build profile does not move
    # these numbers the way it moves a query benchmark. It is still asserted rather than
    # reasoned about: a reader comparing this table against a query benchmark's has no way to
    # know which of the two was profile-checked, and "this one does not need it" is exactly
    # the argument every unguarded timing in this repository was shipped with.
    require_release_build(allow_debug=args.allow_debug_build)
    if args.what in ("startup", "both"):
        run_startup()
    if args.what in ("scheduling", "both"):
        if args.what == "both":
            print()
        run_scheduling()


if __name__ == "__main__":
    main()
