"""A local relational query must not drag a heavyweight optional dependency into the process.

Both probes these pin were real: planning any query built a `HardwareProfile`, whose GPU
inventory fell through to `import torch` (~1.4 s) on a host with no device, and every
terminal op's ``distributed="auto"`` routing ran `import ray` (~0.44 s) to ask whether Ray
was initialized. Each cost was paid once per process, on the first query, to compute an
answer that was already determined -- and neither is visible to a timing test, because both
are amortized away by the second query.

So this asserts the *import*, not the clock: run a query in a fresh interpreter and check
`sys.modules`. That fails loudly the moment either probe is re-armed, and it cannot flake on
a busy machine the way a threshold on the first query's wall time would.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Build a table, plan a projection and a grouped aggregate over it, then run one terminal.
# The aggregate is what makes Kyber run a real optimize (which constructs the HardwareProfile
# that probes for GPUs); the terminal is what resolves `distributed="auto"` and reads the
# cluster's GPU count. Parameterized over the terminals because the two Ray probes sit on
# different paths -- `to_pydict` reached one that `collect` does not.
_PROGRAM = """
import sys
import batcher as bt

ds = bt.from_pydict({{"k": ["a", "b", "a"], "v": [1, 2, 3]}})
out = ds.select(k="k", doubled=bt.col("v") * 2).group_by("k").agg(bt.col("doubled").sum())
{terminal}

print("WATCHED:" + ",".join(name for name in ("torch", "ray") if name in sys.modules))
"""

_TERMINALS = {
    "collect": "out.collect()",
    "to_pydict": "out.to_pydict()",
    "count": "out.count()",
    "iter_batches": "list(out.iter_batches())",
}


def _modules_imported_by_a_local_query(terminal: str) -> set[str]:
    """The watched modules a fresh interpreter ends up with after one local query."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROGRAM.format(terminal=_TERMINALS[terminal])],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"query failed in the subprocess:\n{proc.stderr}"
    marked = [line for line in proc.stdout.splitlines() if line.startswith("WATCHED:")]
    assert marked, f"probe program printed no verdict:\n{proc.stdout}\n{proc.stderr}"
    return {name for name in marked[-1].removeprefix("WATCHED:").split(",") if name}


@pytest.mark.unit
@pytest.mark.parametrize("terminal", sorted(_TERMINALS))
def test_a_local_query_imports_neither_torch_nor_ray(terminal: str) -> None:
    """Planning and running a local query pulls in no GPU framework and no cluster runtime."""
    assert _modules_imported_by_a_local_query(terminal) == set()


_COLD_RESOLVE = """
import sys

from batcher._exports import EXPORTS

first_name_per_module = {}
for name, target in EXPORTS.items():
    first_name_per_module.setdefault(target.partition(":")[0], name)

failures = []
for module_path, name in first_name_per_module.items():
    for cached in [m for m in sys.modules if m == "batcher" or m.startswith("batcher.")]:
        del sys.modules[cached]
    try:
        import batcher

        getattr(batcher, name)
    except Exception as exc:
        failures.append(f"{name} (-> {module_path}): {type(exc).__name__}: {exc}")

print("CHECKED:" + str(len(first_name_per_module)))
print("FAILED:" + "|".join(failures))
"""


def test_every_lazy_export_resolves_when_it_is_the_first_name_touched() -> None:
    """Each public name must import when *it* is the first thing a process asks for.

    `batcher/__init__.py` resolves its exports lazily, so the first attribute a program
    touches decides the import order for everything under it. A module-scope import that
    closes a cycle is then invisible to every test that happens to touch some other name
    first -- and `bt.GroupBy` did exactly that: `import batcher as bt; bt.GroupBy` raised
    ``ImportError: cannot import name 'GroupBy' from partially initialized module``, while
    `bt.Dataset` followed by `bt.GroupBy` worked, because `Dataset` had already pulled the
    cycle's other half in. Every suite in this repo touches `Dataset` first.

    **This runs in a subprocess, and must.** The check works by purging ``batcher*`` from
    `sys.modules` between names, which re-executes the target module -- and that is exactly
    what makes it unsafe in-process: every *other* test module that already did
    ``import batcher as bt`` keeps its binding to the pre-purge module object, so its
    classes stop matching the freshly imported ones and `isinstance` quietly returns False.
    Run in-process this passed itself and then broke eight tests in two other files
    (`col("a").meta.is_column()` answering False, `root_names()` answering `[]`,
    "projection rewrite: unhandled node Scan"). A subprocess throws the corruption away
    with the interpreter, which is why the probes above are spawned the same way.

    One name per *target module* is enough and is what keeps this at a few seconds: the
    cycle being guarded against is a property of the module's import, not of which of its
    names was asked for.
    """
    result = subprocess.run(
        [sys.executable, "-c", _COLD_RESOLVE], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checked = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("CHECKED:")), "CHECKED:0"
    )
    assert int(checked.removeprefix("CHECKED:")) >= 80, f"denominator collapsed: {checked}"
    failed = next((ln for ln in result.stdout.splitlines() if ln.startswith("FAILED:")), "FAILED:")
    names = [n for n in failed.removeprefix("FAILED:").split("|") if n]
    assert not names, "public names that fail to resolve from a cold import:\n" + "\n".join(names)
