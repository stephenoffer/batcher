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
