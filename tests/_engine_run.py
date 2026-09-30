"""Run a JSON IR plan on the engine under an exact memory envelope, in a fresh process.

A test that wants a specific operator to go out of core has to control the envelope that
operator is admitted against, and in-process it cannot. The data plane's memory pool is
process-wide and its limit only ever grows (`bc_py::process`), so admission in a test process
is decided by the largest envelope any earlier test used. A spill assertion made in-process
then passes alone and fails after a neighbour that ran under a wide budget -- it reports the
suite's history rather than the engine's behaviour. `tests/unit/test_memory_budget_error.py`
runs in a subprocess for the same reason.

The child returns the result table and, from `execute_plan_metered`, which operators reported
that they spilled -- the positive control a spill test needs, since a result from an in-memory
run passes any comparison equally well.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

_CHILD = textwrap.dedent(
    """
    import json, sys
    import pyarrow as pa, pyarrow.ipc as ipc
    from batcher._internal.native import engine
    from batcher._internal.errors import MemoryBudgetExceededError
    from batcher.config import Config, MemoryConfig

    plan, budget, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    sources = [ipc.open_file(p).read_all().to_batches() for p in sys.argv[4:]]
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=budget))
    try:
        batches, metrics = engine().execute_plan_metered(plan, sources, cfg.engine_config_json())
    except MemoryBudgetExceededError as exc:
        print(json.dumps({"raised": str(exc)}))
        sys.exit(0)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(0)
    table = pa.Table.from_batches(batches) if batches else None
    if table is not None:
        with ipc.new_file(out_path, table.schema) as w:
            w.write_table(table)
    ops = json.loads(metrics)["ops"]
    spilled = {}
    for o in ops:
        spilled[o["kind"]] = spilled.get(o["kind"], False) or o["spilled"]
    print(json.dumps({"spilled": spilled, "empty": table is None}))
    """
)


def run_plan(tmp_path: Path, plan: dict, sources: list[pa.Table], budget: int) -> dict:
    """Execute `plan` over `sources` under `budget` bytes in a fresh interpreter.

    Args:
        tmp_path: A directory for the input and output IPC files.
        plan: The JSON IR plan; `scan` source ids index `sources`.
        sources: The input tables.
        budget: The `memory.max_memory_bytes` the plan runs under.

    Returns:
        ``{"raised": message}`` for a `MemoryBudgetExceededError`, ``{"error": message}`` for
        any other engine error, else ``{"spilled": {operator kind: bool}, "table": pa.Table |
        None}``.
    """
    run = len(list(tmp_path.iterdir()))
    paths = []
    for i, t in enumerate(sources):
        p = tmp_path / f"run{run}-src{i}.arrow"
        with ipc.new_file(p, t.schema) as w:
            w.write_table(t)
        paths.append(str(p))
    out = tmp_path / f"run{run}-out.arrow"
    done = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(plan), str(budget), str(out), *paths],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, f"child failed:\n{done.stderr}"
    report = json.loads(done.stdout.strip().splitlines()[-1])
    if "spilled" in report:
        report["table"] = None if report["empty"] else ipc.open_file(out).read_all()
    return report
