"""The compatibility probe runs on a real Ray worker and answers with that worker's facts.

The unit suite fabricates every answer; this is the one place the by-value probe crosses a
real Ray task boundary. A local cluster has only the driver's node, which the preflight
deliberately skips, so the probe is pinned to that node directly here. It proves the
mechanism, not a heterogeneous fleet: CI has none.
"""

from __future__ import annotations

import pytest

from batcher.dist.executors.ray_runtime.preflight import report, run

pytestmark = pytest.mark.integration


@pytest.fixture
def ray():
    ray = pytest.importorskip("ray")
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=1, include_dashboard=False, logging_level="ERROR")
    yield ray
    if started:
        ray.shutdown()


def test_the_probe_answers_from_a_real_worker_with_the_drivers_facts(ray):
    here = ray.get_runtime_context().get_node_id()
    node = next(n for n in ray.nodes() if n["NodeID"] == here)
    answers, silent = run.probe_nodes(ray, [node], ("pyarrow", "numpy"))
    assert silent == []
    worker = report.PlatformFacts.from_probe(answers[here])
    driver = report.PlatformFacts.from_probe(report.facts_on_this_node(("pyarrow", "numpy")))
    assert worker == driver
    assert report.compare(driver, {here: ("local", worker)}, ships_driver_build=True) == ()


def test_the_preflight_skips_the_drivers_own_node(ray):
    run.reset_preflight_cache()
    try:
        got = run.ensure_workers_compatible(ray)
        assert got is not None and got.probed == () and got.findings == ()
    finally:
        run.reset_preflight_cache()
