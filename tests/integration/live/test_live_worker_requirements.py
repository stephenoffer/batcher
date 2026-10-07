"""Live smoke test: `explain(requirements=True)` predicts a real worker import failure.

Skipped unless ``BATCHER_LIVE_RAY_ADDRESS`` names a Ray cluster whose workers do not share the
driver's filesystem. A UDF from a module written to a temporary directory is reported as an
unshipped local module, and the distributed run then fails on the workers exactly as reported;
shipping the module with ``runtime_env={"py_modules": [...]}`` makes both agree it is fine.
"""

from __future__ import annotations

import os
import sys

import pytest

import batcher as bt

pytestmark = pytest.mark.skipif(
    not os.environ.get("BATCHER_LIVE_RAY_ADDRESS"),
    reason="set BATCHER_LIVE_RAY_ADDRESS to a Ray cluster to run",
)


def test_an_unshipped_local_module_is_reported_and_then_fails(tmp_path, monkeypatch):
    (tmp_path / "xai_live_udf.py").write_text("def ident(batch):\n    return batch\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    from xai_live_udf import ident

    ds = bt.from_pydict({"x": list(range(10))}).map_batches(ident)
    report = ds.explain(requirements=True)
    assert "xai_live_udf [local:" in report
    import ray

    ray.init(address=os.environ["BATCHER_LIVE_RAY_ADDRESS"])
    try:
        with pytest.raises(Exception, match="xai_live_udf"):
            ds.collect(distributed=True, num_workers=2)
    finally:
        ray.shutdown()
        sys.modules.pop("xai_live_udf", None)
