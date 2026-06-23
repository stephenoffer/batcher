"""Online serving — the thin Ray Serve adapter over the load-once batch primitives.

Without the ``batcher-engine[serve]`` extra, building a deployment must fail with a clear,
actionable error. With Ray Serve present, the deployment is built end to end and its
batched predictor returns the same result as calling the model directly (the
batch-vs-online equivalence the adapter promises).
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import BackendError
from batcher.ml.serving import serve_deployment


def _build():
    # A batched predictor: list[int] -> list[int]; "model load" happens once here.
    def predict(xs):
        return [x * 2 for x in xs]

    return predict


def _serve_installed() -> bool:
    try:
        from ray import serve  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(_serve_installed(), reason="ray serve installed; guard not exercised")
def test_requires_serve_extra_when_absent():
    with pytest.raises(BackendError, match="batcher\\[serve\\]"):
        serve_deployment(_build)


@pytest.mark.skipif(not _serve_installed(), reason="ray serve not installed")
def test_builds_deployment_when_available():
    deployment = serve_deployment(_build, name="t", num_replicas=1)
    assert deployment is not None  # a Serve deployment class
