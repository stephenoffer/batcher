"""`bt.<subpackage>` must resolve in a fresh interpreter, lazily.

Every docstring and documentation page spells the ML surface `bt.ml.vllm_engine(...)`, and
that raised `AttributeError` until the lazy resolution existed: `io`, `config` and
`governance` happened to be imported transitively by `api`, and `ml` and `graph` did not.
Nothing caught it, because every example using the spelling needs a GPU or a model and so
carries `+SKIP`.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

#: Run in a child interpreter, because the failure only appears when nothing else in the
#: process has already imported the subpackage — which is exactly what a test module that
#: imports `batcher.ml` at the top would hide.
_PROBE = """
import sys
import batcher as bt
assert "batcher.ml" not in sys.modules, "batcher.ml must not be imported eagerly"
assert "batcher.graph" not in sys.modules, "batcher.graph must not be imported eagerly"
{body}
print("ok")
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _PROBE.format(body=body)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("name", ["config", "governance", "graph", "io", "ml"])
def test_every_public_subpackage_resolves_from_a_fresh_import(name):
    result = _run(f"assert bt.{name}.__name__ == 'batcher.{name}'")
    assert result.returncode == 0, result.stderr


def test_the_documented_ml_spelling_works():
    result = _run("assert callable(bt.ml.vllm_engine) and callable(bt.ml.onnx_predictor)")
    assert result.returncode == 0, result.stderr


def test_a_subpackage_is_bound_after_the_first_touch():
    # Bound into the module globals so the second lookup never re-enters `__getattr__`.
    result = _run("bt.ml\nassert 'ml' in vars(bt)")
    assert result.returncode == 0, result.stderr


def test_an_unknown_name_still_gets_the_migration_hint():
    import batcher as bt

    with pytest.raises(AttributeError, match="DataFrame"):
        getattr(bt, "DataFrame")  # noqa: B009 - the attribute access IS what is under test


def test_a_private_probe_fails_plainly_without_importing_anything():
    import batcher as bt

    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(bt, "_not_a_real_thing")  # noqa: B009 - the access IS what is under test
