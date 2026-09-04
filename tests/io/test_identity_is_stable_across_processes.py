"""Every source's `identity()` must be the same string in a fresh interpreter.

`identity()` is the key learned statistics are stored under, so it is the hinge the
cross-run half of the engine's moat turns on: measure a query, write the numbers under this
key, look them up next run. An identity built on Python's `hash()` is stable within a
process and different in the next one, because `hash()` is salted per interpreter. Nothing
raises. Statistics are written on every run and read on none, and the feedback loop appears
to work while never once improving a plan.

`tests/io/test_nosql_audit_b.py` pins this for `RedisSource`. That is one connector out of
sixty-five, chosen because it was the one under audit that day, and the other sixty-four
could each reintroduce the defect without a test noticing. This is the same property as a
sweep over the registry, so a source added next year is covered without anyone remembering
the file exists.

The comparison carries its own positive control. Every run also emits `__control__`, a value
built from `hash()` on purpose. If the seeds do not disagree about *that*, the subprocesses
did not actually get different hash seeds and the whole comparison is vacuous -- which is a
thing that happens, because `PYTHONHASHSEED` is easy to lose when an environment is rebuilt
for a subprocess.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.io

_REPO = Path(__file__).resolve().parents[2]

#: Reuses the connector table `test_governed_source_names.py` already maintains, rather than
#: keeping a second copy that would drift. That table is the least each source needs to
#: construct, and it exists because a sweep that skips whatever raises silently covered none
#: of the database, warehouse and document-store family -- thirteen connectors.
_SCRIPT = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("gsn", {table!r})
gsn = importlib.util.module_from_spec(spec); spec.loader.exec_module(gsn)
from batcher.io.formats.base import SOURCES

out = {{"__control__": str(hash("a stable string"))}}
for name in sorted(SOURCES):
    source = None
    try:
        source = SOURCES.get(name)("/data/tbl")
    except Exception:
        kwargs = gsn._CONNECTORS.get(name)
        if kwargs:
            try:
                source = SOURCES.get(name)(**kwargs)
            except Exception:
                source = None
    if source is None:
        continue
    try:
        out[name] = str(source.identity())
    except Exception:
        continue
print(json.dumps(out))
"""


def _child_env(seed: str) -> dict[str, str]:
    """A deliberately minimal environment, plus whatever the dynamic loader needs.

    The point of building the environment by hand is `PYTHONHASHSEED`: the child must be a
    genuinely different interpreter, not this one's inherited settings. It is not about the
    loader, and dropping `LD_LIBRARY_PATH` on a box that needs one only makes the child fail
    to `import` at all -- which reports as "seed 0 failed", three tests down, and says nothing
    about identity stability.
    """
    env = {"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def _identities(seed: str) -> dict[str, str]:
    table = _REPO / "tests" / "unit" / "test_governed_source_names.py"
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(table=str(table))],
        capture_output=True,
        text=True,
        cwd=_REPO,
        env=_child_env(seed),
    )
    assert result.returncode == 0, f"seed {seed} failed:\n{result.stderr[-2000:]}"
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs() -> list[dict[str, str]]:
    return [_identities(seed) for seed in ("0", "1", "12345")]


def test_the_hash_seeds_really_differ(runs):
    """The control. Without it every assertion below passes when the subprocesses all ran
    with the same seed, which proves nothing about `hash()` at all."""
    controls = {run["__control__"] for run in runs}
    assert len(controls) > 1, (
        "all three interpreters produced the same hash() value, so PYTHONHASHSEED did not "
        "take effect and the stability comparison below is vacuous"
    )


def test_the_sweep_covered_the_registry(runs):
    """A registry-driven sweep that constructs nothing passes while checking nothing."""
    shared = set.intersection(*(set(run) for run in runs)) - {"__control__"}
    assert len(shared) >= 40, f"only {len(shared)} sources produced an identity"
    # The formats whose identity carries a connection fingerprint -- the ones where a
    # `hash()` would be easiest to reach for -- named so a refactor that drops them from the
    # sweep fails here rather than narrowing it silently.
    assert {"parquet", "csv", "delta", "iceberg", "kafka", "mongo", "redis"} <= shared


def test_every_identity_is_the_same_in_a_fresh_interpreter(runs):
    """The property itself."""
    shared = set.intersection(*(set(run) for run in runs)) - {"__control__"}
    unstable = {
        name: [run[name] for run in runs]
        for name in sorted(shared)
        if len({run[name] for run in runs}) > 1
    }
    assert unstable == {}, (
        "these identities change between interpreters, so every statistic written under "
        f"them is written once and never read again: {unstable}"
    )
