"""Every zero-copy export routes the way its terminal operation does.

The Arrow PyCapsule interface and the DataFrame interchange protocol take no execution
arguments, so each has to *choose* a routing policy. Two defaults were available and they
disagree: `collect()` resolves ``distributed="auto"``, `iter_batches()` defaults to
``False``. Choosing the second made ``pl.DataFrame(ds)`` run single-node on a multi-node
cluster while ``pl.DataFrame(ds.collect())`` distributed.

That failure is invisible without a cluster — the answer is identical either way, only
slower — so it is asserted here against the *call*, which needs no cluster to check.
"""

from __future__ import annotations

import inspect

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.dataset.frame import Dataset

pytestmark = pytest.mark.unit


def _capture(monkeypatch, method: str) -> list[dict]:
    """Record the keyword arguments `method` is called with, and keep it working."""
    calls: list[dict] = []
    original = getattr(Dataset, method)

    def spy(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, method, spy)
    return calls


def test_the_pycapsule_export_resolves_distribution(monkeypatch):
    """`pa.table(ds)` must ask for the same routing `collect()` would."""
    calls = _capture(monkeypatch, "iter_batches")
    table = pa.table(bt.from_pydict({"x": [1, 2, 3]}).filter(bt.col("x") > 1))

    assert table.to_pydict() == {"x": [2, 3]}
    assert calls, "the export did not go through iter_batches at all"
    assert calls[0].get("distributed") == "auto", calls[0]


def test_the_export_matches_the_terminal_op_it_stands_in_for(monkeypatch):
    """The export's routing tracks `collect()`'s declared default rather than restating it.

    If someone changes `collect()`'s default, this fails and names the export that drifted,
    which is the whole reason to assert the two against each other instead of against the
    literal ``"auto"`` twice.
    """
    collect_default = inspect.signature(Dataset.collect).parameters["distributed"].default
    calls = _capture(monkeypatch, "iter_batches")
    pa.table(bt.from_pydict({"x": [1]}))

    assert calls[0].get("distributed") == collect_default


def test_the_interchange_export_goes_through_collect(monkeypatch):
    """`__dataframe__` materializes through `collect()`, so it inherits its routing."""
    calls = _capture(monkeypatch, "collect")
    obj = bt.from_pydict({"x": [1, 2]}).__dataframe__()

    assert obj is not None
    assert calls, "__dataframe__ did not go through collect"


def test_iter_batches_itself_is_unchanged():
    """The explicit streaming API keeps its documented single-node default.

    The export was changed, not the method behind it: a caller who wrote
    `iter_batches()` asked for a specific thing and must keep getting it.
    """
    assert inspect.signature(Dataset.iter_batches).parameters["distributed"].default is False
