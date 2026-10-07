"""`vector_search(exact=True)` cannot silently become approximate (AP-398).

`import lance` is unavailable in CI, so a fake Lance module records the ``nearest`` dict
the search hands to ``LanceDataset.to_table`` — which is the whole contract: with
``use_index=False`` Lance scans every vector even when an ANN index exists.
"""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
import pytest

from batcher._internal.errors import PlanError
from batcher.io.formats.structured import lance as lance_format
from batcher.ml import vector_search

pytestmark = pytest.mark.unit


class _FakeDataset:
    calls: ClassVar[list[dict]] = []

    def __init__(self, uri: str) -> None:
        self.uri = uri

    def to_table(self, *, columns=None, nearest=None, filter=None):
        _FakeDataset.calls.append(nearest)
        return pa.table({"id": [1], "_distance": [0.0]})


class _FakeLance:
    LanceDataset = _FakeDataset


@pytest.fixture
def fake_lance(monkeypatch):
    _FakeDataset.calls = []
    monkeypatch.setattr(lance_format, "_require_lance", lambda: _FakeLance)
    return _FakeDataset.calls


def test_exact_search_tells_lance_not_to_use_the_index(fake_lance):
    out = vector_search("mem://v.lance", [0.1, 0.2], k=3, exact=True)
    assert out.to_pydict() == {"id": [1], "_distance": [0.0]}
    assert fake_lance[0]["use_index"] is False
    assert fake_lance[0]["k"] == 3


def test_the_default_leaves_index_use_to_lance(fake_lance):
    vector_search("mem://v.lance", [0.1, 0.2], k=3, nprobes=8)
    assert "use_index" not in fake_lance[0]
    assert fake_lance[0]["nprobes"] == 8


@pytest.mark.parametrize("knob", [{"nprobes": 4}, {"refine_factor": 2}])
def test_exact_refuses_the_index_tuning_knobs_rather_than_ignoring_them(fake_lance, knob):
    with pytest.raises(PlanError, match="exact=True"):
        vector_search("mem://v.lance", [0.1, 0.2], exact=True, **knob)
    assert fake_lance == []
