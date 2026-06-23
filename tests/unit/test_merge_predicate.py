"""Unit coverage for the Delta MERGE match-predicate builder.

The end-to-end upsert needs `deltalake` (an integration concern); the predicate
construction is pure and is tested here against the source/target alias contract
the delta sink uses.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PlanError
from batcher.api.merge import merge_predicate_for

pytestmark = pytest.mark.unit


def test_single_key():
    assert merge_predicate_for("id") == "target.id = source.id"


def test_single_key_as_list():
    assert merge_predicate_for(["id"]) == "target.id = source.id"


def test_composite_key():
    assert merge_predicate_for(["id", "day"]) == "target.id = source.id AND target.day = source.day"


def test_empty_keys_raise():
    with pytest.raises(PlanError, match="at least one key"):
        merge_predicate_for([])
