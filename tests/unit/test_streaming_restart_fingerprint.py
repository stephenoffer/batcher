"""A keyed-state streaming plan still binds its checkpoint to a fingerprint.

`transform_with_state` has no engine IR, so its `content_key` is per-process object identity
and the launcher bound such a plan to *no* fingerprint, which skipped the stateful-restart
check outright. A query restarted with different group keys then restored state keyed by
the old ones: never matched, never expired by a match, silently stale. The fingerprint now
keys what is stable about an opaque node and leaves its Python function out.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.streaming._query import _open_checkpoint
from batcher.plan.streaming.fingerprint import restart_fingerprint

pytestmark = pytest.mark.unit


def _events() -> bt.Dataset:
    return bt.from_pydict({"user": ["a"], "region": ["eu"], "v": [1]})


def _keyed(group_by: list[str], *, columns=("user", "total")):
    # A fresh function object per call, as a restarted process has: nothing about the
    # callback may enter the fingerprint, or no restart would ever be accepted.
    def running_total(key, rows, state):
        return None, state

    return (
        _events()
        .transform_with_state(running_total, group_by=group_by, output_columns=list(columns))
        ._plan
    )


def _checkpoint_with_history(location: str, plan) -> None:
    store = _open_checkpoint(location, plan, stateful=True)
    store.record_offsets(0, {0: {"at": 1}})
    store.commit(0)
    store._owner.release()
    store.close()


def test_a_restart_with_different_group_keys_is_refused(tmp_path):
    location = str(tmp_path / "ckpt")
    _checkpoint_with_history(location, _keyed(["user"]))
    with pytest.raises(PlanError, match="different streaming plan"):
        _open_checkpoint(location, _keyed(["user", "region"]), stateful=True)


def test_a_restart_of_the_same_query_in_a_new_process_is_accepted(tmp_path):
    location = str(tmp_path / "ckpt")
    _checkpoint_with_history(location, _keyed(["user"]))
    store = _open_checkpoint(location, _keyed(["user"]), stateful=True)
    store.close()


def test_the_fingerprint_sees_output_columns_and_operators_above_the_opaque_node():
    base = restart_fingerprint(_keyed(["user"]))
    assert restart_fingerprint(_keyed(["user"])) == base
    assert restart_fingerprint(_keyed(["user"], columns=("user", "n"))) != base
    keyed = bt.Dataset(_keyed(["user"]), _events()._sources)
    over = keyed.filter(bt.col("total") > 1)._plan
    assert restart_fingerprint(over) != base
    assert restart_fingerprint(over) == restart_fingerprint(
        bt.Dataset(_keyed(["user"]), _events()._sources).filter(bt.col("total") > 1)._plan
    )


def test_a_plan_with_ir_keeps_the_fingerprint_it_always_had():
    """Checkpoints written before this change bound `content_key`; they must still match."""
    plan = _events().filter(bt.col("v") > 0)._plan
    assert restart_fingerprint(plan) == plan.content_key()
