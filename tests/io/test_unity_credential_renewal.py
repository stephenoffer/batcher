"""A Databricks lakehouse split renews its vended credentials once they near expiry.

Unity vends storage credentials once, at planning time, and they last minutes to an hour.
A split that waited behind a long queue, or a scan running for hours, used to reach the
object store with credentials that had already lapsed. Each split now carries what vending
needs and re-vends on the worker when its lease is within the renewal margin.

No workspace is contacted: the Databricks SDK is replaced by a fake that vends a *local*
Delta table, stamped with whatever expiry the test asks for, and counts its calls.
"""

from __future__ import annotations

import pickle
import time
from types import SimpleNamespace
from typing import ClassVar

import pytest

import batcher as bt
from batcher.io import credentials
from batcher.io.formats.sql import databricks
from batcher.io.formats.sql.databricks import DatabricksSource, UnityDeltaFileSplit

pytestmark = pytest.mark.io


class _FakeWorkspace:
    """A `WorkspaceClient` stand-in: one table, AWS-shaped credentials, a chosen expiry."""

    calls: ClassVar[list[str]] = []
    location = ""
    expires_in_s = 3600.0

    def __init__(self, host: str, token: str) -> None:
        type(self).calls.append(token)
        self.tables = SimpleNamespace(
            get=lambda full_name: SimpleNamespace(table_id="t-1", storage_location=self.location)
        )
        stamp = int((time.time() + type(self).expires_in_s) * 1000)
        creds = SimpleNamespace(
            url=type(self).location,
            aws_temp_credentials=SimpleNamespace(
                access_key_id=f"AKIA{len(type(self).calls)}",
                secret_access_key="s",
                session_token=None,
            ),
            expiration_time=stamp,
        )
        self.temporary_table_credentials = SimpleNamespace(
            generate_temporary_table_credentials=lambda operation, table_id: creds
        )


@pytest.fixture
def unity(tmp_path, monkeypatch):
    path = str(tmp_path / "orders")
    bt.from_pydict({"id": [1, 2, 3]}).write.delta(path)
    bt.from_pydict({"id": [4, 5]}).write.delta(path, mode="append")
    _FakeWorkspace.calls = []
    _FakeWorkspace.location = path
    _FakeWorkspace.expires_in_s = 3600.0
    monkeypatch.setattr(credentials, "_require_databricks_sdk", lambda: _FakeWorkspace)
    monkeypatch.setenv("DBX_TOKEN", "real-token")
    databricks._LEASES.clear()
    return DatabricksSource(table="main.shop.orders", workspace="https://w", token="env:DBX_TOKEN")


def _ids(splits) -> list[int]:
    return sorted(v for s in splits for b in s.read() for v in b.column("id").to_pylist())


def test_the_token_reference_is_resolved_where_it_vends(unity):
    unity.splits()
    assert _FakeWorkspace.calls == ["real-token"]


def test_every_file_split_can_renew(unity):
    splits = unity.splits()
    assert len(splits) == 2
    assert all(isinstance(s, UnityDeltaFileSplit) for s in splits)
    assert all(s.token == "env:DBX_TOKEN" for s in splits), "the reference, not the secret"


def test_a_fresh_lease_is_read_without_renewing(unity):
    splits = unity.splits()
    assert _ids(splits) == [1, 2, 3, 4, 5]
    assert len(_FakeWorkspace.calls) == 1


def test_an_expiring_lease_is_renewed_once_for_every_split(unity):
    _FakeWorkspace.expires_in_s = 60.0  # inside the renewal margin already
    splits = [pickle.loads(pickle.dumps(s)) for s in unity.splits()]  # as a worker gets them
    databricks._LEASES.clear()  # a worker process starts with no leases of its own
    _FakeWorkspace.expires_in_s = 3600.0
    assert _ids(splits) == [1, 2, 3, 4, 5]
    assert len(_FakeWorkspace.calls) == 2, "one vend to plan, one renewal shared by both files"


def test_the_split_repr_carries_no_token(unity):
    assert "env:DBX_TOKEN" not in repr(unity.splits()[0])
