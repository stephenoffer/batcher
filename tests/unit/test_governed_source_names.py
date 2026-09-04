"""Every source names the *table* a policy is written about, not the relation it reads.

`identity` and the governance key are two different things, and conflating them was a
total bypass. `identity` names a **relation**: a source pinned to some of a directory's
files, capped at `n_rows`, narrowed to `columns`, or resolved to a Delta version carries a
qualifier for all of those, because one relation's cached statistics must not be handed to
another. A **policy** is written about the table, before anyone has read it and without
knowing how they will slice it.

Reading the governance key off `identity` therefore produced names no policy mentions, so
`catalog.governs()` was False and the read ran ungoverned. Nothing raised. Measured under a
catalog masking `email` and withholding `ssn`:

* ``read.parquet(path)`` returned the mask and no `ssn` -- correct;
* ``read.parquet(path, n_rows=2)`` returned the raw address and the whole `ssn` column;
* ``read.parquet(path, columns=[...])`` did the same;
* every Delta, Iceberg, Hudi and text read was ungoverned outright, because their
  identities carry a version, a catalog, or a read mode in front of the path.

The first two are the ones to keep in mind when reading this file: the bypass is one
ordinary keyword argument typed by an ordinary user, needing no privileged API.

The sweep at the bottom is the part that matters most. It is a property of the *registry*,
so a source added next year is covered without anyone remembering this file exists.
"""

from __future__ import annotations

import contextlib
import re

import pytest

import batcher as bt
from batcher._internal.errors import AccessDeniedError
from batcher.api.security._binding import table_name
from batcher.io.formats.base import SOURCES

pytestmark = pytest.mark.unit

PATH = "/data/tbl"


#: Sources that can be constructed from a bare path. Recomputed rather than listed, so the
#: sweep covers whatever is registered; the floor below is what stops it passing vacuously.
def _constructible() -> list[tuple[str, object]]:
    out = []
    for name in sorted(SOURCES):
        try:
            out.append((name, SOURCES.get(name)(PATH)))
        except Exception:
            continue
    return out


class TestTheRegistrySweep:
    """The property, over every source that exists rather than the ones someone listed."""

    def test_every_source_names_its_table_or_admits_it_has_none(self):
        wrong = [
            (name, table_name(src))
            for name, src in _constructible()
            if table_name(src) not in (PATH, "")
        ]
        assert wrong == [], (
            "these sources are governed under a name no policy would be written about, "
            f"so a policy on the table silently does not apply: {wrong}"
        )

    def test_the_sweep_actually_swept_something(self):
        """Without this the sweep above passes when the registry cannot be enumerated at
        all -- the failure mode `just lint-methodology` exists to catch, and the one a
        registry-driven test falls into first."""
        constructed = _constructible()
        assert len(constructed) >= 30, f"only {len(constructed)} sources constructed"
        names = {name for name, _ in constructed}
        # The formats the bypass actually hit, named so a refactor that drops them from the
        # sweep fails here rather than silently narrowing it.
        assert {"parquet", "csv", "delta", "iceberg", "hudi", "text"} <= names

    @pytest.mark.parametrize(
        "name", ["delta", "delta_cdf", "hudi", "iceberg", "text", "parquet", "csv"]
    )
    def test_a_durable_table_is_named_by_its_path(self, name):
        assert table_name(SOURCES.get(name)(PATH)) == PATH

    @pytest.mark.parametrize("name", ["kafka", "kinesis", "pulsar", "eventhubs", "pubsub"])
    def test_a_broker_is_named_by_its_topic(self, name):
        """A topic is a durable name: an operator writes "mask `email` on `orders`" without
        knowing which cluster a given job points at, and the mask has to hold on both. The
        connection fingerprint belongs in the statistics key, not in the policy key."""
        assert table_name(SOURCES.get(name)("orders")) == "orders"

    def test_a_source_with_no_durable_table_says_so(self):
        """Empty, not a plausible-looking name. `governance.mode` can only refuse or warn
        about an ungovernable read if the source admits to being one; `localhost:9999`
        looks governable and is governed by nothing."""
        assert table_name(SOURCES.get("socket")("localhost")) == ""


class TestASubsetIsStillTheSameTable:
    """The keyword arguments that made an ordinary read ungoverned."""

    @pytest.mark.parametrize(
        "opts",
        [
            {},
            {"n_rows": 2},
            {"columns": ["id", "email"]},
            {"files": ["/data/tbl/part-00000.parquet"]},
        ],
        ids=["plain", "n_rows", "columns", "files"],
    )
    def test_every_narrowing_keeps_the_table_name(self, opts):
        assert table_name(SOURCES.get("parquet")(PATH, **opts)) == PATH

    def test_the_identities_still_differ(self):
        """The positive control. If `identity` had stopped distinguishing these, the test
        above would pass for the wrong reason -- and the statistics cache would be handing
        a capped read the whole table's row count, which is the bug `identity` carries the
        qualifier to prevent."""
        plain = SOURCES.get("parquet")(PATH).identity()
        capped = SOURCES.get("parquet")(PATH, n_rows=2).identity()
        narrowed = SOURCES.get("parquet")(PATH, columns=["id"]).identity()
        assert len({plain, capped, narrowed}) == 3


class TestTheBypassEndToEnd:
    """The user-visible half: the same catalog, the same file, one extra argument."""

    @staticmethod
    def _fixture(tmp_path):
        path = str(tmp_path / "customers.parquet")
        bt.from_pydict(
            {
                "id": [1, 2, 3],
                "email": ["a@x.com", "b@x.com", "c@x.com"],
                "ssn": ["111", "222", "333"],
            }
        ).write(path, format="parquet")
        catalog = (
            bt.SecurityCatalog()
            .grant("analyst", on=path, select=["id", "email"])
            .mask_column(path, "email", lambda c: bt.mask(c))
        )
        return path, catalog, bt.Principal("ana", roles=["analyst"])

    @pytest.mark.parametrize(
        "opts",
        [{}, {"n_rows": 2}, {"columns": ["id", "email", "ssn"]}],
        ids=["plain", "n_rows", "columns"],
    )
    def test_the_mask_holds_however_the_read_is_narrowed(self, tmp_path, opts):
        path, catalog, analyst = self._fixture(tmp_path)
        with bt.security(catalog, analyst):
            rows = bt.read.parquet(path, **opts).to_pydict()
        assert set(rows["email"]) == {"XXXXXXX"}

    @pytest.mark.parametrize(
        "opts",
        [{}, {"n_rows": 2}, {"columns": ["id", "email", "ssn"]}],
        ids=["plain", "n_rows", "columns"],
    )
    def test_the_withheld_column_stays_withheld(self, tmp_path, opts):
        path, catalog, analyst = self._fixture(tmp_path)
        with bt.security(catalog, analyst):
            assert "ssn" not in bt.read.parquet(path, **opts).columns

    def test_an_ungoverned_read_is_unchanged(self, tmp_path):
        """The narrowing options must still narrow. A fix that governed by breaking them
        would pass every assertion above."""
        path, _, _ = self._fixture(tmp_path)
        assert bt.read.parquet(path, n_rows=2).count() == 2
        assert bt.read.parquet(path, columns=["id"]).columns == ["id"]
        assert bt.read.parquet(path).to_pydict()["email"] == ["a@x.com", "b@x.com", "c@x.com"]


class TestAPinnedReadCannotStrandAPolicy:
    """`read.parquet([a, b])` is modelled as their **common parent** plus a file list.

    So its table name is that parent, and when the parent is not itself a governed table
    the policies on the files underneath it were never consulted: reading
    ``[secret.parquet, other.parquet]`` as one relation returned every column of
    `secret.parquet`, including the ones a grant withheld, because their shared directory
    is a name nobody writes a policy about.

    Governing it properly means resolving several policies into one scan, which `enforce`
    is not shaped for -- it takes one table per scan. Refusing is the answer that cannot be
    wrong, and it has to be narrow, or it would refuse the ordinary case of reading some
    files of one governed table. Both halves are asserted here.
    """

    @staticmethod
    def _two_tables(tmp_path):
        secret = str(tmp_path / "secret.parquet")
        other = str(tmp_path / "other.parquet")
        bt.from_pydict({"id": [1], "ssn": ["111"]}).write(secret, format="parquet")
        bt.from_pydict({"id": [2], "ssn": ["222"]}).write(other, format="parquet")
        catalog = bt.SecurityCatalog().grant("analyst", on=secret, select=["id"])
        return secret, other, catalog, bt.Principal("ana", roles=["analyst"])

    def test_reading_a_governed_file_beside_an_ungoverned_one_is_refused(self, tmp_path):
        secret, other, catalog, analyst = self._two_tables(tmp_path)
        with (
            bt.security(catalog, analyst),
            pytest.raises(AccessDeniedError, match="as one relation"),
        ):
            bt.read.parquet([secret, other])

    def test_the_withheld_column_does_not_come_back(self, tmp_path):
        """What the refusal is protecting. Without it this read returned `ssn`."""
        secret, other, catalog, analyst = self._two_tables(tmp_path)
        with bt.security(catalog, analyst):
            assert bt.read.parquet(secret).columns == ["id"]
            with contextlib.suppress(AccessDeniedError):
                assert "ssn" not in bt.read.parquet([secret, other]).columns

    def test_files_of_one_governed_table_are_still_read_and_governed(self, tmp_path):
        """The narrowness half. A policy on the directory must keep covering an explicit
        read of files inside it, rather than being refused as a split policy."""
        table = str(tmp_path / "tbl")
        bt.from_pydict({"id": [1, 2], "ssn": ["1", "2"]}).repartition(num_files=2).write(
            table, format="parquet"
        )
        files = sorted(str(p) for p in (tmp_path / "tbl").glob("*.parquet"))
        catalog = bt.SecurityCatalog().grant("analyst", on=table, select=["id"])
        with bt.security(catalog, bt.Principal("ana", roles=["analyst"])):
            assert bt.read.parquet(table, files=files).columns == ["id"]

    def test_a_hive_nested_file_finds_the_table_above_it(self, tmp_path):
        """The walk is upward for a reason: a policy names a table and a pinned path names
        a file two directories inside it, sharing no exact name with it."""
        table = str(tmp_path / "hive")
        bt.from_pydict({"id": [1, 2], "ssn": ["1", "2"], "dt": ["a", "b"]}).write(
            table, format="parquet", partition_by=["dt"]
        )
        deep = sorted(str(p) for p in (tmp_path / "hive").glob("dt=*/*.parquet"))
        catalog = bt.SecurityCatalog().grant("analyst", on=table, select=["id"])
        with bt.security(catalog, bt.Principal("ana", roles=["analyst"])):
            assert bt.read.parquet(table, files=deep).columns == ["id"]

    def test_ungoverned_paths_are_left_alone(self, tmp_path):
        _secret, other, catalog, analyst = self._two_tables(tmp_path)
        with bt.security(catalog, analyst):
            assert bt.read.parquet([other]).count() == 1

    def test_outside_a_security_block_nothing_changes(self, tmp_path):
        secret, other, _catalog, _analyst = self._two_tables(tmp_path)
        assert bt.read.parquet([secret, other]).count() == 2


#: Every source that needs more than a path to construct, with the least it needs.
#:
#: This table is the correction to a sweep that looked complete and was not. The registry
#: sweep above constructs each source with a bare path and skips whatever raises -- and 23
#: of them raise, which is the entire database, warehouse and document-store family. They
#: were reported as "could not construct" and read as "nothing to check", and every one of
#: them was mis-named: a policy on a Snowflake table, a Mongo collection, an Elasticsearch
#: index, a Cassandra table or a DynamoDB table matched nothing at all.
#:
#: A sweep is only as complete as the constructions it can make, and a skip count is not a
#: pass. Anything added here that cannot be constructed must be given arguments rather than
#: quietly dropped, which is what `test_the_connector_table_covers_the_skipped_sources`
#: enforces.
_CONNECTORS: dict[str, dict] = {
    "adbc": {"driver": "d", "table": "db.t"},
    "bigquery": {"project": "proj", "table": "ds.t"},
    "cassandra": {"contact_points": ["h"], "keyspace": "ks", "table": "t", "partition_key": "k"},
    "clickhouse": {"query": "select 1", "host": "h"},
    "connectorx": {"query": "select 1", "conn_uri": "postgres://h/db"},
    "couchbase": {
        "connstr": "couchbase://h",
        "username": "u",
        "password": "p",
        "database": "d",
        "scope": "s",
        "collection": "c",
    },
    "databricks": {"table": "db.t", "workspace": "w", "token": "tk"},
    "dbapi": {"module": "sqlite3", "table": "t"},
    "dynamodb": {"table": "t"},
    "elasticsearch": {"hosts": ["http://h"], "index": "i"},
    "files_incremental": {"path": "/data/tbl", "format": "parquet"},
    "hbase": {"host": "h", "table": "t"},
    "hdf5": {"path": "/data/tbl", "dataset": "ds"},
    "mongo": {"uri": "mongodb://h", "database": "d", "collection": "c"},
    "neo4j": {"uri": "bolt://h", "username": "u", "password": "p", "cypher": "MATCH (n) RETURN n"},
    "protobuf": {"path": "/data/tbl", "message_cls": object},
    "redis": {"host": "h"},
    "scylla": {"contact_points": ["h"], "keyspace": "ks", "table": "t", "partition_key": "k"},
    "snowflake": {"query": "select 1", "connection_kwargs": {"account": "a"}},
}

#: What each of those must be governed as. A fingerprint in any of these is the defect:
#: `connection_fingerprint` is a sha256 of the connection options, so a name carrying one is
#: a name no operator could type into a policy, and the read was therefore ungoverned.
_EXPECTED = {
    "adbc": "db.t",
    "bigquery": "ds.t",
    "cassandra": "ks.t",
    "clickhouse": "",
    "connectorx": "",
    "couchbase": "d.s.c",
    "databricks": "db.t",
    "dbapi": "t",
    "dynamodb": "default/t",
    "elasticsearch": "i",
    "files_incremental": "/data/tbl",
    "hbase": "h:9090/t",
    "hdf5": "/data/tbl",
    "mongo": "d.c",
    "neo4j": "bolt://h/default",
    "protobuf": "/data/tbl",
    "redis": "h:6379/0",
    "scylla": "ks.t",
    "snowflake": "",
}


def _looks_fingerprinted(name: str) -> bool:
    """Whether `name` contains a `connection_fingerprint`-shaped digest.

    The digests are 12-16 lowercase hex characters. Matching on shape rather than on the
    exact value is deliberate: the value depends on the connection options, so a test that
    pinned it would be asserting the fingerprint function rather than the property, which is
    that a governance name must contain no fingerprint *at all*.
    """
    return bool(re.search(r"\b[0-9a-f]{12,}\b", name))


class TestTheConnectorFamily:
    """The database, warehouse and document stores -- the sources a path cannot construct."""

    @pytest.mark.parametrize("name", sorted(_CONNECTORS))
    def test_each_is_named_by_something_an_operator_could_type(self, name):
        source = SOURCES.get(name)(**_CONNECTORS[name])
        assert table_name(source) == _EXPECTED[name]

    @pytest.mark.parametrize("name", sorted(_CONNECTORS))
    def test_no_name_carries_a_connection_fingerprint(self, name):
        """The property behind the table above, stated so it survives a rename.

        `identity` folds in a sha256 of the connection options, on purpose: the same table
        on staging and production must not share a statistics entry. That digest is exactly
        what an operator cannot reproduce, so a governance name containing one is a policy
        nobody can write.
        """
        governed = table_name(SOURCES.get(name)(**_CONNECTORS[name]))
        assert not _looks_fingerprinted(governed), governed

    def test_the_fingerprint_check_can_actually_fire(self):
        """The positive control. `_looks_fingerprinted` returning False for everything would
        satisfy the test above, and the pre-fix names are exactly what it must reject."""
        assert _looks_fingerprinted("ks.t:8871473da4b6")
        assert _looks_fingerprinted("4eaba920a72e:select 1")
        assert not _looks_fingerprinted("ks.t")
        assert not _looks_fingerprinted("/data/tbl")

    def test_a_query_shaped_read_admits_it_names_no_table(self):
        """A raw SQL string names no table this engine can resolve without parsing it.
        Returning a plausible-looking name instead is what kept `governance.mode` from
        being able to refuse or warn about the read."""
        for name in ("snowflake", "clickhouse", "connectorx"):
            assert table_name(SOURCES.get(name)(**_CONNECTORS[name])) == ""

    def test_the_connector_table_covers_the_skipped_sources(self):
        """The sweep's blind spot, asserted so it cannot reopen.

        Any registered source the path sweep cannot construct must appear in `_CONNECTORS`.
        Without this a source added with required arguments is skipped by one test and
        absent from the other, and is covered by neither while both stay green.
        """
        unconstructible = set()
        for name in SOURCES:
            try:
                SOURCES.get(name)(PATH)
            except Exception:
                unconstructible.add(name)
        # `rate` and `rate_micro_batch` take no locator at all and are covered by the
        # ungovernable-source test above; everything else must be in the table.
        uncovered = (
            unconstructible
            - set(_CONNECTORS)
            - {"rate", "rate_micro_batch", "odbc", "training_shards"}
        )
        assert uncovered == set(), (
            f"these sources are checked by neither sweep: {sorted(uncovered)}"
        )
