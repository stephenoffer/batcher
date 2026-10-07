"""The vendor profiles, the published capability matrix, and the driver-install guidance.

`docs/integrations/databases/vendor-matrix.md` publishes the capability matrix; the test
here holds that table to `capability_matrix()` row for row, so a profile change that is not
re-published (or a doc edit with no profile behind it) fails rather than drifting.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from batcher._internal.errors import BackendError, MissingDependencyError
from batcher.io.formats.sql.dbapi import _dsn
from batcher.io.formats.sql.dbapi.sink import WRITE_MODES
from batcher.io.formats.sql.uri import parse_uri
from batcher.io.formats.sql.vendors import (
    PROFILES,
    capability_matrix,
    install_hint,
    profile_for,
)
from batcher.io.formats.sql.vendors import profiles as profiles_module
from batcher.io.formats.sql.vendors.connect import adapt_connect_kwargs

pytestmark = pytest.mark.unit

_MATRIX_PAGE = Path(__file__).parents[2] / "docs/integrations/databases/vendor-matrix.md"


def test_the_dbapi_mode_literal_matches_the_sink():
    assert profiles_module._DBAPI_MODES == WRITE_MODES


def test_every_profile_scheme_is_routable():
    for profile in PROFILES.values():
        for scheme in profile.schemes:
            assert parse_uri(f"{scheme}://u@h/db").scheme == scheme


def test_profile_lookup_strips_a_driver_suffix():
    assert profile_for("postgresql+psycopg2").name == "PostgreSQL"
    assert profile_for("SQLSERVER").name == "SQL Server"
    assert profile_for("db2") is None


def test_the_matrix_reads_the_upsert_spelling_from_the_statement_builder():
    by_key = {(r["vendor"], r["route"]): r for r in capability_matrix()}
    assert by_key[("PostgreSQL", "dbapi")]["upsert"] == "on_conflict"
    assert by_key[("MySQL / MariaDB", "dbapi")]["upsert"] == "on_duplicate_key"
    assert by_key[("SQL Server", "dbapi")]["upsert"] == "merge"
    assert by_key[("Oracle", "dbapi")]["upsert"] == "merge"
    assert by_key[("Trino", "dbapi")]["upsert"] == "n/a"
    assert by_key[("MySQL / MariaDB", "connectorx")]["write_modes"] == "read-only"
    assert {r["status"] for r in capability_matrix()} == {"untested"}


def _published_rows() -> list[list[str]]:
    rows = []
    in_table = False
    for line in _MATRIX_PAGE.read_text().splitlines():
        if line.startswith("| Vendor | Route |"):
            in_table = True
            continue
        if in_table and line.startswith("| ---"):
            continue
        if in_table and line.startswith("|"):
            rows.append([cell.strip().strip("`") for cell in line.strip("|").split("|")])
        elif in_table:
            break
    return rows


def test_the_published_matrix_matches_the_profiles():
    published = _published_rows()
    assert published, "the capability table was not found on the page"
    expected = [
        [
            r["vendor"],
            r["route"],
            r["package"],
            r["read"],
            r["write_modes"],
            r["upsert"],
            r["status"],
        ]
        for r in capability_matrix()
    ]
    assert published == expected


def test_install_hint_names_every_route():
    hint = install_hint("mysql")
    assert "pip install connectorx (reads)" in hint
    assert "pip install pymysql (reads, writes)" in hint
    assert install_hint("informix") is None


def test_a_missing_dbapi_driver_names_the_vendor_routes(monkeypatch):
    monkeypatch.setattr(_dsn, "installed_driver", lambda scheme: None)
    with pytest.raises(MissingDependencyError, match="For SQL Server: pip install connectorx"):
        _dsn.driver_for("mssql")


def test_mssql_resolves_to_pymssql_with_generic_keywords(monkeypatch):
    monkeypatch.setattr(_dsn, "module_available", lambda m: m == "pymssql")
    driver, kwargs = _dsn.connect_target(parse_uri("mssql://sa@db:1433/shop", password="env:PW"))
    assert driver == "pymssql"
    assert kwargs == {
        "host": "db",
        "port": 1433,
        "user": "sa",
        "password": "env:PW",
        "database": "shop",
    }


def test_redshift_prefers_the_aws_driver(monkeypatch):
    monkeypatch.setattr(_dsn, "module_available", lambda m: m in ("redshift_connector", "psycopg"))
    driver, kwargs = _dsn.connect_target(parse_uri("redshift://etl@cluster:5439/dev"))
    assert driver == "redshift_connector"
    assert kwargs["database"] == "dev"


def test_trino_uri_maps_catalog_and_schema(monkeypatch):
    monkeypatch.setattr(_dsn, "module_available", lambda m: m == "trino.dbapi")
    driver, kwargs = _dsn.connect_target(
        parse_uri("trino://alice@trino.example:443/hive/web?http_scheme=https")
    )
    assert driver == "trino.dbapi"
    assert kwargs == {
        "host": "trino.example",
        "port": 443,
        "user": "alice",
        "catalog": "hive",
        "schema": "web",
        "http_scheme": "https",
    }


def test_trino_password_becomes_basic_auth_on_the_worker(monkeypatch):
    built = []

    class _Basic:
        def __init__(self, user, password):
            built.append((user, password))

    auth = types.ModuleType("trino.auth")
    auth.BasicAuthentication = _Basic
    monkeypatch.setitem(sys.modules, "trino.auth", auth)
    adapted = adapt_connect_kwargs("trino.dbapi", {"host": "h", "user": "alice", "password": "pw"})
    assert "password" not in adapted
    assert isinstance(adapted["auth"], _Basic)
    assert adapted["http_scheme"] == "https"
    assert built == [("alice", "pw")]
    # An explicit scheme is the caller's to choose.
    kept = adapt_connect_kwargs(
        "trino.dbapi", {"user": "a", "password": "p", "http_scheme": "http"}
    )
    assert kept["http_scheme"] == "http"


def test_trino_password_without_a_user_is_refused():
    with pytest.raises(BackendError, match="needs a user"):
        adapt_connect_kwargs("trino.dbapi", {"host": "h", "password": "pw"})


def test_other_drivers_pass_their_kwargs_through_unchanged():
    kwargs = {"host": "h", "password": "pw"}
    assert adapt_connect_kwargs("psycopg", kwargs) is kwargs


def test_every_type_rule_is_published():
    page = _MATRIX_PAGE.read_text()
    for profile in PROFILES.values():
        assert f"### {profile.name}" in page
        assert profile.live_env in page
        for rule in profile.type_rules:
            assert rule.vendor_type.replace("|", "\\|") in page, rule.vendor_type
