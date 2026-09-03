"""Writes are governed: privileges, revoke, deny, and the path spellings that used to evade them.

Three gaps closed together, because they are one gap seen from three sides — the catalog
knew exactly one privilege (`SELECT`), had no way to take it back, and matched tables by
raw string.

* **Writes were ungoverned.** A `SELECT` grant is the only privilege the catalog knew, so
  a principal that could not read a column could still write anywhere it liked. That
  undoes a masked read entirely: join the masked table, write the result somewhere with
  no policy, read it back.
* **Nothing could be taken away.** Grants union across a principal's roles, so there was
  no way to express "everything except `salary`", and no way to offboard a role.
* **A policy matched one spelling of a path.** ``s3a://vault/pii.parquet`` is the Hadoop
  spelling of ``s3://vault/pii.parquet`` and named the same object; a rule written about
  one did not fire on the other.

Every test here is written the way the contract asks for on an authorization boundary: it
proves the *deny*, and then proves the matching *allow*, because a check that refuses
everything passes the first half of every security test ever written.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest

import batcher as bt
from batcher._internal.errors import AccessDeniedError, PlanError
from batcher.api.merge.clauses import MergeClause
from batcher.api.security._write import merge_privileges, required_privileges
from batcher.governance import PRIVILEGES

pytestmark = pytest.mark.unit


@pytest.fixture
def table(tmp_path):
    """A two-row parquet table on disk, returned as its path."""
    path = str(tmp_path / "src.parquet")
    bt.from_pydict({"id": [1, 2], "email": ["a@x.com", "b@x.com"]}).write(path, format="parquet")
    return path


@pytest.fixture
def dest(tmp_path):
    """A path nothing has been written to yet."""
    return str(tmp_path / "out.parquet")


def _rows():
    return bt.from_pydict({"id": [7], "email": ["c@x.com"]})


# --------------------------------------------------------------------------
# Write privileges
# --------------------------------------------------------------------------
class TestWritePrivileges:
    """A grant closes the table, and each privilege must then be granted by name."""

    def test_a_reader_cannot_write_to_a_governed_table(self, dest):
        catalog = bt.SecurityCatalog().grant("analyst", on=dest)
        with (
            bt.security(catalog, bt.Principal("ana", roles=["analyst"])),
            pytest.raises(AccessDeniedError, match="missing INSERT"),
        ):
            _rows().write(dest, format="parquet")

    def test_the_insert_privilege_permits_the_write(self, dest):
        catalog = bt.SecurityCatalog().grant("loader", on=dest, privilege="INSERT")
        with bt.security(catalog, bt.Principal("etl", roles=["loader"])):
            _rows().write(dest, format="parquet", mode="error")
        assert bt.read.parquet(dest).count() == 1

    def test_insert_does_not_confer_delete(self, dest):
        """The whole reason to grant INSERT rather than "write": a load job cannot
        drop yesterday's data. `overwrite` destroys the rows already there."""
        catalog = bt.SecurityCatalog().grant("loader", on=dest, privilege="INSERT")
        _rows().write(dest, format="parquet")
        with (
            bt.security(catalog, bt.Principal("etl", roles=["loader"])),
            pytest.raises(AccessDeniedError, match="missing DELETE"),
        ):
            _rows().write(dest, format="parquet", mode="overwrite")

    def test_both_privileges_permit_an_overwrite(self, dest):
        catalog = (
            bt.SecurityCatalog()
            .grant("owner", on=dest, privilege="INSERT")
            .grant("owner", on=dest, privilege="DELETE")
        )
        _rows().write(dest, format="parquet")
        with bt.security(catalog, bt.Principal("root", roles=["owner"])):
            _rows().write(dest, format="parquet", mode="overwrite")
        assert bt.read.parquet(dest).count() == 1

    def test_a_table_no_policy_names_is_left_open(self, table, dest):
        """Installing a catalog must not lock every path it says nothing about."""
        catalog = bt.SecurityCatalog().grant("analyst", on=table)
        with bt.security(catalog, bt.Principal("ana", roles=["analyst"])):
            bt.read.parquet(table).write(dest, format="parquet")
        assert bt.read.parquet(dest).count() == 2

    def test_an_ungoverned_process_writes_as_before(self, dest):
        """No `security()` block: the write path is exactly what it was."""
        _rows().write(dest, format="parquet")
        assert bt.read.parquet(dest).count() == 1


class TestModePrivilegeMap:
    """Every write mode is classified, and an unclassified one refuses rather than guesses."""

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("append", ("INSERT",)),
            ("error", ("INSERT",)),
            ("ignore", ("INSERT",)),
            ("overwrite", ("INSERT", "DELETE")),
            ("overwrite_partitions", ("INSERT", "DELETE")),
            ("upsert", ("INSERT", "UPDATE")),
            ("update", ("UPDATE",)),
            ("delete", ("DELETE",)),
            ("delete_insert", ("INSERT", "DELETE")),
        ],
    )
    def test_each_mode_maps_to_what_it_does_to_existing_rows(self, mode, expected):
        assert required_privileges(mode) == expected

    def test_an_unknown_mode_is_refused_not_permitted(self):
        with pytest.raises(PlanError, match="write mode"):
            required_privileges("obliterate")

    def test_every_save_mode_and_dml_verb_is_classified(self):
        """The map must not fall behind the writer's vocabulary. A mode the writer accepts
        but this map has never heard of would raise at write time for everyone, governed or
        not — so the coupling is asserted here rather than discovered in production."""
        from batcher.api.io_namespace._write_opts import SAVE_MODES, dml_write_modes

        modes = set(SAVE_MODES) | set(dml_write_modes("dbapi")) | set(dml_write_modes("mongo"))
        for mode in modes:
            assert required_privileges(mode), mode

    def test_a_merge_needs_only_what_its_clauses_do(self):
        assert merge_privileges([MergeClause("not_matched", "insert")]) == ("INSERT",)
        assert merge_privileges([MergeClause("matched", "update")]) == ("UPDATE",)
        assert merge_privileges(
            [MergeClause("matched", "delete"), MergeClause("not_matched", "insert")]
        ) == ("INSERT", "DELETE")

    def test_an_unclassified_merge_action_fails_closed(self):
        """The one place over-requiring is right: a new action must not be free."""
        assert merge_privileges([MergeClause("matched", "teleport")]) == (
            "INSERT",
            "UPDATE",
            "DELETE",
        )


# --------------------------------------------------------------------------
# revoke
# --------------------------------------------------------------------------
class TestRevoke:
    """`revoke` removes the grant, which is not the same as denying."""

    def test_revoking_the_only_grant_reopens_the_table(self, table):
        """SQL semantics, and the thing to get right in an offboarding path: a table with
        no grants left is a table nobody wrote a policy about."""
        catalog = bt.SecurityCatalog().grant("intern", on=table)
        sam = bt.Principal("sam", roles=["intern"])
        assert catalog.holds(table, sam, "SELECT")
        catalog.revoke("intern", on=table)
        assert catalog.grants_on(table) == []
        assert catalog.holds(table, sam, "SELECT")

    def test_revoking_one_role_leaves_the_others_governed(self, table):
        catalog = bt.SecurityCatalog().grant("intern", on=table).grant("analyst", on=table)
        catalog.revoke("intern", on=table)
        sam, ana = bt.Principal("sam", roles=["intern"]), bt.Principal("ana", roles=["analyst"])
        assert not catalog.holds(table, sam, "SELECT")
        assert catalog.holds(table, ana, "SELECT")

    def test_revoking_one_privilege_leaves_the_other(self, dest):
        catalog = (
            bt.SecurityCatalog()
            .grant("owner", on=dest, privilege="INSERT")
            .grant("owner", on=dest, privilege="DELETE")
        )
        catalog.revoke("owner", on=dest, privilege="DELETE")
        root = bt.Principal("root", roles=["owner"])
        assert catalog.holds(dest, root, "INSERT")
        assert not catalog.holds(dest, root, "DELETE")

    def test_revoking_takes_effect_on_a_real_write(self, tmp_path):
        first, second = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
        catalog = (
            bt.SecurityCatalog()
            .grant("loader", on=first, privilege="INSERT")
            .grant("loader", on=second, privilege="INSERT")
        )
        etl = bt.Principal("etl", roles=["loader"])
        with bt.security(catalog, etl):
            _rows().write(first, format="parquet", mode="error")
        catalog.revoke("loader", on=second, privilege="INSERT")
        catalog.grant("owner", on=second, privilege="INSERT")
        with bt.security(catalog, etl), pytest.raises(AccessDeniedError):
            _rows().write(second, format="parquet", mode="error")

    def test_revoking_what_was_never_granted_is_not_an_error(self, table):
        """An offboarding script that must first check what it is undoing races itself."""
        catalog = bt.SecurityCatalog()
        assert catalog.revoke("nobody", on=table) is catalog

    def test_revoke_matches_across_path_spellings(self):
        catalog = bt.SecurityCatalog().grant("analyst", on="s3://vault/t.parquet")
        catalog.revoke("analyst", on="s3a://vault/t.parquet")
        assert catalog.grants_on("s3://vault/t.parquet") == []


# --------------------------------------------------------------------------
# deny
# --------------------------------------------------------------------------
class TestDeny:
    """A denial outranks every grant, which is the only way to say "all except this"."""

    def test_a_denied_column_is_not_visible_despite_a_full_grant(self, table):
        catalog = (
            bt.SecurityCatalog()
            .grant("analyst", on=table)
            .deny("analyst", on=table, select=["email"])
        )
        ana = bt.Principal("ana", roles=["analyst"])
        assert catalog.visible_columns(table, ["id", "email"], ana) == ["id"]

    def test_the_denied_column_is_gone_from_a_real_read(self, table):
        catalog = (
            bt.SecurityCatalog()
            .grant("analyst", on=table)
            .deny("analyst", on=table, select=["email"])
        )
        with bt.security(catalog, bt.Principal("ana", roles=["analyst"])):
            ds = bt.read.parquet(table)
        assert ds.columns == ["id"]

    def test_a_denial_beats_a_grant_from_another_role(self, table):
        """The union rule is what makes this necessary: a principal holding two roles sees
        whatever either sees, so a block must be expressible as a block."""
        catalog = bt.SecurityCatalog().grant("auditor", on=table).deny("contractor", on=table)
        both = bt.Principal("x", roles=["auditor", "contractor"])
        assert catalog.visible_columns(table, ["id", "email"], both) == []
        assert not catalog.holds(table, both, "SELECT")

    def test_a_denial_survives_a_later_grant(self, table):
        catalog = bt.SecurityCatalog().deny("contractor", on=table)
        catalog.grant("contractor", on=table)
        assert not catalog.holds(table, bt.Principal("c", roles=["contractor"]), "SELECT")

    def test_denying_a_write_privilege_refuses_a_real_write(self, dest):
        catalog = (
            bt.SecurityCatalog()
            .grant("loader", on=dest, privilege="INSERT")
            .deny("loader", on=dest, privilege="INSERT")
        )
        with (
            bt.security(catalog, bt.Principal("etl", roles=["loader"])),
            pytest.raises(AccessDeniedError, match="missing INSERT"),
        ):
            _rows().write(dest, format="parquet", mode="error")

    def test_a_denial_alone_governs_a_table(self, table):
        """A catalog with only denials still governs — otherwise "deny X" on an otherwise
        open table would be a no-op, which is the opposite of what it says."""
        catalog = bt.SecurityCatalog().deny("contractor", on=table, select=["email"])
        assert catalog.governs(table)
        c = bt.Principal("c", roles=["contractor"])
        assert catalog.visible_columns(table, ["id", "email"], c) == ["id"]

    def test_a_column_list_is_refused_for_a_row_level_privilege(self, table):
        with pytest.raises(PlanError, match="acts on whole rows"):
            bt.SecurityCatalog().deny("r", on=table, select=["x"], privilege="DELETE")

    def test_denials_are_reported(self, table):
        catalog = bt.SecurityCatalog().deny("contractor", on=table, select=["email"])
        assert [sorted(d.columns) for d in catalog.denials_on(table)] == [["email"]]


# --------------------------------------------------------------------------
# Path aliasing
# --------------------------------------------------------------------------
class TestPathAliasing:
    """One object, one policy, however the caller spells the path."""

    @pytest.mark.parametrize(
        "alias",
        [
            "s3a://vault/pii.parquet",
            "s3n://vault/pii.parquet",
            "S3://vault/pii.parquet",
            "s3://vault//pii.parquet",
            "s3://vault/./pii.parquet",
            "s3://vault/tmp/../pii.parquet",
        ],
    )
    def test_every_spelling_hits_the_policy(self, alias):
        catalog = bt.SecurityCatalog().grant("analyst", on="s3://vault/pii.parquet", select=["id"])
        ana = bt.Principal("ana", roles=["analyst"])
        assert catalog.governs(alias)
        assert catalog.visible_columns(alias, ["id", "ssn"], ana) == ["id"]

    def test_a_policy_declared_under_an_alias_governs_the_canonical_name(self):
        catalog = bt.SecurityCatalog().grant("analyst", on="gcs://b/t.parquet", select=["id"])
        ana = bt.Principal("ana", roles=["analyst"])
        assert catalog.visible_columns("gs://b/t.parquet", ["id", "ssn"], ana) == ["id"]

    def test_the_bucket_is_not_case_folded(self):
        """Case *is* significant in a bucket name and an S3 key: folding it would merge
        two real objects, which is a worse bug than the one being fixed."""
        catalog = bt.SecurityCatalog().grant("analyst", on="s3://Vault/K")
        assert catalog.governs("s3a://Vault/K")
        assert not catalog.governs("s3://vault/k")

    def test_a_local_path_folds_the_same_way(self, tmp_path):
        catalog = bt.SecurityCatalog().grant("analyst", on=f"{tmp_path}/t.parquet")
        assert catalog.governs(f"{tmp_path}//./t.parquet")

    def test_the_alias_cannot_walk_past_a_write_policy(self, tmp_path):
        dest = str(tmp_path / "out.parquet")
        catalog = bt.SecurityCatalog().grant("analyst", on=dest)
        with (
            bt.security(catalog, bt.Principal("ana", roles=["analyst"])),
            pytest.raises(AccessDeniedError),
        ):
            _rows().write(f"{tmp_path}//./out.parquet", format="parquet")


# --------------------------------------------------------------------------
# The audit trail
# --------------------------------------------------------------------------
class TestWritesAreAudited:
    """ "Who wrote to this table" and "who read it" are one log, in one shape."""

    def test_an_allowed_write_emits_an_event_naming_the_privilege(self, dest):
        catalog = bt.SecurityCatalog().grant("loader", on=dest, privilege="INSERT")
        seen = []
        with bt.security(catalog, bt.Principal("etl", roles=["loader"]), audit=seen.append):
            _rows().write(dest, format="parquet", mode="error")
        events = [e for e in seen if e.privilege == "INSERT"]
        assert len(events) == 1
        assert events[0].allowed
        assert events[0].table == dest
        assert set(events[0].visible) == {"id", "email"}

    def test_a_refused_write_is_audited_before_it_raises(self, dest):
        catalog = bt.SecurityCatalog().grant("analyst", on=dest)
        seen = []
        with (
            bt.security(catalog, bt.Principal("ana", roles=["analyst"]), audit=seen.append),
            pytest.raises(AccessDeniedError),
        ):
            _rows().write(dest, format="parquet", mode="error")
        assert [(e.privilege, e.allowed, e.denied) for e in seen] == [
            ("INSERT", False, ("INSERT",))
        ]

    def test_the_rendering_names_the_write(self, dest):
        catalog = bt.SecurityCatalog().grant("analyst", on=dest)
        seen = []
        with (
            bt.security(catalog, bt.Principal("ana", roles=["analyst"]), audit=seen.append),
            pytest.raises(AccessDeniedError),
        ):
            _rows().write(dest, format="parquet", mode="error")
        assert "DENY INSERT" in str(seen[0])

    def test_the_durable_log_records_which_privilege_was_decided(self, dest, tmp_path):
        """The callback sink is a convenience; `governance.audit_path` is the artifact an
        auditor reads. Without the privilege in it a write decision and a read decision are
        the same record -- same principal, same table, same column list -- and "who wrote to
        this table" is unanswerable from the file."""
        import dataclasses
        import json

        from batcher.config import active_config, set_config

        audit = str(tmp_path / "audit.jsonl")
        previous = active_config()
        set_config(
            previous.replace(governance=dataclasses.replace(previous.governance, audit_path=audit))
        )
        try:
            catalog = bt.SecurityCatalog().grant("loader", on=dest, privilege="INSERT")
            with bt.security(catalog, bt.Principal("etl", roles=["loader"])):
                _rows().write(dest, format="parquet", mode="error")
        finally:
            set_config(previous)
        lines = Path(audit).read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        assert [(r["privilege"], r["allowed"]) for r in records] == [("INSERT", True)]

    def test_a_read_event_still_reads_as_a_read(self, table):
        catalog = bt.SecurityCatalog().grant("analyst", on=table, select=["id"])
        seen = []
        with bt.security(catalog, bt.Principal("ana", roles=["analyst"]), audit=seen.append):
            bt.read.parquet(table)
        assert seen[0].privilege == "SELECT"
        assert "ALLOW ana" in str(seen[0])


# --------------------------------------------------------------------------
# The declaration surface fails loudly
# --------------------------------------------------------------------------
class TestDeclarationChecks:
    """A policy stored under a name nothing matches governs nothing while looking installed."""

    @pytest.mark.parametrize("bad", ["write", "READ", "", "MODIFY", None, 3])
    def test_an_unknown_privilege_is_refused_at_declaration(self, bad, table):
        with pytest.raises(PlanError):
            bt.SecurityCatalog().grant("r", on=table, privilege=bad)

    def test_a_privilege_is_case_normalized(self, table):
        catalog = bt.SecurityCatalog().grant("r", on=table, privilege="insert")
        assert catalog.grants_on(table)[0].privilege == "INSERT"

    def test_a_column_list_is_refused_for_a_row_level_grant(self, table):
        with pytest.raises(PlanError, match="acts on whole rows"):
            bt.SecurityCatalog().grant("r", on=table, select=["id"], privilege="INSERT")

    def test_the_privilege_vocabulary_is_the_sql_one(self):
        assert PRIVILEGES == ("SELECT", "INSERT", "UPDATE", "DELETE")

    def test_holds_refuses_a_non_principal(self, table):
        with pytest.raises(PlanError, match="Principal"):
            bt.SecurityCatalog().holds(table, "ana", "SELECT")


# --------------------------------------------------------------------------
# Every write path, not just the one the check was written against
# --------------------------------------------------------------------------
class TestThereIsNoUngovernedWritePath:
    """An authorization control with a bypass is worse than none, so enumerate the paths.

    Most writes funnel through `api.terminal.core._write`, which is why the check lives
    there. Two do not, and both were found by looking rather than by a test failing:

    * a **native MERGE** into Delta or Iceberg, where the format's own client performs the
      merge -- the targets an enterprise deployment cares about most;
    * a **single-node streaming write**, where the distributed sibling goes through
      `_write` and this one builds its sink directly, so the write would have been
      governed only when it happened to be distributed.
    """

    def test_a_streaming_write_is_authorized_before_the_query_starts(self, tmp_path):
        """Refusing after the first micro-batch would already have written."""
        dest = str(tmp_path / "stream_out")
        catalog = bt.SecurityCatalog().grant("analyst", on=dest)
        source = bt.read.rate(rows_per_second=1)
        with (
            bt.security(catalog, bt.Principal("ana", roles=["analyst"])),
            pytest.raises(AccessDeniedError, match="missing INSERT"),
        ):
            source.write(dest, format="parquet", trigger=bt.Trigger.Once())
        assert not os.path.exists(dest)

    def test_the_granted_role_may_start_the_same_stream(self, tmp_path):
        """The matching allow: without it the test above passes on any refusal at all."""
        dest = str(tmp_path / "stream_ok")
        catalog = bt.SecurityCatalog().grant("loader", on=dest, privilege="INSERT")
        source = bt.read.rate(rows_per_second=1)
        with bt.security(catalog, bt.Principal("etl", roles=["loader"])):
            query = source.write(dest, format="parquet", trigger=bt.Trigger.Once())
        query.stop()

    def test_the_check_precedes_any_distributed_fan_out(self, monkeypatch, tmp_path):
        """The distributed write path inherits authorization by construction, and this is
        what makes that a measurement rather than an argument.

        `_write` authorizes before it resolves `distributed` and before it hands anything to
        `dist`, so a refused write never reaches a worker. Asserted by making the fan-out
        itself an error: if authorization ran after it, this fails with that error instead
        of the refusal. CI installs no Ray, so the ordering is the only part of the
        distributed path testable here -- and it is the part that matters.
        """
        from batcher.dist.executors import write as dist_write

        def _unreachable(*args, **kwargs):
            raise AssertionError("the write fanned out before it was authorized")

        monkeypatch.setattr(dist_write, "_distributed_write_plan", _unreachable)
        dest = str(tmp_path / "out.parquet")
        catalog = bt.SecurityCatalog().grant("analyst", on=dest)
        with (
            bt.security(catalog, bt.Principal("ana", roles=["analyst"])),
            pytest.raises(AccessDeniedError, match="missing INSERT"),
        ):
            _rows().write(dest, format="parquet", distributed=True)

    def test_a_native_merge_authorizes_before_it_materializes(self, monkeypatch, tmp_path):
        """`native_merge` is checked without a Delta install: the refusal must happen
        before it reaches the format client, so reaching the client at all is the failure."""
        from batcher.api.merge import native

        def _unreachable(*args, **kwargs):
            raise AssertionError("the merge ran before it was authorized")

        monkeypatch.setattr(native, "native_merge", native.native_merge)
        monkeypatch.setitem(
            __import__("sys").modules,
            "batcher.api.merge.delta_native",
            type("m", (), {"merge_into_delta": staticmethod(_unreachable)}),
        )
        target = str(tmp_path / "table")
        catalog = bt.SecurityCatalog().grant("analyst", on=target)
        source = bt.from_pydict({"id": [1], "v": [2]})
        with (
            bt.security(catalog, bt.Principal("ana", roles=["analyst"])),
            pytest.raises(AccessDeniedError),
        ):
            native.native_merge(
                source, target, ["id"], [MergeClause("not_matched", "insert")], "delta", {}
            )

    def test_creating_a_merge_target_refuses_to_clobber_one_that_appeared(self, tmp_path):
        """A merge whose target does not exist writes it through the ordinary writer. That
        write is spelled `mode="error"` rather than `"overwrite"`, which says what the path
        knows -- the target was absent when the merge planned.

        Two things follow. If something created the table between the plan and the write,
        refusing beats destroying what that writer put there. And an overwrite is charged
        DELETE by write governance, because an overwrite normally destroys the rows already
        present, while this one has none to destroy.
        """
        from batcher.api.merge.clauses import MergeClause as _Clause
        from batcher.api.merge.execute import _write_new_table

        target = str(tmp_path / "raced.parquet")
        bt.from_pydict({"id": [99], "v": [0]}).write(target, format="parquet")
        with pytest.raises(PlanError, match="already exists"):
            _write_new_table(
                bt.from_pydict({"id": [1], "v": [10]}),
                target,
                ["id"],
                [_Clause("not_matched", "insert")],
                "parquet",
                {},
            )
        # The row the concurrent writer put there is still the row that is there.
        assert bt.read.parquet(target).to_pydict()["id"] == [99]

    def test_that_create_is_charged_insert_and_not_delete(self, tmp_path):
        """The privilege half of the same change: an insert-only load job granted INSERT
        would otherwise have worked on every run except its first."""
        assert required_privileges("error") == ("INSERT",)
        assert required_privileges("overwrite") == ("INSERT", "DELETE")

    def test_a_native_merge_needs_only_what_its_clauses_do(self, monkeypatch, tmp_path):
        """The allow half, and it also pins the narrowness: an insert-only merge is
        permitted by INSERT alone, without UPDATE or DELETE."""
        from batcher.api.merge import native

        reached = []
        monkeypatch.setitem(
            __import__("sys").modules,
            "batcher.api.merge.delta_native",
            type(
                "m",
                (),
                {"merge_into_delta": staticmethod(lambda *a, **k: reached.append(True))},
            ),
        )
        target = str(tmp_path / "table")
        catalog = bt.SecurityCatalog().grant("loader", on=target, privilege="INSERT")
        source = bt.from_pydict({"id": [1], "v": [2]})
        with bt.security(catalog, bt.Principal("etl", roles=["loader"])):
            native.native_merge(
                source, target, ["id"], [MergeClause("not_matched", "insert")], "delta", {}
            )
        assert reached == [True]


class TestMaintenanceCannotDestroyAGovernedTable:
    """The worst thing found in this pass, and it was not a bypass -- it was data loss.

    `bt.compact()` reads a table and writes the result back over it. Inside a
    `security()` block that read is the *principal's* view, so the write replaced the
    table with it: measured on a two-file table under a catalog masking `email` and
    withholding `ssn`, compaction left ``'XXXXXXX'`` for every address and no `ssn`
    column at all. The real values were gone, nothing raised, and the result carried no
    sign anything had happened.

    A governed read narrowing what one principal sees is the feature. The same read fed
    back into a write narrows the *table*, permanently.

    `bt.vacuum(dry_run=False)` is the second half and a plainer one: it deletes data
    files by reaching the format backend directly, never passing through the write path,
    so it was a way to destroy a governed table's data holding no privilege at all.
    """

    @staticmethod
    def _table(tmp_path):
        path = str(tmp_path / "governed")
        bt.from_pydict(
            {"id": [1, 2], "email": ["a@x.com", "b@x.com"], "ssn": ["111", "222"]}
        ).repartition(num_files=2).write(path, format="parquet")
        return path

    @staticmethod
    def _catalog(path):
        return (
            bt.SecurityCatalog()
            .grant("ops", on=path, select=["id", "email"])
            .grant("ops", on=path, privilege="INSERT")
            .grant("ops", on=path, privilege="DELETE")
            .mask_column(path, "email", lambda c: bt.mask(c))
        )

    def test_compacting_a_governed_table_is_refused(self, tmp_path):
        path = self._table(tmp_path)
        with (
            bt.security(self._catalog(path), bt.Principal("ops", roles=["ops"])),
            pytest.raises(AccessDeniedError, match="masked and column-pruned view"),
        ):
            bt.compact(path, num_files=1, format="parquet")

    def test_the_data_survives_the_refusal(self, tmp_path):
        """The assertion that matters. A refusal that still destroyed the table would
        satisfy the test above."""
        path = self._table(tmp_path)
        with (
            contextlib.suppress(AccessDeniedError),
            bt.security(self._catalog(path), bt.Principal("ops", roles=["ops"])),
        ):
            bt.compact(path, num_files=1, format="parquet")
        rows = bt.read.parquet(path).sort("id").to_pydict()
        assert rows["email"] == ["a@x.com", "b@x.com"]
        assert rows["ssn"] == ["111", "222"]

    def test_holding_every_privilege_does_not_make_it_safe(self, tmp_path):
        """Granting more cannot help: the damage comes from the *mask*, which applies to a
        principal the catalog trusts completely. Refusing on "the table is governed" rather
        than on "this principal is restricted" is what covers it."""
        path = self._table(tmp_path)
        catalog = (
            bt.SecurityCatalog()
            .grant("ops", on=path)
            .grant("ops", on=path, privilege="INSERT")
            .grant("ops", on=path, privilege="DELETE")
            .mask_column(path, "email", lambda c: bt.mask(c))
        )
        with (
            bt.security(catalog, bt.Principal("ops", roles=["ops"])),
            pytest.raises(AccessDeniedError),
        ):
            bt.compact(path, num_files=1, format="parquet")

    def test_an_ungoverned_compaction_is_untouched(self, tmp_path):
        """Maintenance outside a security() block is how every warehouse runs OPTIMIZE,
        and it must keep working exactly as before."""
        path = self._table(tmp_path)
        bt.compact(path, num_files=1, format="parquet")
        rows = bt.read.parquet(path).sort("id").to_pydict()
        assert rows["email"] == ["a@x.com", "b@x.com"]
        assert rows["ssn"] == ["111", "222"]

    def test_a_table_the_catalog_does_not_name_is_still_compactable(self, tmp_path):
        """Installing a catalog must not stop maintenance on everything else."""
        path = self._table(tmp_path)
        other = str(tmp_path / "elsewhere")
        catalog = bt.SecurityCatalog().grant("ops", on=other)
        with bt.security(catalog, bt.Principal("ops", roles=["ops"])):
            bt.compact(path, num_files=1, format="parquet")
        assert bt.read.parquet(path).count() == 2

    def test_a_copy_on_write_merge_into_a_governed_table_is_refused(self, tmp_path):
        """A merge is the same read-modify-write shape. It reads the files it will rewrite,
        composes the clauses over them and writes the result back -- so every row the
        clauses do *not* match is carried through the principal's view and rewritten from
        it."""
        path = self._table(tmp_path)
        with (
            bt.security(self._catalog(path), bt.Principal("ops", roles=["ops"])),
            pytest.raises(AccessDeniedError, match="masked and column-pruned view"),
        ):
            bt.from_pydict({"id": [1], "email": ["z@z.com"], "ssn": ["9"]}).write.merge_into(
                path, on="id"
            ).when_matched().update_all().when_not_matched().insert_all().execute()

    def test_the_unmatched_rows_survive_the_refusal(self, tmp_path):
        """The row the merge does not match is the one that was destroyed: it has no new
        value to be given, so it was rewritten from whatever the governed read returned."""
        path = str(tmp_path / "one_file")
        bt.from_pydict(
            {"id": [1, 2], "email": ["a@x.com", "b@x.com"], "ssn": ["111", "222"]}
        ).repartition(num_files=1).write(path, format="parquet")
        catalog = self._catalog(path)
        with (
            contextlib.suppress(AccessDeniedError),
            bt.security(catalog, bt.Principal("ops", roles=["ops"])),
        ):
            bt.from_pydict({"id": [1], "email": ["z@z.com"], "ssn": ["9"]}).write.merge_into(
                path, on="id"
            ).when_matched().update_all().when_not_matched().insert_all().execute()
        assert bt.read.parquet(path).sort("id").to_pydict()["email"] == ["a@x.com", "b@x.com"]

    def test_an_ungoverned_merge_is_untouched(self, tmp_path):
        path = str(tmp_path / "one_file")
        bt.from_pydict({"id": [1, 2], "v": [10, 20]}).repartition(num_files=1).write(
            path, format="parquet"
        )
        bt.from_pydict({"id": [1], "v": [99]}).write.merge_into(
            path, on="id"
        ).when_matched().update_all().when_not_matched().insert_all().execute()
        assert bt.read.parquet(path).sort("id").to_pydict()["v"] == [99, 20]

    def test_a_native_merge_stays_available_on_a_governed_table(self, monkeypatch, tmp_path):
        """Delta and Iceberg merge inside their own client, against the raw table, so
        nothing is read through the principal's view. Refusing those too would take the
        capability away from exactly the formats an enterprise deployment uses."""
        from batcher.api.merge import native

        reached = []
        monkeypatch.setitem(
            __import__("sys").modules,
            "batcher.api.merge.delta_native",
            type(
                "m",
                (),
                {"merge_into_delta": staticmethod(lambda *a, **k: reached.append(True))},
            ),
        )
        target = str(tmp_path / "delta_table")
        catalog = (
            bt.SecurityCatalog().grant("ops", on=target).grant("ops", on=target, privilege="INSERT")
        )
        with bt.security(catalog, bt.Principal("ops", roles=["ops"])):
            native.native_merge(
                bt.from_pydict({"id": [1]}),
                target,
                ["id"],
                [MergeClause("not_matched", "insert")],
                "delta",
                {},
            )
        assert reached == [True]

    def test_a_real_vacuum_needs_delete(self, tmp_path):
        """`vacuum` reaches the format backend directly, so the refusal has to happen
        before it. The table need not exist: if authorization ran after the backend, this
        would fail on the missing table instead."""
        path = str(tmp_path / "delta_table")
        catalog = bt.SecurityCatalog().grant("ops", on=path, select=["id"])
        with (
            bt.security(catalog, bt.Principal("ops", roles=["ops"])),
            pytest.raises(AccessDeniedError, match="missing DELETE"),
        ):
            bt.vacuum(path, dry_run=False, format="delta")

    def test_a_dry_run_needs_no_privilege(self, tmp_path):
        """It deletes nothing, and refusing it would take away the check an operator runs
        *before* deciding whether the deletion is safe. It must fail for some other reason
        than authorization."""
        path = str(tmp_path / "delta_table")
        catalog = bt.SecurityCatalog().grant("ops", on=path, select=["id"])
        with (
            bt.security(catalog, bt.Principal("ops", roles=["ops"])),
            pytest.raises(Exception) as excinfo,
        ):
            bt.vacuum(path, dry_run=True, format="delta")
        assert not isinstance(excinfo.value, AccessDeniedError)


# --------------------------------------------------------------------------
# The same bypass, in the place where it breaches sovereignty
# --------------------------------------------------------------------------
class TestResidencyMatchesCanonically:
    """`ResidencyCatalog` keys on paths too, and had the identical alias hole.

    The consequence differs, which is why it is worth its own tests. An evaded
    `SecurityCatalog` rule returns columns someone should not have seen. An evaded
    residency rule lets a scheduler place a stage on regulated data in a region the
    operator has a legal obligation to keep it out of.
    """

    @staticmethod
    def _catalog():
        from batcher.governance import DataResidency, ResidencyCatalog

        return ResidencyCatalog(mode="enforce").register(
            DataResidency("s3://eu/", frozenset({"eu-west-1"}), obligation="GDPR")
        )

    @pytest.mark.parametrize(
        "alias",
        [
            "s3://eu/orders",
            "s3a://eu/orders",
            "s3n://eu/orders",
            "S3://eu/orders",
            "s3://eu//orders",
            "s3://eu/./orders",
        ],
    )
    def test_every_spelling_is_governed(self, alias):
        assert self._catalog().rule_for(alias) is not None
        assert not self._catalog().check(alias, "us-east-1").allowed

    def test_the_permitted_region_is_still_permitted(self):
        """The matching allow, without which "it refuses everything" would pass too."""
        assert self._catalog().check("s3a://eu/orders", "eu-west-1").allowed

    def test_a_trailing_slash_still_means_a_directory_prefix(self):
        """Canonicalization drops a trailing slash, and the slash is load-bearing here:
        without it `s3://eu/` would start matching `s3://eubank/`, quarantining a
        different customer's data to the EU."""
        assert self._catalog().rule_for("s3://eubank/orders") is None

    def test_an_unregistered_dataset_is_still_unrestricted(self):
        assert self._catalog().check("s3://us/orders", "us-east-1").allowed

    def test_longest_prefix_still_wins(self):
        from batcher.governance import DataResidency, ResidencyCatalog

        catalog = (
            ResidencyCatalog(mode="enforce")
            .register(DataResidency("s3://eu/", frozenset({"eu-west-1"})))
            .register(DataResidency("s3://eu/public/", frozenset({"eu-west-1", "us-east-1"})))
        )
        assert catalog.check("s3a://eu/public/x", "us-east-1").allowed
        assert not catalog.check("s3a://eu/private/x", "us-east-1").allowed
