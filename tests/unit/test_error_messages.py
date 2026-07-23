"""Error quality is a contract, not a nicety.

Every assertion here is on a *structured field* (`.column`, `.available`, `.suggestion`,
`.hint`, `.install`) or on a type relationship, never on a substring of prose. That is
deliberate: a message should be free to be reworded, and a test that pins its wording
turns every improvement into a failure. What must not change is that the facts are
carried, that the closest name is offered, and that the exception is catchable as the
builtin a Python user would reach for.
"""

from __future__ import annotations

import pickle

import pytest

from batcher._internal.errors import (
    AccessDeniedError,
    BackendError,
    BatcherError,
    ColumnNotFoundError,
    ConfigError,
    DataQualityError,
    FormatError,
    MissingDependencyError,
    PlanError,
    candidate_list,
    did_you_mean,
    suggestion,
    unknown_message,
    unknown_value,
)
from batcher._internal.optional import require
from batcher._internal.registry import Registry
from batcher.governance import (
    AttributeIn,
    Encrypt,
    MatchesAttribute,
    Principal,
    Pseudonymize,
    Redact,
    SecurityCatalog,
    column_lineage,
    enforce,
)
from batcher.metadata.backends import BACKEND_NAMES, make_backend
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.backends.layered import LayeredBackend
from batcher.metadata.backends.redis import RedisBackend
from batcher.metadata.backends.sqlite import SQLiteBackend
from batcher.metadata.hub import MetadataHub

pytestmark = pytest.mark.unit


# --- the suggestion engine -------------------------------------------------
@pytest.mark.parametrize(
    ("typed", "pool", "expected"),
    [
        ("nmae", ["name", "age", "city"], "name"),  # transposition
        ("NAME", ["name", "age"], "name"),  # case only
        ("cust", ["customer_id", "order_id"], "customer_id"),  # abbreviation
        ("quantitiy", ["quantity"], "quantity"),  # inserted character
    ],
)
def test_did_you_mean_catches_each_class_of_typo(typed, pool, expected):
    assert did_you_mean(typed, pool)[0] == expected


def test_did_you_mean_offers_nothing_when_nothing_is_close():
    # A wrong suggestion is worse than none: it sends the reader after a name that was
    # never the answer.
    assert did_you_mean("zzzzzz", ["name", "age"]) == ()
    assert suggestion("zzzzzz", ["name", "age"]) == ""


def test_did_you_mean_tolerates_an_empty_or_non_string_name():
    assert did_you_mean("", ["a"]) == ()
    assert did_you_mean(None, ["a"]) == ()  # type: ignore[arg-type]
    assert did_you_mean("a", []) == ()


def test_did_you_mean_is_capped():
    pool = [f"col_{i}" for i in range(50)]
    assert len(did_you_mean("col_1", pool, n=3)) <= 3


def test_candidate_list_truncates_a_wide_schema():
    wide = [f"c{i:03d}" for i in range(400)]
    rendered = candidate_list(wide, limit=5)
    assert "+395 more" in rendered
    assert rendered.count("'") == 10  # exactly five quoted names


def test_candidate_list_is_empty_for_no_candidates():
    assert candidate_list([]) == ""


def test_unknown_message_omits_the_parts_it_has_nothing_for():
    bare = unknown_message("format", "zzz")
    assert "Did you mean" not in bare
    assert "Available" not in bare


# --- structured fields -----------------------------------------------------
def test_error_fields_carry_the_facts_without_parsing_the_message():
    err = ColumnNotFoundError.of("nmae", ["name", "age"])
    assert err.column == "nmae"
    assert err.suggestion.startswith("Did you mean")
    assert err.available == ("name", "age")


def test_str_renders_every_part_exactly_once():
    err = unknown_value(FormatError, "format", "parqet", ["parquet", "csv"])
    rendered = str(err)
    assert rendered.count("Did you mean") == 1
    assert rendered.count("Available") == 1


def test_doc_becomes_a_traceback_note_not_message_noise():
    err = BatcherError("boom", doc="docs/user-guide/io.md")
    assert "docs/user-guide/io.md" not in str(err)
    assert any("docs/user-guide/io.md" in note for note in err.__notes__)


def test_errors_survive_pickling_with_their_fields():
    # Distributed execution moves exceptions between processes; a field-carrying error
    # that arrives as a bare message on the driver is a field nobody can use.
    err = pickle.loads(pickle.dumps(ColumnNotFoundError.of("nmae", ["name"])))
    assert err.column == "nmae"
    assert err.available == ("name",)


# --- catchability ----------------------------------------------------------
@pytest.mark.parametrize(
    ("error", "builtin"),
    [
        (ColumnNotFoundError("x"), KeyError),
        (PlanError("x"), ValueError),
        (ConfigError("x"), ValueError),
        (DataQualityError("x"), ValueError),
        (AccessDeniedError("x"), PermissionError),
        (MissingDependencyError("x"), ImportError),
    ],
)
def test_catchable_as_the_builtin_a_user_would_reach_for(error, builtin):
    assert isinstance(error, builtin)


@pytest.mark.parametrize(
    "error",
    [
        ColumnNotFoundError("x"),
        PlanError("x"),
        ConfigError("x"),
        DataQualityError("x"),
        AccessDeniedError("x"),
        MissingDependencyError("x"),
        FormatError("x"),
    ],
)
def test_every_error_stays_catchable_as_batcher_error(error):
    # The root contract: one `except BatcherError` still catches everything, no matter
    # which builtin a subclass also inherits.
    assert isinstance(error, BatcherError)


def test_a_key_error_subclass_still_renders_its_message_unquoted():
    # `KeyError.__str__` is `repr(args[0])`, so inheriting it the wrong way round turns
    # every column error into a quoted blob. BatcherError must win the MRO.
    assert not str(ColumnNotFoundError("no column 'x'")).startswith('"')


def test_missing_dependency_is_still_a_backend_error():
    # The handlers written against the older shape must keep working.
    assert isinstance(MissingDependencyError("x"), BackendError)


# --- missing dependencies --------------------------------------------------
def test_missing_dependency_names_the_exact_install_command():
    with pytest.raises(MissingDependencyError) as caught:
        require("definitely_not_installed_xyz", feature="Foo", provides="foo-lib", extra="foo")
    assert caught.value.install == "pip install 'batcher-engine[foo]'"
    assert caught.value.install in str(caught.value)


def test_missing_dependency_chains_the_original_import_error():
    with pytest.raises(MissingDependencyError) as caught:
        require("definitely_not_installed_xyz", feature="Foo", provides="foo", extra="foo")
    assert isinstance(caught.value.__cause__, ImportError)


# --- the registry ----------------------------------------------------------
def test_registry_suggests_the_closest_registered_name():
    reg: Registry[str] = Registry("source")
    reg.add("parquet", "p")
    reg.add("csv", "c")
    with pytest.raises(BatcherError) as caught:
        reg.get("parqet")
    assert caught.value.suggestion.startswith("Did you mean")
    assert set(caught.value.available) == {"parquet", "csv"}


def test_registry_says_so_when_nothing_is_registered():
    with pytest.raises(BatcherError) as caught:
        Registry[str]("sink").get("delta")
    assert caught.value.hint  # "the registering module may not be imported"


def test_registry_rejects_a_name_no_lookup_could_ever_use():
    reg: Registry[str] = Registry("source")
    for bad in ("", None, 7):
        with pytest.raises(BatcherError):
            reg.add(bad, "x")  # type: ignore[arg-type]


def test_registry_repr_names_what_is_registered():
    reg: Registry[str] = Registry("source")
    reg.add("csv", "c")
    assert "csv" in repr(reg)
    assert "source" in repr(reg)


def test_registry_get_with_an_unhashable_name_is_typed_not_a_type_error():
    with pytest.raises(BatcherError):
        Registry[str]("source").get(["csv"])  # type: ignore[arg-type]


# --- metadata --------------------------------------------------------------
def test_unknown_metadata_backend_suggests_the_closest():
    with pytest.raises(ConfigError) as caught:
        make_backend("sqlight")
    assert "sqlite" in caught.value.suggestion
    assert set(caught.value.available) == set(BACKEND_NAMES)


def test_every_advertised_backend_name_is_actually_buildable():
    # The list in the error must never offer a name the factory cannot make.
    for name in ("in_process", "sqlite"):
        assert make_backend(name, ":memory:" if name == "sqlite" else None) is not None
    for name in set(BACKEND_NAMES) - {"in_process", "sqlite"}:
        with pytest.raises(ConfigError):  # needs a uri, not "unknown backend"
            make_backend(name)


def test_a_non_backend_is_rejected_where_it_is_passed():
    with pytest.raises(ConfigError):
        MetadataHub(object())
    with pytest.raises(ConfigError):
        LayeredBackend(object())


def test_metadata_hub_repr_answers_is_anything_being_recorded():
    rendered = repr(MetadataHub(InProcessBackend()))
    assert "recorded=0" in rendered
    assert "InProcessBackend" in rendered


def test_backend_reprs_name_their_location():
    assert ":memory:" in repr(SQLiteBackend(":memory:"))
    assert "InProcessBackend" in repr(InProcessBackend())


def test_sqlite_backend_names_the_path_it_could_not_open():
    with pytest.raises(ConfigError) as caught:
        SQLiteBackend("/nonexistent-dir-xyz-9f2/meta.db")
    assert "/nonexistent-dir-xyz-9f2/meta.db" in str(caught.value)


def test_sqlite_backend_rejects_a_non_path():
    with pytest.raises(ConfigError):
        SQLiteBackend(42)  # type: ignore[arg-type]


def test_redis_backend_explains_a_missing_scheme():
    with pytest.raises(ConfigError) as caught:
        RedisBackend("localhost:6379")
    assert "redis://" in caught.value.hint


def test_backends_that_need_a_uri_say_what_one_looks_like():
    with pytest.raises(ConfigError) as caught:
        RedisBackend(None)
    assert "redis://" in caught.value.hint


def test_unserializable_learned_params_name_the_offending_entry():
    hub = MetadataHub(InProcessBackend())
    with pytest.raises(ConfigError) as caught:
        hub.save_params("ns", {"good": 1, "bad": object()})
    assert "'bad'" in str(caught.value)


def test_learned_param_names_must_be_strings():
    hub = MetadataHub(InProcessBackend())
    with pytest.raises(ConfigError):
        hub.load_params(None)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        hub.put_keyed_param("ns", 7, 1)  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        hub.operator_history("scan")  # type: ignore[arg-type]


# --- governance ------------------------------------------------------------
def test_roles_as_a_bare_string_is_rejected_not_split_into_letters():
    # It would otherwise become eight one-character roles, matching no grant, so the
    # principal silently sees nothing and the catalog looks broken.
    with pytest.raises(PlanError) as caught:
        Principal("ana", roles="analyst")  # type: ignore[arg-type]
    assert 'roles=["analyst"]' in caught.value.hint


def test_principal_requires_a_usable_name_and_string_attributes():
    with pytest.raises(PlanError):
        Principal("")
    with pytest.raises(PlanError):
        Principal(123)  # type: ignore[arg-type]
    with pytest.raises(PlanError):
        Principal("ana", attrs={"level": 3})  # type: ignore[dict-item]


def test_principal_repr_round_trips_the_way_it_was_written():
    assert repr(Principal("ana", roles=["analyst"], attrs={"region": "EU"})) == (
        "Principal('ana', roles=['analyst'], attrs={'region': 'EU'})"
    )


def test_a_missing_row_filter_attribute_suggests_the_nearest_one():
    principal = Principal("ana", attrs={"region": "EU", "dept": "x"})
    with pytest.raises(PlanError) as caught:
        MatchesAttribute("region", "regoin")(principal)
    assert "region" in caught.value.suggestion
    assert set(caught.value.available) == {"region", "dept"}


def test_attribute_in_rejects_a_separator_split_cannot_use():
    with pytest.raises(PlanError):
        AttributeIn("region", "regions", sep="")


def test_row_filters_reject_a_blank_column_or_attribute_name():
    # A blank name survives declaration and fails at read time as an unknown column,
    # pointing at the governed scan instead of at the policy that is wrong.
    for build in (
        lambda: MatchesAttribute("", "region"),
        lambda: MatchesAttribute("region", ""),
        lambda: AttributeIn("", "regions"),
    ):
        with pytest.raises(PlanError):
            build()


def test_a_missing_attribute_with_no_near_match_points_at_the_principal():
    with pytest.raises(PlanError) as caught:
        MatchesAttribute("region", "clearance")(Principal("ana", attrs={"zzz": "1"}))
    assert caught.value.suggestion == ""
    assert "Principal" in caught.value.hint


def test_grant_select_as_a_bare_string_is_rejected():
    with pytest.raises(PlanError) as caught:
        SecurityCatalog().grant("analyst", on="t", select="id")  # type: ignore[arg-type]
    assert 'select=["id"]' in caught.value.hint


def test_policy_names_must_be_usable():
    for build in (
        lambda: SecurityCatalog().grant("analyst", on=""),
        lambda: SecurityCatalog().grant(None, on="t"),  # type: ignore[arg-type]
        lambda: SecurityCatalog().tag("t", "c"),  # no tags: governs nothing
        lambda: SecurityCatalog().mask_column("", "c", lambda c: c),
    ):
        with pytest.raises(PlanError):
            build()


def test_a_non_callable_policy_body_fails_where_it_is_declared():
    # Not at read time, in a traceback pointing at the plan rewrite.
    with pytest.raises(PlanError):
        SecurityCatalog().mask_column("t", "ssn", "REDACT")  # type: ignore[arg-type]
    with pytest.raises(PlanError):
        SecurityCatalog().mask_tag("pii", "nullify")  # type: ignore[arg-type]
    with pytest.raises(PlanError):
        SecurityCatalog().filter_rows("t", "region = EU")  # type: ignore[arg-type]


def test_catalog_repr_says_what_is_installed():
    catalog = SecurityCatalog().grant("analyst", on="t", select=["id"]).tag("t", "ssn", "pii")
    assert "grants=1" in repr(catalog)
    assert "tags=1" in repr(catalog)


def test_redact_rejects_settings_that_would_under_mask():
    with pytest.raises(PlanError):
        Redact(show_last=-3)
    with pytest.raises(PlanError):
        Redact(char="XY")


def test_key_masks_require_a_key():
    # An inline key is *not* warned about here: `hmac_sha256`/`aes_encrypt` already do
    # that when the mask is applied, and warning twice for one mistake is noise.
    for factory in (Pseudonymize, Encrypt):
        with pytest.raises(PlanError):
            factory("")
        with pytest.raises(PlanError):
            factory(None)  # type: ignore[arg-type]
        factory("env:KEY")


def test_a_bare_string_where_a_table_list_belongs_is_rejected():
    schema = _schema()
    catalog = SecurityCatalog().grant("analyst", on="people")
    with pytest.raises(PlanError):
        enforce(_scan(schema), "people", Principal("ana", roles=["analyst"]), catalog)
    with pytest.raises(PlanError):
        column_lineage(_scan(schema), "people.parquet")


def test_enforce_fails_closed_when_a_scanned_source_is_unnamed():
    # Previously the scan was left ungoverned, so a caller's off-by-one silently
    # disabled the policy on that table.
    from batcher.plan.logical import Scan

    catalog = SecurityCatalog().grant("analyst", on="people")
    with pytest.raises(PlanError):
        enforce(Scan(3, _schema()), ["people"], Principal("ana", roles=["analyst"]), catalog)


def test_enforce_leaves_an_ungoverned_plan_alone_even_with_no_table_names():
    from batcher.plan.logical import Scan

    plan = Scan(3, _schema())
    governed, events = enforce(plan, [], Principal("ana"), SecurityCatalog())
    assert governed is plan
    assert events == ()


def test_access_denied_carries_the_table_and_a_way_forward():
    catalog = SecurityCatalog().grant("admin", on="people", select=["id"])
    with pytest.raises(AccessDeniedError) as caught:
        enforce(_scan(_schema()), ["people"], Principal("ana", roles=["analyst"]), catalog)
    assert caught.value.table == "people"
    assert "grant" in caught.value.hint
    assert isinstance(caught.value, PermissionError)


def _schema():
    import pyarrow as pa

    from batcher.plan.schema import SchemaRef

    return SchemaRef.from_arrow(pa.schema([("id", pa.int64()), ("ssn", pa.string())]))


def _scan(schema):
    from batcher.plan.logical import Scan

    return Scan(0, schema)
