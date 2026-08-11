"""A ``**kwargs`` bag on the public API rejects what it cannot honour.

Four surfaces collect keywords and forward them somewhere else: `register_function`, the
`@udf` decorator, `bt.tenant`, and `Config.replace`. A catch-all is the right shape for each —
the option set belongs to the thing being configured, not to the wrapper — but it moves every
mistake away from the line that made it. The failures that motivated these tests:

* `register_function(..., aggregate=True)` looked like it registered a UDAF, registered nothing
  of the sort, and failed at query time inside pyarrow;
* a misspelled `result_typ` was accepted and dropped, leaving the required option unset;
* a `map_batches` option on the scalar form was accepted and never read;
* `@udf(output_column=...)` failed at *apply* time with a `TypeError` naming
  `DatasetML.map_batches()`, a method the user never called;
* `bt.tenant(..., typo=1)` and `Config.replace(typo=1)` failed with a `TypeError` naming a
  dataclass `__init__` and listing nothing.
"""

from __future__ import annotations

import pyarrow.compute as pc
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


@pytest.fixture
def session():
    s = bt.Session()
    s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
    return s


# --- what is rejected -------------------------------------------------------------------


@pytest.mark.parametrize("option", ["aggregate", "agg", "is_aggregate"])
def test_asking_for_an_aggregate_says_what_to_use_instead(session, option):
    with pytest.raises(PlanError, match="map_groups"):
        session.register_function("f", lambda v: v, **{option: True})


def test_a_misspelled_keyword_is_not_swallowed(session):
    with pytest.raises(PlanError, match="result_typ"):
        session.register_function("f", lambda v: v, result_typ="int64")


def test_a_map_batches_option_on_a_scalar_function_is_rejected(session):
    """The scalar form never reads config, so this was accepted and had no effect."""
    with pytest.raises(PlanError, match="scalar function takes no map_batches options"):
        session.register_function("f", lambda v: v, num_gpus=1)


def test_an_unknown_option_on_a_table_function_lists_the_valid_ones(session):
    with pytest.raises(PlanError, match="Valid options:"):
        session.register_function("f", lambda b: b, table=True, bogus=1)


def test_batch_format_on_a_scalar_function_is_rejected(session):
    """It is a named parameter, so it bypasses the ``**config`` check and needs its own."""
    with pytest.raises(PlanError, match="has no effect on a scalar function"):
        session.register_function("f", lambda v: v, batch_format="numpy")


def test_batch_format_on_a_per_row_table_function_is_rejected(session):
    with pytest.raises(PlanError, match="per-row table function"):
        session.register_function(
            "f", lambda row: row, table=True, per_row=True, batch_format="numpy"
        )


def test_the_module_level_helper_validates_too():
    """`bt.register_function` forwards to a session, so the guard must not be session-only."""
    with pytest.raises(PlanError, match="map_groups"):
        bt.register_function("module_level_agg", lambda v: v, aggregate=True)


# --- what still works -------------------------------------------------------------------


def test_a_scalar_function_still_registers_and_runs(session):
    session.register_function("dbl", lambda a: pc.multiply(a, 2), result_type="int64")
    assert session.sql("SELECT dbl(x) AS y FROM t").to_pydict() == {"y": [2, 4, 6]}


def test_a_per_row_scalar_function_still_runs(session):
    session.register_function("tri", lambda v: v * 3, result_type="int64", vectorized=False)
    assert session.sql("SELECT tri(x) AS y FROM t").to_pydict() == {"y": [3, 6, 9]}


def test_a_table_function_still_forwards_its_options(session):
    session.register_function("ident", lambda b: b, table=True, batch_size=2, num_workers=2)
    assert session.sql("SELECT * FROM ident(t)").to_pydict() == {"x": [1, 2, 3]}


def test_a_per_row_table_function_now_forwards_its_options(session):
    """`rf.config` used to be dropped for this form, so `batch_size` did nothing at all."""
    session.register_function(
        "tripled", lambda row: {"x": row["x"] * 3}, table=True, per_row=True, output_columns=["x"]
    )
    assert _stage(session.sql("SELECT * FROM tripled(t)")).batch_size is None
    session.register_function(
        "chunked",
        lambda row: {"x": row["x"] * 3},
        table=True,
        per_row=True,
        output_columns=["x"],
        batch_size=2,
    )
    assert _stage(session.sql("SELECT * FROM chunked(t)")).batch_size == 2
    assert session.sql("SELECT * FROM chunked(t)").to_pydict() == {"x": [3, 6, 9]}


def _stage(ds):
    """The one `MapBatches` a registered table function lowers to."""
    from batcher.plan.logical import MapBatches
    from batcher.plan.visitor import children

    def walk(node):
        found = [node] if isinstance(node, MapBatches) else []
        for child in children(node):
            found.extend(walk(child))
        return found

    stages = walk(ds._plan)
    assert len(stages) == 1
    return stages[0]


# --- the same check on the decorator ----------------------------------------------------


def test_the_udf_decorator_rejects_an_unknown_option_where_it_was_written():
    """`**config` reaches `map_batches` only when the transform is finally applied, so a
    misspelling used to surface as a `TypeError` naming a method the user never called, at
    whatever line applied it."""
    with pytest.raises(PlanError, match=r"@udf.*output_column"):
        bt.udf(output_column=["y"])


def test_the_decorator_checks_the_per_row_target_instead():
    with pytest.raises(PlanError, match="per-row callback"):
        bt.udf(per_row=True, batch_format="numpy")


def test_a_valid_decorator_option_still_works():
    @bt.udf(concurrency=2)
    def double(batch):
        return batch.set_column(0, "x", pc.multiply(batch.column("x"), 2))

    assert double(bt.from_pydict({"x": [1, 2]})).to_pydict() == {"x": [2, 4]}


# --- the same shape on a config bag -----------------------------------------------------


def test_a_misspelled_tenant_override_names_the_caller_and_the_settings():
    """`dataclasses.replace` refuses already, but as a `TypeError` naming
    `TenantConfig.__init__()` — a class the user did not mention, from a call they did not
    make."""
    from batcher._internal.errors import ConfigError

    with (
        pytest.raises(ConfigError, match=r"tenant\(\).*nonsense_option"),
        bt.tenant("t", nonsense_option=5),
    ):
        pass


def test_a_real_tenant_override_still_applies():
    with bt.tenant("analytics", max_concurrent_queries=4) as cfg:
        assert cfg.tenant.tenant_id == "analytics"
        assert cfg.tenant.max_concurrent_queries == 4


def test_a_misspelled_config_section_names_the_sections():
    from batcher._internal.errors import ConfigError

    with pytest.raises(ConfigError, match=r"Config\.replace\(\).*nonsense"):
        bt.active_config().replace(nonsense=1)


def test_replacing_a_real_section_still_works():
    from batcher.config import Config, ExecutionConfig

    replaced = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
    assert replaced.execution.morsel_rows == 4096
