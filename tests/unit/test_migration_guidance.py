"""The traceback is the documentation: absent ecosystem APIs must explain themselves.

A migrant from pandas, Polars, or PySpark types the method they already know and reads
the traceback. `Dataset`, `Expr`, and `GroupBy` each route a failed attribute lookup
through a curated redirect table, so ``ds.pivot_table``, ``col('x').map_elements``, and
``gb.transform`` come back with the Batcher spelling instead of a bare `AttributeError`.

These tests hold three lines: every table entry actually fires (it is not shadowed by a
real method, which would make the guidance dead), the message a migrant sees carries the
curated guidance, and `__getattr__` never decorates a dunder probe (or copy/pickle and
``hasattr`` break).
"""

from __future__ import annotations

import copy
import pickle

import pytest

import batcher as bt
from batcher.api.dataset.compat.guidance._dataset_table import DATASET_UNSUPPORTED
from batcher.api.dataset.compat.guidance._groupby_table import GROUPBY_UNSUPPORTED
from batcher.api.session.onboarding import TOP_LEVEL_UNSUPPORTED
from batcher.plan.expr_ir.compat.guidance import (
    DT_UNSUPPORTED,
    EXPR_UNSUPPORTED,
    LIST_UNSUPPORTED,
    STR_UNSUPPORTED,
)


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2, 3], "k": ["a", "b", "a"]})


# --- every entry fires with its curated guidance ------------------------------------


@pytest.mark.parametrize("name", sorted(DATASET_UNSUPPORTED))
def test_dataset_absent_api_carries_guidance(ds: bt.Dataset, name: str) -> None:
    with pytest.raises(AttributeError) as exc:
        getattr(ds, name)
    assert DATASET_UNSUPPORTED[name] in str(exc.value)


@pytest.mark.parametrize("name", sorted(EXPR_UNSUPPORTED))
def test_expr_absent_api_carries_guidance(name: str) -> None:
    with pytest.raises(AttributeError) as exc:
        getattr(bt.col("x"), name)
    assert EXPR_UNSUPPORTED[name] in str(exc.value)


@pytest.mark.parametrize("name", sorted(GROUPBY_UNSUPPORTED))
def test_groupby_absent_api_carries_guidance(ds: bt.Dataset, name: str) -> None:
    with pytest.raises(AttributeError) as exc:
        getattr(ds.group_by("k"), name)
    assert GROUPBY_UNSUPPORTED[name] in str(exc.value)


@pytest.mark.parametrize("name", sorted(TOP_LEVEL_UNSUPPORTED))
def test_top_level_absent_api_carries_guidance(name: str) -> None:
    with pytest.raises(AttributeError) as exc:
        getattr(bt, name)
    assert TOP_LEVEL_UNSUPPORTED[name] in str(exc.value)


@pytest.mark.parametrize("name", sorted(STR_UNSUPPORTED))
def test_str_accessor_absent_api_carries_guidance(name: str) -> None:
    with pytest.raises(AttributeError) as exc:
        getattr(bt.col("x").str, name)
    assert STR_UNSUPPORTED[name] in str(exc.value)


@pytest.mark.parametrize("name", sorted(DT_UNSUPPORTED))
def test_dt_accessor_absent_api_carries_guidance(name: str) -> None:
    with pytest.raises(AttributeError) as exc:
        getattr(bt.col("x").dt, name)
    assert DT_UNSUPPORTED[name] in str(exc.value)


@pytest.mark.parametrize("name", sorted(LIST_UNSUPPORTED))
def test_list_accessor_absent_api_carries_guidance(name: str) -> None:
    with pytest.raises(AttributeError) as exc:
        getattr(bt.col("x").list, name)
    assert LIST_UNSUPPORTED[name] in str(exc.value)


def test_no_accessor_entry_shadows_a_real_method() -> None:
    str_real = set(dir(bt.col("x").str))
    dt_real = set(dir(bt.col("x").dt))
    list_real = set(dir(bt.col("x").list))
    str_dead = sorted(k for k in STR_UNSUPPORTED if k in str_real)
    dt_dead = sorted(k for k in DT_UNSUPPORTED if k in dt_real)
    list_dead = sorted(k for k in LIST_UNSUPPORTED if k in list_real)
    assert str_dead == [], f"shadowed .str redirects: {str_dead}"
    assert dt_dead == [], f"shadowed .dt redirects: {dt_dead}"
    assert list_dead == [], f"shadowed .list redirects: {list_dead}"


def test_accessor_typo_suggests_the_real_method() -> None:
    with pytest.raises(AttributeError, match="Did you mean 'upper'"):
        _ = bt.col("x").str.uper
    with pytest.raises(AttributeError, match="Did you mean 'year'"):
        _ = bt.col("x").dt.yaer


# --- no dead entries: a redirect that shadows a real method never fires --------------


def test_no_dataset_entry_shadows_a_real_method(ds: bt.Dataset) -> None:
    real = set(dir(ds))
    dead = sorted(k for k in DATASET_UNSUPPORTED if k in real)
    assert dead == [], f"these redirects are shadowed by a real Dataset method: {dead}"


def test_no_expr_entry_shadows_a_real_method() -> None:
    real = set(dir(bt.col("x")))
    dead = sorted(k for k in EXPR_UNSUPPORTED if k in real)
    assert dead == [], f"these redirects are shadowed by a real Expr method: {dead}"


def test_no_groupby_entry_shadows_a_real_method(ds: bt.Dataset) -> None:
    real = set(dir(ds.group_by("k")))
    dead = sorted(k for k in GROUPBY_UNSUPPORTED if k in real)
    assert dead == [], f"these redirects are shadowed by a real GroupBy method: {dead}"


def test_no_top_level_entry_shadows_a_real_name() -> None:
    real = set(dir(bt))
    dead = sorted(k for k in TOP_LEVEL_UNSUPPORTED if k in real)
    assert dead == [], f"these redirects are shadowed by a real top-level name: {dead}"


def test_top_level_typo_suggests_the_real_name() -> None:
    with pytest.raises(AttributeError, match="Did you mean 'read'"):
        _ = bt.reed


def test_top_level_getattr_leaves_dunders_plain() -> None:
    # Import machinery and IPython probe dunders/private names; they must fail plainly.
    assert not hasattr(bt, "_ipython_canary_method_should_not_exist_")
    assert not hasattr(bt, "__wrapped__")
    # Polars renamed the array accessor; typing .arr points at .list.
    with pytest.raises(AttributeError, match=r"\.list"):
        _ = bt.col("x").arr


# --- the guidance names a real replacement, spot-checked -----------------------------


@pytest.mark.parametrize(
    ("obj", "attr", "expected"),
    [
        ("dataset", "pivot_table", "ds.pivot"),
        ("dataset", "groupBy", "ds.group_by"),
        ("dataset", "iterrows", "iter_rows"),
        ("dataset", "idxmax", "arg_max"),
        ("dataset", "unionAll", "ds.union"),
        ("dataset", "to_sql", "ds.write.sql"),
        ("expr", "map_elements", "map_batches"),
        ("expr", "clip_lower", "clip_min"),
        ("expr", "value_counts", "ds.value_counts"),
        ("expr", "argmax", "arg_max"),
        ("groupby", "transform", "ds.window"),
        ("groupby", "get_group", "ds.filter"),
        ("groupby", "cumcount", "row_number"),
    ],
)
def test_guidance_names_the_replacement(ds: bt.Dataset, obj: str, attr: str, expected: str) -> None:
    target = {"dataset": ds, "expr": bt.col("x"), "groupby": ds.group_by("k")}[obj]
    with pytest.raises(AttributeError, match=expected.replace(".", r"\.").replace("(", r"\(")):
        getattr(target, attr)


# --- a near miss on a real method still suggests it ----------------------------------


def test_expr_typo_suggests_the_real_method() -> None:
    with pytest.raises(AttributeError, match="Did you mean 'mean'"):
        _ = bt.col("x").meen


def test_groupby_typo_suggests_the_real_method(ds: bt.Dataset) -> None:
    with pytest.raises(AttributeError, match="Did you mean 'mean'"):
        _ = ds.group_by("k").maen


# --- dunder safety: copy/pickle/hasattr must not be decorated ------------------------


def test_expr_getattr_leaves_dunders_plain() -> None:
    e = bt.col("x") * 2
    assert copy.deepcopy(e).to_ir() == e.to_ir()
    assert pickle.loads(pickle.dumps(e)).to_ir() == e.to_ir()
    assert not hasattr(e, "definitely_not_here")


def test_groupby_getattr_leaves_dunders_plain(ds: bt.Dataset) -> None:
    gb = ds.group_by("k")
    assert not hasattr(gb, "definitely_not_here")
    # A real aggregate still resolves through normal lookup, unshadowed by __getattr__.
    out = gb.agg(total=bt.col("x").sum()).sort("k").to_pydict()
    assert out == {"k": ["a", "b"], "total": [4, 2]}
