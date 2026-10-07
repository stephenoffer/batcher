"""The `Literal` option aliases equal the vocabularies the runtime validators accept.

`batcher.config.option_types` types the closed keywords on the public signatures, and the
runtime checks keep their own accepted sets. These tests hold each alias to its validator,
so widening a validator without the alias (or the reverse) fails here instead of shipping a
signature that rejects a valid call or advertises an invalid one.
"""

from __future__ import annotations

import inspect
import typing

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.config import option_types


def _runtime_sets() -> dict[str, set[str]]:
    """Each alias's name -> the set its runtime validator accepts, read from that validator."""
    from batcher.api.catalog.catalog import TABLE_WRITE_MODES
    from batcher.api.dataset._build.combine import _UPDATE_HOWS
    from batcher.api.dataset._build.conform import _EXTRA
    from batcher.api.dataset._build.core import _DISTINCT_KEEPS
    from batcher.api.dataset._build.join import _VALIDATE, JOIN_HOWS
    from batcher.api.dataset._export import DASK_POLICIES, HF_MODES, NULL_MODES
    from batcher.api.dataset._nulls import _FILL_AGG_STRATEGIES, _FILL_ORDERED_STRATEGIES
    from batcher.api.dataset._udf.build import _ROW_FORMATS
    from batcher.api.session.combine import _HOWS
    from batcher.api.terminal.core import _BACKENDS
    from batcher.interop.arrays import DTYPE_BACKENDS
    from batcher.interop.formats import FORMATS
    from batcher.plan.expr_rewrite.over import MAPPING_STRATEGIES
    from batcher.plan.functions.aggregate_semantics import NAN_POLICIES
    from batcher.plan.ir_tags import QUANTILE_INTERPOLATIONS
    from batcher.plan.logical.join import ASOF_DIRECTIONS
    from batcher.plan.streaming.spec import OutputMode

    return {
        # `Dataset.join` reads "outer" as "full" and routes "cross" to `cross_join` before
        # `build_join` validates the rest against `JOIN_HOWS`.
        "JoinHow": {*JOIN_HOWS, "outer", "cross"},
        "JoinValidate": set(_VALIDATE),
        "AsofDirection": set(ASOF_DIRECTIONS),
        "BatchFormat": set(FORMATS),
        "RowBatchFormat": set(_ROW_FORMATS),
        "QuantileInterpolation": set(QUANTILE_INTERPOLATIONS),
        "NanPolicy": set(NAN_POLICIES),
        # `build_fill_null_strategy` handles "zero" inline before the two tables.
        "FillStrategy": {"zero", *_FILL_AGG_STRATEGIES, *_FILL_ORDERED_STRATEGIES},
        "DistinctKeep": set(_DISTINCT_KEEPS),
        "UpdateHow": set(_UPDATE_HOWS),
        "ExtraColumns": set(_EXTRA),
        "NumpyNulls": set(NULL_MODES),
        "DtypeBackend": set(DTYPE_BACKENDS),
        "DaskMaterialize": set(DASK_POLICIES),
        "HuggingFaceMode": set(HF_MODES),
        "Backend": set(_BACKENDS),
        "OutputModeName": set(OutputMode._ALL),
        "TableWriteMode": set(TABLE_WRITE_MODES),
        "ConcatHow": set(_HOWS),
        "MappingStrategy": set(MAPPING_STRATEGIES),
    }


# `drop_nulls(how=...)` is checked inline in `Dataset.drop_nulls` with no named set, so it is
# held to its behaviour below instead of to a constant.
_BEHAVIOURAL = {"DropNullsHow"}


def test_every_alias_is_checked() -> None:
    """A new alias must be added to a check here, or nothing holds it to the runtime."""
    assert set(option_types.__all__) == set(_runtime_sets()) | _BEHAVIOURAL


@pytest.mark.parametrize("name", sorted(_runtime_sets()))
def test_alias_equals_the_validator_set(name: str) -> None:
    alias = getattr(option_types, name)
    assert set(typing.get_args(alias)) == _runtime_sets()[name]


def test_drop_nulls_how_matches_behaviour() -> None:
    ds = bt.from_pydict({"a": [1, None], "b": [None, None]})
    for how in typing.get_args(option_types.DropNullsHow):
        ds.drop_nulls(how=how)  # every advertised value builds a plan
    with pytest.raises(PlanError, match="how must be 'any' or 'all'"):
        ds.drop_nulls(how="some")  # type: ignore[arg-type]


def test_every_join_how_builds_a_plan() -> None:
    """The two spellings `Dataset.join` handles itself really are accepted end to end."""
    left = bt.from_pydict({"k": [1, 2]})
    right = bt.from_pydict({"k": [2, 3]})
    for how in typing.get_args(option_types.JoinHow):
        on = None if how == "cross" else "k"
        joined = left.join(right, on=on, how=how) if on else left.join(right, how=how)
        assert isinstance(joined, bt.Dataset)


@pytest.mark.parametrize(
    ("method", "param", "alias"),
    [
        (bt.Dataset.join, "how", "JoinHow"),
        (bt.Dataset.join, "validate", "JoinValidate"),
        (bt.Dataset.map_batches, "batch_format", "BatchFormat"),
        (bt.Dataset.iter_batches, "batch_format", "BatchFormat"),
        (bt.Dataset.map, "batch_format", "RowBatchFormat"),
        (bt.Dataset.fill_null, "strategy", "FillStrategy"),
        (bt.Dataset.quantile, "interpolation", "QuantileInterpolation"),
        (bt.Dataset.collect, "backend", "Backend"),
        (bt.Expr.quantile, "interpolation", "QuantileInterpolation"),
        (bt.Expr.max, "nan_policy", "NanPolicy"),
        (bt.concat, "how", "ConcatHow"),
    ],
)
def test_public_signature_uses_the_alias(method: object, param: str, alias: str) -> None:
    """The alias is what the signature resolves to, so an editor offers the vocabulary."""
    annotation = inspect.signature(method).parameters[param].annotation
    # Only this parameter's annotation is evaluated: others name `TYPE_CHECKING` imports.
    got = eval(annotation, method.__globals__)  # type: ignore[attr-defined]
    expected = getattr(option_types, alias)
    assert got == expected or expected in typing.get_args(got), (param, got)
