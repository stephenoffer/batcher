"""Golden-snapshot lock on every `Expr` node's `to_ir()` wire output.

The JSON IR is the Python<->Rust wire contract (CLAUDE.md invariant #8): the
``to_ir()`` dict a node emits must stay byte-for-byte stable, because the Rust
``serde`` enums deserialize exactly these shapes. This test builds one
representative instance of *every* ``Expr`` subclass (plus the special cases:
literal type-dispatch, the transparent ``Aliased`` wrapper, and ``AggExpr``) and
asserts the emitted IR equals a checked-in golden file.

It exists to make the declarative-node refactor safe: any change that alters a
node's serialized shape — intended or accidental — turns this test red. To
intentionally re-baseline (only when the wire contract genuinely changed, in the
same commit as the matching Rust change), run::

    BATCHER_REGEN_IR_SNAPSHOT=1 pytest tests/unit/test_ir_snapshot.py

and review the diff to ``ir_snapshot_golden.json``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import pytest

from batcher.plan.expr_ir.audio import AudioFunc
from batcher.plan.expr_ir.core import (
    AggExpr,
    Aliased,
    Binary,
    Cast,
    Coalesce,
    IsInf,
    IsNan,
    IsNotNull,
    IsNull,
    Lit,
    Math2Expr,
    MathExpr,
    Not,
)
from batcher.plan.expr_ir.func_nodes import (
    ConvertTimezone,
    DateFunc,
    DateOffset,
    DateTrunc,
    GeoFunc,
    ListBinary,
    ListContains,
    ListFilter,
    ListFunc,
    ListGet,
    ListPosition,
    ListSet,
    ListSimhash,
    ListSlice,
    ListTransform,
    ListZip,
    MakeTemporal,
    MapFunc,
    SpatialFunc,
    Strftime,
    StrFunc,
    Strptime,
    StructField,
    WindowBuckets,
    WindowStart,
)
from batcher.plan.expr_ir.image import ImageCrop, ImageFunc
from batcher.plan.expr_ir.namespaces.sequence import SeqFunc
from batcher.plan.expr_ir.nodes import (
    Array,
    Case,
    Col,
    Greatest,
    HashRows,
    Least,
    ListJoin,
    MakeStruct,
    NullIf,
    Sequence,
)
from batcher.plan.expr_ir.video import VideoFunc

GOLDEN = Path(__file__).parent / "data" / "ir_snapshot_golden.json"

_X = Col("x")
_Y = Col("y")
_ELEM = Binary("multiply", Col("x"), Lit(2))
_PRED = Binary("gt", Col("x"), Lit(0))


def _representatives() -> dict[str, Any]:
    """One representative per node type, mapped label -> emitted IR dict.

    Optional fields and omit-policies are exercised with both present and absent
    variants where they matter (``*_full`` / ``*_min`` labels).
    """
    nodes: dict[str, Any] = {
        # --- leaves & literals (literal type-dispatch lives in Lit.to_ir) -------
        "col": Col("x"),
        "lit_int": Lit(7),
        "lit_float": Lit(1.5),
        "lit_bool": Lit(True),
        "lit_str": Lit("hello"),
        "lit_date": Lit(dt.date(2021, 3, 15)),
        "lit_datetime": Lit(dt.datetime(2021, 3, 15, 13, 45, 30)),
        # --- core scalar nodes --------------------------------------------------
        "binary": Binary("add", _X, Lit(1)),
        "not": Not(Col("b")),
        "cast": Cast(_X, "int64"),
        "cast_try": Cast(_X, "int64", try_cast=True),
        "is_null": IsNull(_X),
        "is_not_null": IsNotNull(_X),
        "is_nan": IsNan(_X),
        "is_inf": IsInf(_X),
        "aliased": Aliased(_X, "y"),  # transparent: delegates to inner
        "math": MathExpr("abs", _X),
        # One variadic `geo` node per arity band, because the whole `ST_*` surface rides
        # one wire shape and the argument *list* is the only thing that varies: a change
        # to how it is emitted would move all 113 functions at once.
        "geo_unary": GeoFunc("st_area", [_X]),
        "geo_binary": GeoFunc("st_intersects", [_X, Lit("POINT(1 2)")]),
        "geo_ternary": GeoFunc("st_dwithin", [_X, Lit("POINT(1 2)"), Lit(5.0)]),
        # And one `spatial` node per arity band, for the same reason: the whole
        # rigid-body surface rides one wire shape, so a change to how the argument list
        # is emitted would move all 42 functions at once.
        "spatial_quat": SpatialFunc("quat_norm", [_X, _Y, Lit(0.0), Lit(1.0)]),
        "spatial_pose": SpatialFunc(
            "se3_transform_x",
            [
                _X,
                _Y,
                Lit(0.0),
                Lit(0.0),
                Lit(0.0),
                Lit(0.0),
                Lit(1.0),
                Lit(1.0),
                Lit(2.0),
                Lit(3.0),
            ],
        ),
        "math2": Math2Expr("pow", _X, Lit(2)),
        "coalesce": Coalesce([_X, Lit(0)]),
        # --- nodes.py leaves ----------------------------------------------------
        "case": Case([(_PRED, Lit(1))], Lit(0)),
        "nullif": NullIf(_X, Lit(0)),
        "greatest": Greatest([_X, _Y]),
        "least": Least([_X, _Y]),
        "array": Array([Lit(1), Lit(2)]),
        "hash": HashRows([_X, _Y], 7),
        "hash_default_seed": HashRows([_X], 0),
        "sequence": Sequence(Lit(1), Lit(10), Lit(2)),
        "make_struct": MakeStruct([("a", _X), ("b", Lit(1))]),
        "list_join": ListJoin(Col("a"), ","),
        "list_simhash": ListSimhash(Col("v"), 64),
        "list_simhash_seeded": ListSimhash(Col("v"), 64, 3),
        # --- string functions ---------------------------------------------------
        "str_simple": StrFunc("upper", Col("s")),
        "str_contains": StrFunc("contains", Col("s"), pattern="x"),
        "str_substr": StrFunc("substr", Col("s"), start=1, length=3),
        "str_replace": StrFunc("replace", Col("s"), pattern="a", replacement="b"),
        # --- date/time ----------------------------------------------------------
        "date_func": DateFunc("year", Col("d")),
        "date_trunc": DateTrunc(Col("d"), "month"),
        "make_temporal": MakeTemporal("make_date", [Col("y"), Col("m"), Col("d")]),
        "convert_timezone": ConvertTimezone(Col("d"), "UTC", "America/New_York"),
        "date_offset_full": DateOffset(Col("d"), 1, 2, 3),
        "date_offset_partial": DateOffset(Col("d"), 0, 5, 0),  # omit zero months/micros
        "strftime": Strftime(Col("d"), "%Y-%m-%d"),
        "strptime": Strptime(Col("s"), "%Y-%m-%d"),
        "window_start_min": WindowStart(Col("d"), 1000),
        "window_start_origin": WindowStart(Col("d"), 1000, 500),
        "window_buckets": WindowBuckets(Col("d"), 1000, 500),
        # --- list / collection --------------------------------------------------
        "list_func": ListFunc("sum", Col("a")),
        "list_binary": ListBinary("dot", Col("a"), Col("b")),
        "list_set": ListSet("array_intersect", Col("a"), Col("b")),
        "list_zip": ListZip("list_add", Col("a"), Col("b")),
        "list_transform": ListTransform(Col("a"), _ELEM),
        "list_filter": ListFilter(Col("a"), _PRED),
        "list_get": ListGet(Col("a"), -1),
        "list_contains": ListContains(Col("a"), 5),
        "list_position": ListPosition(Col("a"), 5),
        "list_slice_min": ListSlice(Col("a"), 1),
        "list_slice_full": ListSlice(Col("a"), 1, 2),
        # --- struct / map -------------------------------------------------------
        "struct_field": StructField(Col("s"), "field"),
        "map_simple": MapFunc("map_keys", Col("m")),
        "map_element_at": MapFunc("element_at", Col("m"), key="k"),
        # --- multimodal ---------------------------------------------------------
        "image_simple": ImageFunc("decode", Col("img")),
        "image_to_tensor": ImageFunc("to_tensor", Col("img"), width=224, height=224),
        # A crop's window is four *sub-expressions*, not four scalars, so it is the one
        # multimodal node whose wire shape can drift without any `ImageFunc` case moving.
        "image_crop": ImageCrop(Col("img"), Col("x"), Col("y"), Lit(64), Lit(64)),
        # The parameterized forms, one per group of optional slots. A bare `decode` was
        # the only representative of each of these three nodes, so every optional slot
        # they own was unlocked -- and `_EXPR_REBUILD` was silently dropping all of them
        # (`.image.encode` came back with no format, `.audio.mel_spectrogram` with no
        # filterbank sizes). A golden that only ever saw `decode` could not see it.
        "image_tensor_f32": ImageFunc(
            "to_tensor_f32",
            Col("img"),
            width=224,
            height=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            channels_first=True,
        ),
        "image_encode": ImageFunc("encode", Col("img"), format="jpeg"),
        "image_letterbox": ImageFunc("letterbox", Col("img"), width=640, height=640, fill=114),
        "audio": AudioFunc("decode", Col("clip")),
        "audio_mel": AudioFunc(
            "mel_spectrogram", Col("clip"), rate=16000, n_fft=400, hop_length=160, n_mels=80
        ),
        "audio_mfcc": AudioFunc(
            "mfcc", Col("clip"), rate=16000, n_fft=400, hop_length=160, n_mels=80, n_mfcc=13
        ),
        "audio_trim": AudioFunc("trim_silence", Col("clip"), threshold_db=-40),
        "video": VideoFunc("decode", Col("clip")),
        "video_frames": VideoFunc("frames", Col("clip"), num_frames=8, width=224, height=224),
        # `second` is a *child*, not a scalar: the row names the moment it wants a still
        # from. That made it the one media slot a rebuild could drop and a column walk
        # could miss, so it is represented explicitly.
        "video_frame_at": VideoFunc("frame_at", Col("clip"), width=256, second=Col("ts")),
        # --- sequence -----------------------------------------------------------
        # Three representatives, because the node's eight argument slots are used by
        # disjoint subsets of the family and only a bare op proves the rest stay off the
        # wire: a `reverse_complement` that started emitting `"k": null` would break the
        # Rust round-trip while every Python test still passed.
        "seq_bare": SeqFunc("reverse_complement", Col("dna")),
        "seq_kmers": SeqFunc("minimizers", Col("dna"), k=21, window=10),
        "seq_translate": SeqFunc("translate", Col("dna"), frame=1, to_stop=True),
    }
    out = {label: node.to_ir() for label, node in nodes.items()}
    # AggExpr is not an Expr: its to_ir takes an output alias.
    out["agg_unary"] = AggExpr("sum", _X).to_ir("total")
    out["agg_param"] = AggExpr("quantile", _X, param=0.5).to_ir("p50")
    out["agg_binary"] = AggExpr("corr", _X, input2=_Y).to_ir("r")
    return out


def test_ir_snapshot() -> None:
    current = _representatives()
    if os.environ.get("BATCHER_REGEN_IR_SNAPSHOT"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        pytest.skip("regenerated IR snapshot golden")
    assert GOLDEN.exists(), "run BATCHER_REGEN_IR_SNAPSHOT=1 pytest to create the golden"
    golden = json.loads(GOLDEN.read_text())
    assert current == golden, "to_ir() wire output drifted from the golden snapshot"


def _all_subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def _tags_in(obj: Any, seen: set[str]) -> None:
    if isinstance(obj, dict):
        e = obj.get("e")
        if isinstance(e, str):
            seen.add(e)
        for value in obj.values():
            _tags_in(value, seen)
    elif isinstance(obj, list):
        for value in obj:
            _tags_in(value, seen)


def test_every_ir_node_tag_is_snapshotted() -> None:
    """Every `IRNode` subclass's wire tag appears in the golden.

    This makes the snapshot self-maintaining: add a new declarative node and forget
    to add a representative above, and this test fails — so a node type cannot slip
    past the byte-for-byte wire-contract lock as the engine grows toward thousands of
    expressions.
    """
    import batcher  # noqa: F401  — ensure every node module is imported for __subclasses__
    from batcher.plan.expr_ir.node_base import IRNode

    node_tags = {
        sub.tag for sub in _all_subclasses(IRNode) if isinstance(getattr(sub, "tag", None), str)
    }
    golden_tags: set[str] = set()
    _tags_in(json.loads(GOLDEN.read_text()), golden_tags)
    missing = sorted(node_tags - golden_tags)
    assert not missing, (
        f"IRNode subclasses whose wire tag has no representative in the snapshot: {missing}. "
        "Add one to _representatives() and re-baseline with BATCHER_REGEN_IR_SNAPSHOT=1."
    )
