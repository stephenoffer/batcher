"""Media sources prune files at plan time from listing-only metadata.

`uri`, `size` and `mime` are known before any payload byte is read, so a predicate over
just those decides which files are worth opening. On a media corpus that is the difference
between reading a directory listing and reading a terabyte.

Pruning is invisible in a result — the engine's `Filter` produces the same rows either way
— so the tests that matter are the ones that would catch pruning too *much* (rows lost)
and pruning when it must not (a predicate over a column a listing cannot answer).
"""

from __future__ import annotations

import pytest

from batcher.io.formats.multimodal._pruning import prunable_columns, prune_files
from batcher.io.formats.multimodal.images import ImageSource

pytestmark = pytest.mark.unit

_PNG = bytes.fromhex("89504e470d0a1a0a") + b"\0" * 200


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


def _int(value: int) -> dict:
    return {"e": "lit", "value": {"int": value}}


def _str(value: str) -> dict:
    return {"e": "lit", "value": {"str": value}}


def _cmp(op: str, left: dict, right: dict) -> dict:
    return {"e": "binary", "op": op, "left": left, "right": right}


@pytest.fixture
def corpus(tmp_path):
    """Six small images and three large ones."""
    for i in range(6):
        (tmp_path / f"s{i}.png").write_bytes(_PNG)
    for i in range(3):
        (tmp_path / f"b{i}.png").write_bytes(_PNG + b"\0" * 50_000)
    return str(tmp_path)


# ---- which predicates are decidable from a listing --------------------------


def test_listing_columns_are_prunable() -> None:
    assert prunable_columns(_cmp("gt", _col("size"), _int(10)))
    assert prunable_columns(_cmp("eq", _col("mime"), _str("image/png")))
    assert prunable_columns(_cmp("eq", _col("uri"), _str("a.png")))


def test_a_column_the_listing_cannot_answer_is_not_prunable() -> None:
    """`width` needs the file header — deciding it here would be a guess."""
    assert not prunable_columns(_cmp("gt", _col("width"), _int(5)))


def test_a_mixed_predicate_is_not_prunable() -> None:
    """One undecidable conjunct makes the whole expression undecidable."""
    mixed = {
        "e": "binary",
        "op": "and",
        "left": _cmp("gt", _col("size"), _int(10)),
        "right": _cmp("gt", _col("width"), _int(5)),
    }
    assert not prunable_columns(mixed)


def test_no_predicate_is_not_prunable() -> None:
    assert not prunable_columns(None)


def test_prune_files_returns_none_when_it_cannot_decide() -> None:
    """None means 'read everything' — always the safe answer."""
    assert prune_files(["a.png"], [1], None) is None
    assert prune_files(["a.png"], [1], _cmp("gt", _col("width"), _int(5))) is None
    assert prune_files([], [], _cmp("gt", _col("size"), _int(5))) is None


# ---- the source wiring ------------------------------------------------------


def _split_files(src, **kwargs) -> list[str]:
    return sorted(f.rsplit("/", 1)[-1] for s in src.splits(**kwargs) for f in s.files)


def test_a_size_predicate_drops_the_files_it_excludes(corpus) -> None:
    src = ImageSource(corpus, batch_files=2)
    assert _split_files(src, predicate=_cmp("gt", _col("size"), _int(1000))) == [
        "b0.png",
        "b1.png",
        "b2.png",
    ]


def test_a_predicate_matching_nothing_yields_no_splits(corpus) -> None:
    src = ImageSource(corpus, batch_files=2)
    assert _split_files(src, predicate=_cmp("gt", _col("size"), _int(10**9))) == []


def test_a_predicate_matching_everything_keeps_every_file(corpus) -> None:
    src = ImageSource(corpus, batch_files=2)
    assert len(_split_files(src, predicate=_cmp("eq", _col("mime"), _str("image/png")))) == 9


def test_an_undecidable_predicate_keeps_every_file(corpus) -> None:
    """Pruning wrongly here would silently lose rows; reading too much is merely slow."""
    src = ImageSource(corpus, batch_files=2)
    assert len(_split_files(src, predicate=_cmp("gt", _col("width"), _int(5)))) == 9


def test_no_predicate_keeps_every_file(corpus) -> None:
    src = ImageSource(corpus, batch_files=2)
    assert len(_split_files(src)) == 9
    assert len(_split_files(src, predicate=None)) == 9


def test_pruning_agrees_with_filtering_after_the_fact(corpus) -> None:
    """The soundness property: pushdown must not change which rows a query returns."""
    import pyarrow.compute as pc

    import batcher as bt
    from batcher.plan.expr_ir import col

    ds = bt.read.images(corpus)
    pushed = ds.filter(col("size") > 1000).select("uri", "size").collect()
    everything = ds.select("uri", "size").collect()
    manual = everything.filter(pc.greater(everything.column("size"), 1000))

    assert sorted(pushed.column("uri").to_pylist()) == sorted(manual.column("uri").to_pylist())
