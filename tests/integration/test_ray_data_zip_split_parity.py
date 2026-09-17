"""`Dataset.zip` and `Dataset.split` against Ray Data 2.58, the engine they are named after.

Ray Data pairs and cuts rows by their position in the dataset's block order. Batcher has no such
order, so the port numbers each input where it is read (``with_row_index``) and passes that
number as `order_by`, which is exactly what the `batcher.migrate` codemod emits. With that in
place the two must agree on rows, on column names (Ray's ``_1``/``_2`` suffixes), on part sizes
(including ``equal=True`` dropping the remainder), and on the refusal of a row-count mismatch.

The DuckDB side of the same claims is `tests/differential/test_diff_zip_split_transpose.py`.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _ray():
    ray = pytest.importorskip("ray", reason="ray not installed")
    pytest.importorskip("ray.data", reason="ray.data not installed")
    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        configure_logging=False,
        log_to_driver=False,
    )


def _items(n: int, scale: int, name: str) -> list[dict]:
    return [{"id": i, name: i * scale} for i in range(n)]


def _ported(items: list[dict]) -> bt.Dataset:
    """The codemod's port of a Ray source: number the rows in source order where they are read."""
    return bt.from_pylist(items).with_row_index("_row")


def test_zip_matches_ray_data():
    import ray.data

    a, b, c = _items(7, 10, "x"), _items(7, -1, "y"), _items(7, 1, "x")
    theirs = ray.data.from_items(a).zip(ray.data.from_items(b), ray.data.from_items(c)).take_all()
    ours = (
        _ported(a)
        .zip(_ported(b), _ported(c), order_by="_row")
        .drop("_row", "_row_1", "_row_2")
        .to_pylist()
    )
    assert ours == theirs


def test_zip_refuses_a_row_count_mismatch_like_ray_data():
    import ray.data

    with pytest.raises(ValueError, match="different number of rows"):
        ray.data.from_items(_items(5, 1, "x")).zip(
            ray.data.from_items(_items(4, 1, "y"))
        ).take_all()
    with pytest.raises(bt.PlanError, match="same number of rows"):
        _ported(_items(5, 1, "x")).zip(_ported(_items(4, 1, "y")), order_by="_row")


@pytest.mark.parametrize("equal", [False, True])
@pytest.mark.parametrize(("rows", "parts"), [(10, 3), (2, 3), (9, 3), (1, 2)])
def test_split_matches_ray_data(rows, parts, equal):
    import ray.data

    theirs = [part.take_all() for part in ray.data.range(rows).split(parts, equal=equal)]
    ours = [
        part.drop("_row").to_pylist()
        for part in _ported([{"id": i} for i in range(rows)]).split(
            parts, order_by="_row", equal=equal
        )
    ]
    assert ours == theirs
