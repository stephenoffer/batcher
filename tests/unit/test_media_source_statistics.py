"""Media sources report statistics, and the distributed planner sizes tasks by bytes.

Three gaps that compounded into one: a media source reported no `statistics()` at all, its
splits carried no `rows`, and the map/scan task fan-out had no byte term. Together they
meant a directory of 200 MB videos was planned exactly like one of 4 KB thumbnails —
uncountable, unweighted, and sized purely by row count.

Every number involved is free: it comes from the file listing and a stat, which the
batching already performs.
"""

from __future__ import annotations

import io

import pytest

from batcher.io.formats.multimodal.images import ImageSource

Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.unit


def _png(pad: int = 0) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(buf, "PNG")
    return buf.getvalue() + b"\0" * pad


@pytest.fixture
def corpus(tmp_path):
    """Ten images, each padded to a known, near-identical size."""
    for i in range(10):
        (tmp_path / f"i{i}.png").write_bytes(_png(pad=10_000))
    return str(tmp_path)


def test_statistics_reports_an_exact_row_count(corpus) -> None:
    assert ImageSource(corpus).statistics().row_count == 10


def test_statistics_reports_the_real_byte_size(corpus) -> None:
    """Not a per-row prior: the estimator's width for a binary column is 36 bytes,
    which is wrong by orders of magnitude for exactly the corpora that matter."""
    import os

    stats = ImageSource(corpus).statistics()
    on_disk = sum(os.path.getsize(os.path.join(corpus, f)) for f in os.listdir(corpus))
    assert stats.byte_size == on_disk
    assert stats.byte_size > 100_000


def test_statistics_reports_an_exact_size_zone_map(corpus) -> None:
    """`size` bounds are the values themselves, so a `WHERE size > …` prunes exactly."""
    from batcher.plan.stats import Provenance

    size = ImageSource(corpus).statistics().columns["size"]

    assert size.provenance is Provenance.EXACT
    assert size.null_count == 0
    assert size.min <= size.max


def test_splits_carry_their_exact_row_count(corpus) -> None:
    """One row per file — knowable with no I/O, and what the planner reads to size tasks."""
    src = ImageSource(corpus, batch_files=3)
    splits = src.splits()

    assert [s.rows for s in splits] == [len(s.files) for s in splits]
    assert sum(s.rows for s in splits) == 10


def test_the_planner_can_now_count_a_media_source(corpus) -> None:
    """`_source_total_rows` returned None before, so the fan-out fell back to a blunt
    worker count and `_balance` weighted every split as 1."""
    from batcher.dist.executors.map import _source_total_rows

    assert _source_total_rows(ImageSource(corpus)) == 10


def test_task_fanout_follows_bytes_not_rows(corpus) -> None:
    """Ten rows would be one task by row count; their bytes say otherwise."""
    import dataclasses

    import batcher as bt
    from batcher.config import active_config, config_context
    from batcher.dist.executors.map import _adaptive_partition_count, _byte_partition_count

    stats = ImageSource(corpus).statistics()
    # A per-task budget of a quarter of the corpus must ask for ~4 tasks.
    budget = max(1, stats.byte_size // 4)
    cfg = active_config()
    scoped = cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, target_bytes_per_task=budget))

    src = ImageSource(corpus, batch_bytes=budget)
    plan = bt.read.images(corpus)._plan
    with config_context(scoped):
        assert _byte_partition_count(src, plan, 10) >= 4
        tasks = _adaptive_partition_count(src, plan, 1)
        # Ten rows are far under one row-target, so a row-only fan-out is 1. Bytes must
        # push it up — that is the whole change.
        assert tasks > 1
        # ...but never past the split count, which would be unreachable work. (The two
        # need not be equal: the byte-derived count and the split packing round
        # independently.)
        assert tasks <= len(src.splits())


def test_a_narrow_source_is_unaffected_by_the_byte_term(corpus) -> None:
    """The byte term takes a max with the row term, so it can only ever add tasks."""
    import batcher as bt
    from batcher.dist.executors.map import _byte_partition_count

    plan = bt.read.images(corpus)._plan
    # The default 256 MiB budget dwarfs this corpus, so bytes imply a single task.
    assert _byte_partition_count(ImageSource(corpus), plan, 10) == 1
