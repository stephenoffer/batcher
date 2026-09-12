"""A media source stats each file once, not once per planning call.

`_file_sizes` answered from `_size_cache` when a completed `read()` had filled it and
otherwise probed -- and threw the probe away. Sizes are asked for by *planning*, though,
not only by reading: `splits()` needs them for its byte bound, and a distributed query
calls it at least twice, once in `_adaptive_partition_count` and once in
`partition_descriptors`. Each of those was a full stat storm.

Measured on a 10,000-JPEG S3 corpus, warm, best of three: the driver spent 6,139 ms in the
sizing call and 4,169 ms in the descriptor build of an 11.6 s query. With the probe cached
those are 4 ms and 4 ms, and the query is 687 ms -- 15x, none of it on the cluster.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.multimodal.images import ImageSource

pytestmark = pytest.mark.unit


@pytest.fixture
def corpus(tmp_path):
    """Ten tiny files with an image suffix. Only their sizes are under test."""
    for i in range(10):
        (tmp_path / f"{i:03d}.jpg").write_bytes(b"x" * (100 + i))
    return tmp_path


def _counting_source(corpus, monkeypatch) -> tuple[ImageSource, list[str]]:
    """An `ImageSource` over `corpus` whose every stat-probed file is recorded.

    The probe is intercepted at `media.probe_sizes` rather than on the filesystem object,
    whose attributes are read-only -- and it is the right seam anyway: it is the one place
    a size is fetched rather than remembered.
    """
    from batcher.io.formats.multimodal import media

    source = ImageSource(str(corpus))
    stats: list[str] = []
    real = media.probe_sizes

    def counting(files, size_of):
        stats.extend(files)
        return real(files, size_of)

    monkeypatch.setattr(media, "probe_sizes", counting)
    return source, stats


def test_the_first_planning_call_stats_every_file_once(corpus, monkeypatch) -> None:
    """The positive control: without this, every assertion below is vacuous."""
    source, stats = _counting_source(corpus, monkeypatch)

    source.splits()

    assert sorted(stats) == sorted(set(stats)), "no file is stat-ed twice"
    assert len(stats) == 10


def test_a_second_planning_call_stats_nothing(corpus, monkeypatch) -> None:
    """The defect. A distributed query plans twice; it paid the storm twice."""
    source, stats = _counting_source(corpus, monkeypatch)

    source.splits()
    stats.clear()
    source.splits(target_size=1 << 20)  # a different byte bound: not the memoized chunking

    assert stats == []


def test_the_cached_sizes_are_the_probed_sizes(corpus, monkeypatch) -> None:
    """A cache that returns a different answer would repartition wrongly, silently."""
    source, _ = _counting_source(corpus, monkeypatch)
    files = source._files()

    first = source._file_sizes(files)
    second = source._file_sizes(files)

    assert first == second == [100 + i for i in range(10)]
