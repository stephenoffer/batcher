"""What a media listing knows about a file before anything decodes it.

Three things this covers, each one a place the reader was answering less than the header
it had already parsed could tell it:

- **The suffix lists decide what the source can see at all.** A file whose extension is not
  listed is not "unsupported", it is invisible: the listing returns nothing and the error
  reads as an empty directory. A corpus of iPhone photographs (`.heic`) or WebRTC captures
  (`.opus`) hit exactly that.
- **A container with no video stream is ordinary, not corrupt.** `streams.video[0]` raised
  `IndexError` on an audio-only MP4, which the reader tolerates by nulling *every* metadata
  column — so an audio-only file was indistinguishable from a truncated one.
- **Codec, audio presence and image container are free.** All three come out of the header
  parse the reader already performs, and learning any of them otherwise meant opening every
  file a second time.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

import batcher as bt

pytestmark = pytest.mark.integration


def _png(width: int, height: int) -> bytes:
    """A real, complete PNG, so Pillow reports its format as well as its size."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _mp4(tmp_path: Path, *, video: bool, audio: bool) -> bytes:
    """An MP4 holding the streams asked for, written with PyAV.

    Both streams are declared **before** any packet is muxed. FFmpeg writes the container
    header from the stream list the first `mux` sees, so adding a second stream after
    encoding has begun is invalid — and it does not raise, it aborts the interpreter with
    a SIGFPE while writing the trailer.

    Every frame carries an explicit `pts` and `time_base`, and so does its stream: without
    them PyAV refuses with "Cannot rebase to zero time", a message that names neither the
    stream nor the cause.
    """
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from fractions import Fraction

    path = tmp_path / "fixture.mp4"
    with av.open(str(path), "w") as out:
        video_stream = None
        audio_stream = None
        if video:
            video_stream = out.add_stream("mpeg4", rate=10)
            video_stream.width, video_stream.height = 32, 32
            video_stream.pix_fmt = "yuv420p"
            video_stream.time_base = Fraction(1, 10)
        if audio:
            audio_stream = out.add_stream("aac", rate=16000)
            audio_stream.layout = "mono"
            audio_stream.time_base = Fraction(1, 16000)
        if video_stream is not None:
            for i in range(5):
                frame = av.VideoFrame(32, 32, "yuv420p")
                frame.pts = i
                frame.time_base = Fraction(1, 10)
                out.mux(video_stream.encode(frame))
            out.mux(video_stream.encode(None))
        if audio_stream is not None:
            pts = 0
            for _ in range(4):
                frame = av.AudioFrame.from_ndarray(
                    np.zeros((1, 1024), dtype="float32"), format="fltp", layout="mono"
                )
                frame.sample_rate = 16000
                frame.pts = pts
                frame.time_base = Fraction(1, 16000)
                pts += 1024
                for packet in audio_stream.encode(frame):
                    out.mux(packet)
            out.mux(audio_stream.encode(None))
    data = path.read_bytes()
    path.unlink()
    return data


def _rows(ds) -> dict[str, dict]:
    return {r["uri"].rsplit("/", 1)[-1]: r for r in ds.collect().to_pylist()}


def test_an_image_corpus_is_listed_by_the_extensions_it_actually_has(tmp_path: Path) -> None:
    """An unlisted extension is invisible, and the error reads as an empty directory.

    The bytes are PNG in every file here on purpose: the point is the *listing*, and a
    reader that only sees `.png` would report this directory as having no images at all.
    """
    for name in ("phone.heic", "modern.avif", "legacy.jfif", "ordinary.png"):
        (tmp_path / name).write_bytes(_png(4, 4))
    got = _rows(bt.read.images(str(tmp_path)))
    assert set(got) == {"phone.heic", "modern.avif", "legacy.jfif", "ordinary.png"}


def test_an_audio_corpus_includes_the_modern_containers(tmp_path: Path) -> None:
    """Opus is what every WebRTC capture and modern voice recording is stored as."""
    for name in ("call.opus", "clip.oga", "voice.aac", "old.wma", "plain.wav"):
        (tmp_path / name).write_bytes(b"not really audio")
    got = _rows(bt.read.audio(str(tmp_path), on_error="skip"))
    assert set(got) == {"call.opus", "clip.oga", "voice.aac", "old.wma", "plain.wav"}


def test_a_file_the_header_parser_rejects_still_produces_a_row(tmp_path: Path) -> None:
    """The rows are worth having even when the metadata is not.

    `bytes`, `size` and `mime` come from the read itself, so a container Pillow cannot
    parse nulls that file's metadata columns rather than dropping it — which is what makes
    listing a format the decoder may not handle the right trade.
    """
    (tmp_path / "good.png").write_bytes(_png(6, 3))
    (tmp_path / "unreadable.heic").write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 32)
    got = _rows(bt.read.images(str(tmp_path)))
    assert got["good.png"]["width"] == 6
    assert got["unreadable.heic"]["width"] is None
    assert got["unreadable.heic"]["size"] > 0, "the payload is still read"


def test_an_image_reports_the_container_its_bytes_actually_are(tmp_path: Path) -> None:
    """From the header, not the path: the two disagree constantly in a scraped corpus."""
    (tmp_path / "lying.jpg").write_bytes(_png(4, 4))
    got = _rows(bt.read.images(str(tmp_path)))
    assert got["lying.jpg"]["format"] == "png"
    assert got["lying.jpg"]["mime"] == "image/png"


def test_a_video_reports_its_codec_and_whether_it_has_a_soundtrack(tmp_path: Path) -> None:
    """The two facts a video corpus is triaged on, from the parse already happening."""
    (tmp_path / "silent.mp4").write_bytes(_mp4(tmp_path, video=True, audio=False))
    (tmp_path / "talkie.mp4").write_bytes(_mp4(tmp_path, video=True, audio=True))
    got = _rows(bt.read.video(str(tmp_path)))
    assert got["silent.mp4"]["has_audio"] is False
    assert got["talkie.mp4"]["has_audio"] is True
    for row in got.values():
        assert row["codec"], "every decodable video stream names its codec"
        assert (row["width"], row["height"]) == (32, 32)


def test_an_audio_only_container_is_described_rather_than_nulled(tmp_path: Path) -> None:
    """`streams.video[0]` raised `IndexError`, which nulled *every* metadata column.

    An audio-only MP4 then looked exactly like a truncated file. Reporting the dimensions
    as absent and `has_audio` as true is what tells the two apart.
    """
    (tmp_path / "podcast.mp4").write_bytes(_mp4(tmp_path, video=False, audio=True))
    row = _rows(bt.read.video(str(tmp_path)))["podcast.mp4"]
    assert row["has_audio"] is True, "the fact that distinguishes it from a corrupt file"
    assert row["width"] is None and row["height"] is None
    assert row["codec"] is None


def test_the_new_metadata_columns_are_pushdown_visible(tmp_path: Path) -> None:
    """A metadata column nobody can filter on is a column nobody uses.

    Projection and predicate both have to reach them, on the same read the listing does.
    """
    (tmp_path / "silent.mp4").write_bytes(_mp4(tmp_path, video=True, audio=False))
    (tmp_path / "talkie.mp4").write_bytes(_mp4(tmp_path, video=True, audio=True))
    ds = bt.read.video(str(tmp_path))
    with_sound = ds.filter(bt.col("has_audio")).select("uri", "codec").collect().to_pylist()
    assert [r["uri"].rsplit("/", 1)[-1] for r in with_sound] == ["talkie.mp4"]


def test_the_header_parse_is_skipped_when_no_metadata_column_is_projected(
    tmp_path: Path, monkeypatch
) -> None:
    """Extracting header metadata is a **Python** parse per file, and it ran unconditionally.

    That made the namespace's most common pipeline pay for something it never read: a decode
    query projects the decoded tensor, and every file still had Pillow opened on it to fill
    width/height/mode/format columns the projection had already dropped. Measured on 2,000
    JPEGs it was about a third of the wall clock.

    Counted rather than timed, because a timing assertion on a shared machine is a flaky
    test that says nothing about *why* it got faster.
    """
    from batcher.io.formats.multimodal.images import ImageSource

    for i in range(8):
        (tmp_path / f"{i}.png").write_bytes(_png(4 + i, 4))

    parses = 0
    original = ImageSource._extract_meta

    def counting(self, data):
        nonlocal parses
        parses += 1
        return original(self, data)

    monkeypatch.setattr(ImageSource, "_extract_meta", counting)

    def run(build) -> int:
        nonlocal parses
        parses = 0
        build().collect()
        return parses

    assert run(lambda: bt.read.images(str(tmp_path)).select("uri")) == 0
    assert run(lambda: bt.read.images(str(tmp_path)).select("uri", "bytes")) == 0
    # A decode pipeline: the tensor is what is wanted, the header facts are not.
    decoded = lambda: bt.read.images(str(tmp_path), decode=True, size=(4, 4)).select("image")  # noqa: E731
    assert run(decoded) == 0

    # ...and it still runs, exactly once per file, when a metadata column *is* asked for.
    assert run(lambda: bt.read.images(str(tmp_path)).select("uri", "width")) == 8
    assert run(lambda: bt.read.images(str(tmp_path))) == 8


def test_skipping_the_parse_does_not_change_what_a_projection_returns(tmp_path: Path) -> None:
    """The optimization must be invisible: same rows, same values, same column types.

    The metadata columns still exist in the schema when `with_meta` is on, so a batch is
    still built with them — all-null when the parse was skipped, which is the honest value
    for "not read" and which the `.select` immediately discards.
    """
    for i in range(6):
        (tmp_path / f"{i}.png").write_bytes(_png(4 + i, 4))

    full = bt.read.images(str(tmp_path)).collect().to_pydict()
    projected = bt.read.images(str(tmp_path)).select("uri", "size", "mime").collect().to_pydict()
    assert projected["uri"] == full["uri"]
    assert projected["size"] == full["size"]
    assert projected["mime"] == full["mime"]

    # And a projection that mixes the two still gets real metadata.
    mixed = bt.read.images(str(tmp_path)).select("uri", "height").collect().to_pydict()
    assert mixed["height"] == full["height"]


def test_a_media_read_gives_the_same_answer_pooled_or_serial(tmp_path: Path, monkeypatch) -> None:
    """The reader chooses concurrency by whether the path is local, so both paths must agree.

    A remote read is a round trip and wants a pool; a local read is a syscall on page cache
    and is measurably *slower* pooled (2,000 local JPEGs: 52 ms serial against 118 ms on an
    8-thread pool). This reader used to pool unconditionally and paid that cost on every
    developer machine. Now the decision comes from `read_each_file`, which owns it for every
    connector — so what has to be asserted is that the two branches are interchangeable:
    same rows, same order, same values.
    """
    import batcher.io._concurrent as concurrent

    for i in range(24):
        (tmp_path / f"{i:02}.png").write_bytes(_png(4 + (i % 3), 4))

    serial = bt.read.images(str(tmp_path)).collect().to_pydict()
    # Force the pooled branch by telling the helper these paths are remote.
    monkeypatch.setattr(concurrent, "is_local_path", lambda _path: False)
    pooled = bt.read.images(str(tmp_path)).collect().to_pydict()

    assert pooled["uri"] == serial["uri"], "the pool must preserve file order"
    assert pooled == serial
