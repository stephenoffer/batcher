"""The unstructured story end to end, through the seams rather than around them.

Each piece here has its own focused tests. This file exists for what those cannot see: the
*joins* between them. A media listing feeds an expression; an expression feeds a decode; a
decode feeds a model-shaped tensor; the whole thing has to give the same answer collected,
streamed, and distributed.

Every assertion is about a boundary two components share, so a change that breaks one of
them fails here even when both sides' own tests stay green.
"""

from __future__ import annotations

import io
import math
import struct
from pathlib import Path

import pytest

import batcher as bt

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

pytestmark = pytest.mark.integration


def _png(path: Path, array) -> None:
    Image.fromarray(array.astype("uint8")).save(path)


def _wav(path: Path, samples: list[float], rate: int = 16000) -> None:
    pcm = b"".join(struct.pack("<h", round(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


@pytest.fixture
def photos(tmp_path: Path) -> Path:
    """A scraped-looking image corpus: real scenes, a blank tile, and a duplicate."""
    d = tmp_path / "photos"
    d.mkdir()
    scene = np.indices((96, 64)).sum(0).astype("uint8")[:, :, None].repeat(3, 2)
    scene[:, :, 1] = np.indices((96, 64))[1].astype("uint8")
    _png(d / "scene.png", scene)
    # The same picture at a quarter size — what a perceptual hash exists to catch.
    _png(d / "scene_small.png", np.asarray(Image.fromarray(scene).resize((16, 24))))
    _png(d / "blank.png", np.full((96, 64, 3), 128, dtype="uint8"))
    # A file whose extension lies about its container: PNG bytes under a `.jpg` name.
    # Saving through the path would let Pillow believe the extension and write a real
    # JPEG, which is the opposite of the corpus this is standing in for.
    buf = io.BytesIO()
    Image.fromarray(scene).save(buf, "PNG")
    (d / "mislabelled.jpg").write_bytes(buf.getvalue())
    # An unrelated *textured* picture, so a fingerprint's separation can be measured
    # against something rather than against a remembered absolute number. Textured, not
    # two-tone: a striped tile has almost no entropy and the curation filter drops it.
    noise = ((np.indices((96, 64))[0] * 37 + np.indices((96, 64))[1] * 91) % 256).astype("uint8")
    _png(d / "noise.png", noise[:, :, None].repeat(3, 2))
    return d


def test_a_listing_feeds_the_expressions_that_curate_it(photos: Path) -> None:
    """The seam between the reader's metadata and the `.image` namespace.

    Both answer questions about the same bytes, and they have to agree: the listing's
    header parse and the expression's header parse are different code reaching the same
    file, so a corpus where they disagree is one where a filter and a projection select
    different rows.
    """
    ds = bt.read.images(str(photos))
    rows = (
        ds.select(
            name=bt.col("uri").str.parse_filename(),
            listed_w=bt.col("width"),
            listed_fmt=bt.col("format"),
            expr_ratio=bt.col("bytes").image.aspect_ratio(),
            expr_fmt=bt.col("bytes").image.format(),
        )
        .collect()
        .to_pylist()
    )
    by_name = {r["name"]: r for r in rows}
    for row in rows:
        assert row["listed_fmt"] == row["expr_fmt"], row["name"]
    # The extension says JPEG; both the listing and the expression say PNG.
    assert by_name["mislabelled.jpg"]["listed_fmt"] == "png"
    scene = by_name["scene.png"]
    assert scene["expr_ratio"] == pytest.approx(scene["listed_w"] / 96)


def test_curation_then_deduplication_then_a_model_tensor(photos: Path) -> None:
    """The whole ingest, as one lazy plan: screen, fingerprint, and shape.

    This is the pipeline the namespace exists for, and the thing it proves is that the
    stages compose — a measure feeding a filter feeding a hash feeding a fixed-shape
    tensor, with no `map_batches` and no Python between them.
    """
    bytes_col = bt.col("bytes")
    ds = (
        bt.read.images(str(photos))
        .with_columns(
            detail=bytes_col.image.entropy(),
            digest=bytes_col.image.phash(),
        )
        # A flat tile has near-zero entropy where a photograph has 6-8 bits.
        .filter(bt.col("detail") > 4.0)
        .with_columns(x=bytes_col.image.letterbox(32, 32))
    )
    out = ds.select("uri", "digest", "x").collect().to_pydict()
    names = sorted(u.rsplit("/", 1)[-1] for u in out["uri"])
    assert "blank.png" not in names, "the placeholder tile should have been screened out"
    assert len(names) == 4

    # The rescaled copy must fingerprint far closer to its original than an unrelated
    # picture does — the *separation* is what makes the dedup a join rather than a model,
    # and an absolute bit count would only pin the resampler's current behaviour.
    by_name = dict(zip((u.rsplit("/", 1)[-1] for u in out["uri"]), out["digest"], strict=True))

    def _distance(a: str, b: str) -> int:
        return bin((by_name[a] ^ by_name[b]) & (2**64 - 1)).count("1")

    rescaled = _distance("scene.png", "scene_small.png")
    unrelated = _distance("scene.png", "noise.png")
    assert rescaled < unrelated / 2, f"rescale moved {rescaled} bits, a new picture {unrelated}"
    assert rescaled <= 12, f"a 4x rescale moved {rescaled} bits, past the usual cutoff"

    # Every surviving row is the same model-ready shape, whatever it started as.
    assert {len(row) for row in out["x"]} == {32 * 32 * 3}


def test_the_reader_and_the_expression_reach_the_same_tensor(photos: Path) -> None:
    """`read.images(fit=...)` is a shorthand for the expression, not a second implementation.

    If they ever diverge, a pipeline that switched from one spelling to the other would
    silently change its pixels — with the tensor's shape identical either way.
    """
    via_reader = (
        bt.read.images(str(photos), size=(32, 32), fit="letterbox")
        .select("uri", "image")
        .collect()
        .to_pydict()
    )
    via_expr = (
        bt.read.images(str(photos))
        .select(uri=bt.col("uri"), image=bt.col("bytes").image.letterbox(32, 32))
        .collect()
        .to_pydict()
    )
    assert via_reader["uri"] == via_expr["uri"]
    for a, b in zip(via_reader["image"], via_expr["image"], strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_an_audio_corpus_is_triaged_levelled_and_written_back(tmp_path: Path) -> None:
    """The audio counterpart, ending where a cleaned corpus has to end: as audio.

    The seam under test is that the measures, the shaping ops and the encoder all read the
    same clip — including reading each other's *output*, which is what makes a two-step
    clean one expression.
    """
    d = tmp_path / "clips"
    d.mkdir()
    _wav(d / "speech.wav", [0.0] * 800 + [0.3 * math.sin(i / 9) for i in range(4000)] + [0.0] * 800)
    _wav(d / "silence.wav", [0.0] * 4000)
    clip = bt.col("bytes")

    ds = bt.read.audio(str(d)).with_columns(
        quiet=clip.audio.silence_ratio(),
        level=clip.audio.dbfs(),
    )
    triage = ds.select("uri", "quiet", "level").collect().to_pylist()
    by_name = {r["uri"].rsplit("/", 1)[-1]: r for r in triage}
    assert by_name["silence.wav"]["quiet"] == 1.0
    assert by_name["silence.wav"]["level"] is None, "digital silence has no level"
    assert by_name["speech.wav"]["quiet"] < 0.5

    cleaned = clip.audio.trim_silence().audio.rms_normalize().audio.encode_wav(16000)
    written = (
        ds.filter(bt.col("quiet") < 0.9)
        .select(meta=cleaned.audio.decode())
        .collect()
        .to_pydict()["meta"]
    )
    assert len(written) == 1
    assert written[0]["sample_rate"] == 16000
    # The sine starts at zero, so its first sample is part of the leading quiet.
    assert written[0]["num_frames"] == 3999, "the silent ends should have gone"


def test_documents_chunk_into_rows_a_retrieval_index_can_hold(tmp_path: Path) -> None:
    """The RAG ingest, across formats: read a mixed corpus, chunk, and explode to rows.

    The seam is between the reader's `{path, page, text}` shape and the string namespace:
    a chunk list has to explode into rows that still say which document and page they came
    from, or the retrieved passage cannot be cited.
    """
    d = tmp_path / "docs"
    d.mkdir()
    (d / "guide.html").write_text(
        "<h1>Install</h1><p>Run the installer and wait.</p><p>Then restart.</p>"
    )
    (d / "notes.md").write_text("# Notes\n\nA short note about the thing.\n")

    passages = (
        bt.read.documents(str(d))
        .with_columns(chunk=bt.col("text").str.chunk(24, overlap=6))
        .explode("chunk")
        .select(
            doc=bt.col("path").str.parse_filename(),
            page=bt.col("page"),
            chunk=bt.col("chunk"),
        )
        .collect()
        .to_pylist()
    )
    assert len(passages) > 2, "a document longer than the chunk size must split"
    assert {p["doc"] for p in passages} == {"guide.html", "notes.md"}
    assert all(p["page"] == 0 for p in passages), "neither format paginates"
    assert all(p["chunk"] for p in passages), "no empty passages"
    # The HTML markup is gone but its prose survived the split.
    joined = " ".join(p["chunk"] for p in passages if p["doc"] == "guide.html")
    assert "Install" in joined and "restart" in joined
    assert "<p>" not in joined


def test_the_pipeline_gives_one_answer_collected_and_streamed(photos: Path) -> None:
    """Two executors, one plan, one answer — asserted positionally, since order is part of it.

    A per-row kernel that reads anything batch-scoped is exactly what makes these differ,
    and a media pipeline is full of per-batch resolved parameters.
    """
    bytes_col = bt.col("bytes")
    query = bt.read.images(str(photos)).select(
        name=bt.col("uri").str.parse_filename(),
        detail=bytes_col.image.entropy(),
        digest=bytes_col.image.ahash(),
        colour=bytes_col.image.mean_color(),
        shaped=bytes_col.image.thumbnail(16, format="jpeg").image.format(),
    )
    collected = query.collect().to_pydict()
    streamed: dict[str, list] = {name: [] for name in collected}
    for batch in query.iter_batches(batch_size=2):
        for name, values in batch.to_pydict().items():
            streamed[name] += values
    assert streamed == collected
