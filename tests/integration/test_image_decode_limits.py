"""An image too large to decode is a null row, not an out-of-memory worker.

A "decompression bomb" is a small file declaring enormous dimensions: a solid-colour
20000x20000 PNG is about a megabyte on disk and 1.2 GB decoded. They arrive in real corpora
without malice — a scanned map, a gigapixel panorama, a satellite tile — and a decoder that
honours the header allocates whatever it is told to.

The engine already refuses, because the decoder carries a 512 MiB allocation ceiling, and
the refusal surfaces through the media convention: the row is null. This file exists
because nothing said so. The behavior is load-bearing for any pipeline pointed at a corpus
it did not create, it is invisible in ordinary use, and the failure mode if it regressed —
one row taking down the worker that touched it — is exactly the kind that only shows up at
scale, on someone else's data.

Each test runs the decode in a **subprocess under an explicit address-space cap**, so a
regression fails here as a bounded `MemoryError` rather than by evicting whatever else is
running on the machine.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.integration

#: Decoded size of the fixture, well past the decoder's 512 MiB ceiling.
_BOMB_EDGE = 20_000

#: Address-space cap for the child. Generous enough for the interpreter, pyarrow and the
#: engine; far below the 1.2 GB a successful bomb decode would need.
_CHILD_MEMORY_BYTES = 4 * 1024**3


def _run_in_capped_child(body: str) -> str:
    """Run `body` under an address-space rlimit and return its stdout.

    The cap is the point: without it a regression in the decoder's limit does not fail this
    test, it takes the machine down — and on a shared box that is everyone else's problem
    rather than a red build.
    """
    pytest.importorskip("PIL")
    script = textwrap.dedent(f"""
        import resource
        resource.setrlimit(
            resource.RLIMIT_AS, ({_CHILD_MEMORY_BYTES}, {_CHILD_MEMORY_BYTES})
        )
        import io
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None  # Pillow guards its own writer; we want the bomb.
        import batcher as bt
        from batcher import col

        buf = io.BytesIO()
        Image.new("RGB", ({_BOMB_EDGE}, {_BOMB_EDGE}), (7, 7, 7)).save(
            buf, format="PNG", compress_level=9
        )
        bomb = buf.getvalue()
        ds = bt.from_pydict({{"i": [bomb]}})
    """) + textwrap.dedent(body)
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, f"child failed:\n{done.stdout}\n{done.stderr[-2000:]}"
    return done.stdout


def test_a_decompression_bomb_nulls_its_row_rather_than_exhausting_memory():
    """The property this file exists for, on every op that touches pixels."""
    out = _run_in_capped_child("""
        ops = {
            "to_tensor": lambda c: c.image.to_tensor(8, 8),
            "to_tensor_f32": lambda c: c.image.to_tensor_f32(8, 8),
            "to_grayscale": lambda c: c.image.to_grayscale(8, 8),
            "center_crop": lambda c: c.image.center_crop(8, 8),
            "letterbox": lambda c: c.image.letterbox(8, 8),
            "thumbnail": lambda c: c.image.thumbnail(64),
            "resize": lambda c: c.image.resize(8, 8),
            "brightness": lambda c: c.image.brightness(),
            "sharpness": lambda c: c.image.sharpness(),
            "dhash": lambda c: c.image.dhash(),
            "auto_orient": lambda c: c.image.auto_orient(),
        }
        for name, build in ops.items():
            got = ds.select(x=build(col("i"))).to_pydict()["x"]
            print(f"{name}={got[0] is None}")
    """)

    results = dict(line.split("=") for line in out.strip().splitlines())
    assert results, "the child produced no results"
    not_null = [name for name, is_null in results.items() if is_null != "True"]
    assert not not_null, f"these decoded a 1.2 GB image instead of nulling it: {not_null}"


def test_the_header_is_still_readable():
    """`decode` reads the header and stops, so it answers for an image nothing can decode.

    That is the difference that makes the null explicable rather than mysterious: a corpus
    can be *surveyed* for oversized images even though it cannot decode them, so the rows
    that will come back null are knowable in advance.
    """
    out = _run_in_capped_child("""
        d = ds.select(x=col("i").image.decode()).to_pydict()["x"][0]
        print(f"width={d['width']}")
        print(f"height={d['height']}")
    """)

    facts = dict(line.split("=") for line in out.strip().splitlines())
    assert int(facts["width"]) == _BOMB_EDGE
    assert int(facts["height"]) == _BOMB_EDGE


def test_a_bomb_does_not_take_its_neighbours_with_it():
    """One unusable row in a batch must cost that row only.

    A corpus mixing ordinary photographs with the occasional gigapixel scan is the normal
    case, not the adversarial one, and losing the batch to the scan would lose the
    photographs that travelled with it.
    """
    out = _run_in_capped_child("""
        small = io.BytesIO()
        Image.new("RGB", (32, 32), (200, 30, 30)).save(small, format="PNG")
        mixed = bt.from_pydict({"i": [small.getvalue(), bomb, small.getvalue()]})
        got = mixed.select(x=col("i").image.to_tensor(8, 8)).to_pydict()["x"]
        print(f"nulls={[v is None for v in got]}")
    """)

    assert out.strip() == "nulls=[False, True, False]"
