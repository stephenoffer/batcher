"""A payload the engine cannot decode is a null row, not an engine error.

Every media expression follows the same convention: a payload it cannot parse produces a
null, because an unstructured corpus is *expected* to be heterogeneous. A directory of
scraped blobs holds JSON next to JPEGs next to WARC records, and asking for
`.audio.to_waveform()` over it is a normal thing to do, not a mistake.

The decoders underneath do not share that convention. `symphonia` logs at ERROR every time
it fails to identify a payload, through the `log` facade, which `tracing-subscriber`
bridges into `tracing` by default and `bc_py::tracing_init` then forwards to the Python
`batcher.engine` logger. So a healthy job over a mixed corpus used to emit one ERROR line
*per row* for the rows behaving exactly as documented.

That cost more than the work it described. Each forwarded record acquires the GIL, which
serializes the rayon fan-out `eval::media::map_rows` exists to provide. Measured over 2,000
non-audio blobs through `.audio.to_waveform()`: **123 microseconds per row forwarding the
noise against 3.1 with the bridge quiet** — and 11 microseconds for a row that decodes
*successfully*, so failing to decode cost 33x more than succeeding.

The noise was the worse half though. Thousands of ERROR lines from a job with nothing wrong
with it is alarm fatigue, and it buries the engine's real diagnostics in the same stream.

Two properties are pinned here, and the pair is what makes the test discriminating:

- At `WARNING` no record arrives. A level filter alone cannot produce that, because these
  records *are* ERRORs and ERROR passes `WARNING` — so this can only hold if the bridge
  distinguishes third-party records from the engine's own.
- At `DEBUG` they come back, demoted to DEBUG, still carrying the `log.target` marker that
  identifies them. That proves the first property is a level-gated suppression of exactly
  the third-party population, not a blanket deletion, and that the caller who asks for
  debug detail still gets it.

Each case runs in a **subprocess**, because the subscriber is installed once per process
(`INIT`, a `OnceLock`) and the level it captures is therefore whatever the first query in
that process configured.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.integration

#: Non-media payloads, of the kind that sit beside real media in a scraped corpus. Every
#: one of them is a null row and an ERROR from `symphonia` on the way there.
_JUNK_ROWS = 40

#: The field `tracing-log` stamps on a record it synthesized from the `log` facade. The
#: bridge appends non-message fields to the text, so its presence in a forwarded message is
#: the observable proof that the record came from a third-party crate and not the engine.
_FOREIGN_MARKER = "log.target="


def _capture_engine_log(level: str) -> list[dict[str, str]]:
    """Decode a corpus of non-media blobs at `level`; return the `batcher.engine` records.

    The handler is attached before the first query so the subscriber is installed at the
    level under test, and `propagate` is left alone — this reads the records the engine
    hands to Python, not whatever the console configuration does with them afterwards.
    """
    script = textwrap.dedent(f"""
        import json, logging
        import batcher as bt
        from batcher.config import set_log_level

        set_log_level({level!r})

        seen = []

        class Capture(logging.Handler):
            def emit(self, record):
                seen.append({{"level": record.levelname, "message": record.getMessage()}})

        engine_log = logging.getLogger("batcher.engine")
        engine_log.addHandler(Capture())
        engine_log.setLevel(logging.DEBUG)

        from batcher import col

        rows = [b'{{"id": %d, "caption": "a photo"}}' % i for i in range({_JUNK_ROWS})]
        out = bt.from_pydict({{"a": rows}}).select(
            x=col("a").audio.to_waveform()
        ).to_pydict()["x"]
        assert out == [None] * {_JUNK_ROWS}, "the corpus should decode to nulls"

        print("---RECORDS---")
        print(json.dumps(seen))
    """)
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, f"child failed:\n{done.stdout}\n{done.stderr[-2000:]}"
    return json.loads(done.stdout.split("---RECORDS---", 1)[1])


def test_undecodable_payloads_are_not_reported_as_engine_errors():
    """The property this file exists for.

    These records are ERRORs, and ERROR passes a `WARNING` filter, so a level filter cannot
    explain an empty result here. Only a bridge that tells the engine's own events apart
    from a third-party crate's can.
    """
    records = _capture_engine_log("warning")

    assert records == [], (
        f"{len(records)} engine log records for {_JUNK_ROWS} rows behaving as documented; "
        f"first: {records[:1]}"
    )


def test_debug_restores_the_third_party_records_it_suppressed():
    """Suppression is level-gated, not deletion — and it is the foreign population.

    Every record that reappears carries the `log.target` marker, which is what makes this
    the other half of a discriminating pair rather than a restatement of it: the bridge
    drops records *because* they came from the `log` facade, not because of their level.
    """
    records = _capture_engine_log("debug")

    assert records, "debug asked for third-party detail and got none"
    unmarked = [r for r in records if _FOREIGN_MARKER not in r["message"]]
    assert not unmarked, f"records reappeared that were not third-party: {unmarked[:3]}"


def test_a_third_party_error_is_not_reported_at_error_severity():
    """A decoder's idea of an error is not the engine's.

    `symphonia` logs an unidentifiable payload at ERROR. For a media expression that is the
    expected outcome for every non-audio row, so forwarding it at ERROR would let a healthy
    job look like a failing one to anything watching severity.
    """
    records = _capture_engine_log("debug")

    louder = sorted({r["level"] for r in records} - {"DEBUG"})
    assert not louder, f"third-party records forwarded above DEBUG: {louder}"
