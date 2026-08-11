"""The two untyped readers: whole-file bytes and line-by-line text.

`read.text` gives one row per line, carrying `path`, `line_number` and `text` — so the
line's content is the `text` column, not the first one. `read.binary` gives one row per
file. Both are the way in for anything with no reader of its own: land the bytes as a
column and parse them with expressions, keeping the parsing in the engine.

    python examples/io/text_and_binary.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    log_lines = [
        "2024-01-05 INFO  request id=41 duration=13ms",
        "2024-01-05 ERROR request id=42 duration=220ms",
        "2024-01-06 INFO  request id=43 duration=8ms",
    ]

    with tempfile.TemporaryDirectory() as directory:
        log_path = Path(directory) / "app.log"
        log_path.write_text("\n".join(log_lines) + "\n")

        lines = bt.read.text(str(log_path))
        print(lines.schema)
        assert lines.count() == 3

        assert lines.columns == ["path", "line_number", "text"]
        parsed = lines.select(
            "line_number",
            level=col("text").str.extract(r"\d{4}-\d{2}-\d{2} (\w+)"),
            request=col("text").str.extract(r"id=(\d+)").cast("int64"),
            millis=col("text").str.extract(r"duration=(\d+)ms").cast("int64"),
        )
        result = parsed.to_pydict()
        print(result)
        assert result["level"] == ["INFO", "ERROR", "INFO"]
        assert result["request"] == [41, 42, 43]
        assert result["millis"] == [13, 220, 8]

        # Whole files as bytes: one row per file, and the parsing is yours to do.
        blob_path = Path(directory) / "payload.bin"
        blob_path.write_bytes(b"\x00\x01\x02\x03payload")
        blobs = bt.read.binary(str(blob_path))
        print(blobs.schema)
        assert blobs.count() == 1
        raw = blobs.to_pydict()
        payload = next(value for value in raw.values() if isinstance(value[0], bytes))
        assert payload[0].endswith(b"payload")


if __name__ == "__main__":
    main()
