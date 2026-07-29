"""The source and sink registries: what formats exist, and the objects behind them.

``bt.read.parquet(...)`` is a façade over a registry of ``SourceFormat`` implementations.
Reading the registry is how you discover what is supported in *this* build, rather than
trusting a docs page that may predate an extra you have not installed.

    python examples/io/sources_and_sinks.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import batcher as bt
from batcher.io import SINKS, SOURCES, CSVSink, JSONSink, ParquetSink, ParquetSource


def main() -> None:
    # What this build can read and write.
    source_names = sorted(SOURCES)
    sink_names = sorted(SINKS)
    print("sources:", source_names[:12])
    print("sinks:", sink_names[:12])
    assert "parquet" in source_names
    assert "csv" in source_names
    assert "json" in source_names
    assert "parquet" in sink_names

    # There are many more readers than writers, which is the usual shape: you ingest
    # from everything and write to a few curated formats.
    assert len(source_names) > len(sink_names)

    # The reader façade lists the same formats as methods.
    for name in ("parquet", "csv", "json", "arrow"):
        assert hasattr(bt.read, name), name
    for name in ("parquet", "csv", "json"):
        assert hasattr(bt.from_pydict({"a": [1]}).write, name), name

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "t.parquet")
        ds = bt.from_pydict({"a": [1, 2, 3]})

        # The façade, which is what you normally write.
        ds.write.parquet(path)
        assert bt.read.parquet(path).count() == 3

        # The registry entries are the objects behind it. Useful when you are writing a
        # connector and want to see the contract you must satisfy. Look one up with
        # `get`; iterate the registry for its names.
        assert SOURCES.get("parquet") is ParquetSource
        assert SINKS.get("parquet") is ParquetSink
        assert "parquet" in SOURCES.names()
        print("ParquetSource:", ParquetSource.__name__)
        print("ParquetSink:", ParquetSink.__name__)
        print("CSVSink / JSONSink:", CSVSink.__name__, JSONSink.__name__)

        # A source knows its own schema before any rows are read, which is what makes
        # `ds.schema` free.
        reader = bt.read.parquet(path)
        assert reader.schema.names == ["a"]
        # And the row count comes from the footer.
        assert reader.count() == 3

    # A dataset built in memory has a source too -- the interop constructors are just
    # another registered way in.
    assert bt.from_pydict({"x": [1]}).count() == 1
    assert bt.from_pylist([{"x": 1}, {"x": 2}]).count() == 2


if __name__ == "__main__":
    main()
