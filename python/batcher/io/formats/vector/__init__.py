"""`io.formats.vector` — vector-store connectors over one shared contract.

`contract` states what every vector store here agrees on: the frame's shape (an id column,
``fixed_size_list<float32>`` vector columns, payload columns), validation of every point and of
the target's dimension and metric before the first request, batched sends with idempotent
retries, and a per-point failure report (`VectorWriteError`). The four connectors register
into `SOURCES` and `SINKS` on import: Qdrant, Pinecone, Milvus and Turbopuffer. Each client
library is imported on the worker, so importing this package needs none of them.
"""

from __future__ import annotations

from batcher.io.formats.vector.contract import PointFailure, VectorSink, VectorWriteError
from batcher.io.formats.vector.milvus import MilvusSink, MilvusSource
from batcher.io.formats.vector.pinecone import PineconeSink, PineconeSource
from batcher.io.formats.vector.qdrant import QdrantSink, QdrantSource
from batcher.io.formats.vector.turbopuffer import TurbopufferSink, TurbopufferSource

__all__ = [
    "MilvusSink",
    "MilvusSource",
    "PineconeSink",
    "PineconeSource",
    "PointFailure",
    "QdrantSink",
    "QdrantSource",
    "TurbopufferSink",
    "TurbopufferSource",
    "VectorSink",
    "VectorWriteError",
]
