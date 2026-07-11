"""Multimodal suites: unstructured-data ingest (images, and future audio/video/text).

The unstructured counterpart to ``standard`` (SQL tables) and ``scan`` (parquet layouts):
read + decode + transform file corpora of non-tabular data, across the engines that have a
multimodal path (Batcher, Ray Data, Daft; PyArrow for byte-level reads). Family modules are
auto-discovered, as in the other suites.
"""

from __future__ import annotations

from discover import import_submodules

import_submodules(__name__)
