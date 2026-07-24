"""Pipeline identity — the durable name and note attached to a plan shape.

A *pipeline* is every run of one plan shape, identified by the plan signature Kyber already
keys learned stats on. That signature is the pipeline's stable id. This package adds the one
thing the signature cannot carry: a human name and note that a person assigns and that
outlives the process, held in a small JSON registry under `$BATCHER_HOME`.
"""

from __future__ import annotations

from batcher.observe.pipelines.grouping import group_pipelines
from batcher.observe.pipelines.registry import PipelineMeta, PipelineRegistry

__all__ = ["PipelineMeta", "PipelineRegistry", "group_pipelines"]
