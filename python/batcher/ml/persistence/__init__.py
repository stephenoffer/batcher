"""Persisting fitted state — the document format, and estimator save/load.

A fitted object is only useful because its state outlives the process that fitted it. The
`Preprocessor` family has had `save`/`load` since it existed; the estimators had nothing,
so a model trained across a cluster could not be moved anywhere and the only route to a
prediction was to fit again.

`document` holds the shape both halves write — version, class, constructor parameters,
learned state — so the two cannot drift into files only one loader understands. `models`
adds that surface for the estimators.

JSON rather than a pickle, for the same reasons throughout: a pickle is opaque, breaks when
a class moves or a slot is renamed, and is unsafe to load from a store you do not fully
control. What is written is reviewable, diffable, and readable from another language.
"""

from __future__ import annotations

from batcher.ml.persistence.models import (
    load_model,
    model_from_dict,
    model_to_dict,
    save_model,
)

__all__ = ["load_model", "model_from_dict", "model_to_dict", "save_model"]
