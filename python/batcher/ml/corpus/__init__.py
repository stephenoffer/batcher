"""Preparing a training corpus: mixing sources, filtering junk, removing eval leakage.

The three steps between "we have some text" and "we can train on it", each of which is a data
bug that presents as a model bug when skipped.

`mixing` samples several corpora at declared weights, because concatenating them gives the
ratio of their *sizes* rather than the ratio you intended. `filtering` drops the documents that
are not prose, using the published web-corpus heuristics with a report so you can see what each
rule removes before trusting it. `decontamination` drops the training documents that quote your
evaluation set, without which a benchmark score measures memorization.

Everything here is built on the public `Dataset` API, so each step is a plan the optimizer sees
whole rather than a pass over materialized rows.
"""

from __future__ import annotations

from batcher.ml.corpus.decontamination import contamination_rate, decontaminate
from batcher.ml.corpus.filtering import QualityThresholds, quality_filter, quality_report
from batcher.ml.corpus.mixing import MixtureReport, mix_corpora

__all__ = [
    "MixtureReport",
    "QualityThresholds",
    "contamination_rate",
    "decontaminate",
    "mix_corpora",
    "quality_filter",
    "quality_report",
]
