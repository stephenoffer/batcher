"""Bag-of-words text vectorizers — counts, TF-IDF, and the hashing trick.

The classical text feature stack, for the models that still beat an embedding on short
labelled corpora: spam and intent classifiers, ticket routing, deduplication, and any
linear model where an interpretable feature name matters more than a dense vector.

Which to reach for:

`CountVectorizer`
    Learns a vocabulary and counts each document's terms. The right default when the corpus
    fits a vocabulary you want to be able to read.
`TfidfVectorizer`
    The same, with each count divided down by how many documents use the term. Almost
    always the better features, and the standard input to a linear text classifier.
`HashingVectorizer`
    No vocabulary at all — the feature index is a hash of the term. Stateless, so it needs
    no fit pass, cannot skew between training and serving, and is the only one of the three
    that works on an unbounded stream.
"""

from __future__ import annotations

from batcher.ml.preprocessors.vectorizers.counts import CountVectorizer
from batcher.ml.preprocessors.vectorizers.hashing import HashingVectorizer
from batcher.ml.preprocessors.vectorizers.tokens import ENGLISH_STOP_WORDS
from batcher.ml.preprocessors.vectorizers.weighting import TfidfVectorizer

__all__ = [
    "ENGLISH_STOP_WORDS",
    "CountVectorizer",
    "HashingVectorizer",
    "TfidfVectorizer",
]
