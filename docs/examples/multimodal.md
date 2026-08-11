# Multimodal and text analytics

This page covers the scripts that work with images and binary payloads, and the ones that
analyse text corpora at scale.

## Filter on metadata before you decode

An image read produces one row per file, carrying the bytes plus the metadata that can be had
without decoding: size, mime type, width and height. Decoding is opt-in because it is the
expensive part, and putting the metadata filter first is the single biggest win in a
multimodal pipeline.

```python
# docs: skip
import batcher as bt
from batcher import col

pictures = bt.read.images("s3://ray-benchmark-data/profile-pictures/1GiB/00000*.jpg")

# Header only: no pixels have been decoded.
assert {"uri", "bytes", "size", "mime", "width", "height"} <= set(pictures.columns)

wanted = pictures.filter(col("size") > 4096)

# Decoding requires a target size, because a decoded batch has to be one rectangular
# tensor and the shape has to be known before the first file is opened.
decoded = bt.read.images(
    "s3://ray-benchmark-data/profile-pictures/1GiB/00000*.jpg",
    decode=True,
    size=(32, 32),
)
```

`width` and `height` keep describing the source file after a decode; the pixels land in a new
`image` column. `offload_blobs` and `materialize_blobs` are the pair that let a pipeline
filter and join on metadata without carrying the payload, and re-materialize only what
survived.

## Text at corpus scale

Splitting into a list, exploding into rows, and grouping is the whole shape of a word
frequency table, and no Python touches a token.

```python
import batcher as bt
from batcher import col

documents = bt.from_pydict(
    {"text": ["the quick brown fox", "the lazy dog", "the quick dog"]}
)

frequencies = (
    documents.select(word=col("text").str.split(" "))
    .explode("word")
    .group_by("word")
    .agg(n=bt.count())
    .sort("n", "word", descending=[True, False])
)
top = frequencies.to_pydict()
assert top["word"][0] == "the"
assert top["n"][0] == 3
```

Structural detection comes before semantic work. Questions, all-caps, code fences and
markdown are all detectable without a model, and routing on them is how you notice that a
third of a corpus is stack traces before you pay to embed it.

## Every script on this page

The table below lists the multimodal and text scripts in path order.

<!-- library-table: multimodal,text_analytics -->
| Script | Shows |
| --- | --- |
| `examples/multimodal/audio_and_video_metadata.py` | Reading audio and video without decoding the media |
| `examples/multimodal/image_decode_and_resize.py` | Decoding image bytes, and resizing on the way in |
| `examples/multimodal/image_metadata.py` | Filtering images on metadata before paying to decode them |
| `examples/multimodal/image_pipeline.py` | An end-to-end multimodal pipeline: read, filter, decode, score, write |
| `examples/text_analytics/corpus_statistics.py` | Profiling a text corpus before deciding what to do with it |
| `examples/text_analytics/deduplicating_documents.py` | Finding near-duplicate text with a similarity hash |
| `examples/text_analytics/language_shape.py` | Classifying text by shape before classifying it by meaning |
| `examples/text_analytics/ngrams_and_collocations.py` | Word pairs: the bigrams that appear together more than chance |
| `examples/text_analytics/readability_signals.py` | Cheap readability signals over a text column |
| `examples/text_analytics/topic_keywords.py` | Keywords that distinguish one group from the rest |
| `examples/text_analytics/word_frequencies.py` | A word-frequency table over real text, entirely in the engine |
<!-- /library-table -->
