# Prepare the data

This section covers the work between a raw source and a model: feature preprocessing for tabular models, decoding images, audio and video into tensors, and tokenizing text.

In Batcher preparation is ordinary plan work. Every step on these pages is an expression or an operator, so it streams in bounded memory, runs on every core, and distributes without a second code path.

## Tabular features: fit once, replay anywhere

The preprocessors follow the scikit-learn contract. `fit` learns its state with one mergeable aggregate, so fitting a scaler on a billion rows is a single distributed pass. `transform` bakes the learned values into an expression and stays lazy.

The example below splits by a stable key, fits an imputer, a scaler and an encoder on the training rows only, and applies the same learned state to the held-out rows:

```python
import batcher as bt
from batcher.ml.preprocessors import Chain, OneHotEncoder, SimpleImputer, StandardScaler

ds = bt.from_pydict(
    {
        "id": list(range(8)),
        "age": [20.0, None, 40.0, 50.0, 30.0, 60.0, None, 45.0],
        "city": ["oslo", "rome", "oslo", "paris", "rome", "oslo", "paris", "rome"],
    }
)
train, test = ds.ml.train_test_split(0.25, seed=7, key="id")

prep = Chain(SimpleImputer(["age"], strategy="median"), StandardScaler(["age"]), OneHotEncoder(["city"]))
prep.fit(train)
print(prep.transform(test).columns)
# ['id', 'age', 'city_paris', 'city_rome']
```

The encoder learned its categories from the training rows, so the held-out rows get exactly those indicator columns. A fitted `Chain` saves to one file that serving can load.

## Media: decode in the engine

Images, audio and video decode through expressions on the `.image`, `.audio` and `.video` namespaces, implemented in Rust. Curation measures run as predicates you can filter on:

```python
import io

import numpy as np
from PIL import Image


def png(pixels):
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


bright = np.full((8, 8, 3), 200, dtype="uint8")
photos = bt.from_pydict({"id": [1, 2], "bytes": [png(bright), png(bright // 20)]})
print(photos.select("id", lum=bt.col("bytes").image.brightness().round(2)).to_pydict())
# {'id': [1, 2], 'lum': [0.78, 0.04]}
```

An image goes from encoded bytes to a normalized `float32` tensor in one expression. The block below needs image files, so it is shown but not executed:

```python
# docs: skip
images = bt.read.images("s3://<your-bucket>/images/")
tensors = images.with_columns(
    pixels=bt.col("bytes").image.to_tensor_f32(
        224, 224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channels_first=True
    )
)
```

## Text: tokens as a column

A tokenizer in the training loop leaves the GPU waiting on CPU work. As a pipeline stage it runs once, in parallel. {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>` takes a Hugging Face fast tokenizer or any `str -> list` callable:

```python
from batcher.ml import Tokenizer

notes = bt.from_pydict({"id": [1, 2], "text": ["hello world", "one two three"]})
tok = Tokenizer("text", lambda s: s.split(), output_column="tokens")
print(tok.fit_transform(notes).to_pydict()["tokens"])
# [['hello', 'world'], ['one', 'two', 'three']]
```

{py:func}`pack_sequences <batcher.ml.pack_sequences>` packs token lists into dense fixed-length blocks for causal-LM pretraining, with an EOS at each seam:

```python
from batcher.ml import pack_sequences

corpus = bt.from_pydict({"tokens": [[1, 2, 3], [4, 5], [6, 7, 8, 9]]})
packed = list(pack_sequences(corpus.iter_batches(), token_column="tokens", seq_len=4, eos_token=0))
print(packed[0].to_pydict())
# {'tokens': [[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]]}
```

## In this section

The following table lists the pages and sub-sections here:

| Page | Covers |
|---|---|
| {doc}`/ml/preparing/preprocessors/index` | Scalers, encoders, imputers, binning, text vectorizers, feature generation and selection, chaining, and persistence. |
| {doc}`/ml/preparing/multimodal/index` | Fetching media, decoding it into tensor columns, augmenting and curating it, and moving it through a pipeline. |
| {doc}`/ml/preparing/tokenization` | The `Tokenizer` preprocessor, token ids as a list column, and sequence packing for pretraining. |

## See also

- {doc}`/getting-started/tutorials/ml/feature-engineering`: a raw table to a model-ready matrix, end to end.
- {doc}`/ml/training/training-corpus`: mixing, filtering and decontaminating a text corpus before training.
- {doc}`/ml/inference/index`: running a model over the prepared columns.
- {doc}`/api/models/preprocessors`: the complete preprocessor reference.

```{toctree}
:hidden:

preprocessors/index
multimodal/index
tokenization
```
