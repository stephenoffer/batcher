# Prepare the data

This section covers the work between a raw source and a model: feature preprocessing for tabular models, decoding images, audio and video into tensors, and tokenizing text.

Preparation is usually where an ML pipeline spends its CPU and its engineering time. In Batcher it is ordinary plan work. Every step on these pages is an expression or an operator, so it streams in bounded memory, runs on every core, and distributes across a cluster without a second code path. Nothing gets pulled onto the driver to be looped over in Python.

## Tabular features: fit once, replay anywhere

The preprocessors follow the scikit-learn contract you already know, with one difference that matters at scale. `fit` learns its state with one mergeable aggregate over the data, so fitting a scaler on a billion rows is a single distributed pass rather than a sample pulled into memory. `transform` bakes the learned values into an expression and stays lazy.

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

The encoder learned its categories from the training rows, so the held-out rows get exactly those indicator columns. That is the leak-free behavior an offline score depends on, and a fitted `Chain` saves to one file that serving can load.

## Media: decode in the engine

Images, audio and video decode through expressions on the `.image`, `.audio` and `.video` namespaces, implemented in Rust rather than as a Python UDF per row. An image goes from encoded bytes to a normalized `float32` tensor in one expression, audio becomes a mel spectrogram that matches `torchaudio`, and curation measures such as sharpness, entropy and perceptual hashes run as predicates you can filter on. The block below needs image files, so it is shown but not executed:

```python
# docs: skip
import batcher as bt

images = bt.read.images("s3://<your-bucket>/images/")
tensors = images.with_columns(
    pixels=bt.col("bytes").image.to_tensor_f32(
        224, 224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channels_first=True
    )
)
```

On one 96-core machine, decoding and resizing 2,000 JPEGs to 224x224 ran at 4,649-4,788 img/s, 1.87-1.96x Daft and 6.35-6.61x Ray Data on the same corpus. {doc}`/benchmarks/results/multimodal-ingest` has the full measurement.

## Text: tokens as a column

A tokenizer in the training loop leaves the GPU waiting on CPU work. As a pipeline stage it runs once, in parallel, and writes token ids to disk for every later epoch. {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>` drives a Hugging Face fast tokenizer once per Arrow batch, and {py:func}`pack_sequences <batcher.ml.pack_sequences>` packs the result into dense fixed-length blocks for causal-LM pretraining.

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
