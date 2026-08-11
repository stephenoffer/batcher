# Audio transcription

Whisper wants mono PCM at 16 kHz. Your corpus is stereo MP3s at 44.1 kHz, 48 kHz, and one
directory somebody exported at 8 kHz. Getting from one to the other is where an ASR job
spends its time, usually inside a Python loop calling `librosa`, one file at a time, while
eight GPUs wait.

Batcher decodes and resamples in the data plane: `symphonia` for the decode, a sinc
resampler in Rust for the rate conversion, fanned out across every core. The model stage
then gets batches that are already the shape it asked for.

## Decode and resample, in the data plane

The audio reaches you as a directory of files or as a column of encoded bytes. Both routes end
at the same mono waveform column, and neither one touches Python per sample.

::::{tab-set}
:::{tab-item} A directory of clips

{py:meth}`bt.read.audio(..., decode=True, sample_rate=16000) <batcher.api.io_namespace.reader.Reader.audio>` lists the files, decodes them, and
resamples natively, giving you a `list<float>` waveform column per row.

```python
# docs: skip
import batcher as bt

clips = bt.read.audio("s3://bucket/calls/", decode=True, sample_rate=16000)
```

Without `decode=True` you get the file paths and header metadata (duration, channels, rate)
without touching a sample. That is the cheap way to size a job, or to drop the clips that are
obviously wrong before paying for a decode.
:::

:::{tab-item} A column of encoded bytes

When the audio is already a column of encoded bytes (downloaded, or read from a table), the
{py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>` expressions do the same work in-pipeline. {py:meth}`.audio.to_waveform() <batcher.plan.expr_ir.audio._AudioNamespace.to_waveform>` decodes and
averages the channels down to mono. {py:meth}`.audio.resample(rate) <batcher.plan.expr_ir.audio._AudioNamespace.resample>` decodes and band-limit-resamples
in one native pass, which is the one you want in front of a model with a fixed input rate.

When the model wants **mel features** rather than a raw waveform (Whisper, wav2vec2, HuBERT
all do), {py:meth}`.audio.mel_spectrogram(rate, n_fft=400, hop_length=160, n_mels=80) <batcher.plan.expr_ir.audio._AudioNamespace.mel_spectrogram>` decodes,
resamples, and computes the mel power spectrogram in one native pass. Its output
numerically matches `torchaudio.transforms.MelSpectrogram`, so it drops straight into a
model trained on torchaudio features (apply the model's own log/normalization downstream):

```python
# docs: skip
from batcher import col
# Whisper's front end: 16 kHz, 80 mel bands, in the data plane instead of a per-file UDF.
feats = clips.with_columns(mel=col("bytes").audio.mel_spectrogram(16000, n_mels=80))
```

### Conditioning the waveform first

Three expressions do what a speech pipeline does to a clip before the features are computed,
and they matter more than they look. Recorded clips carry seconds of room tone at each end, and
every one of those samples is paid for twice: once in the spectrogram and again in the model's
sequence length.

{py:meth}`.audio.trim_silence() <batcher.plan.expr_ir.audio._AudioNamespace.trim_silence>` drops the leading and trailing quiet. It leaves interior pauses alone,
because those carry the timing an acoustic model reads — an utterance with its pauses removed is
not the same utterance. A clip that is quiet throughout trims to an empty list, which is how you
find the silent recordings:

```python
# docs: skip
speech = clips.filter(col("bytes").audio.trim_silence().list.len() > 0)
```

{py:meth}`.audio.peak_normalize() <batcher.plan.expr_ir.audio._AudioNamespace.peak_normalize>` scales each clip so its loudest sample sits at full scale. A model
trained on normalized audio reads a quiet recording as a different distribution rather than a
quieter one, so mismatched levels cost accuracy invisibly. It is peak, not loudness (LUFS),
normalization: a clip with one loud click stays quiet everywhere else.

{py:meth}`.audio.zero_crossing_rate() <batcher.plan.expr_ir.audio._AudioNamespace.zero_crossing_rate>` is the cheapest useful descriptor of a waveform and the classic
voiced/unvoiced split — a vowel crosses zero rarely, a fricative constantly. It separates speech
from silence-with-hiss without computing a spectrogram, which makes it a good first-pass filter
over a corpus nobody has curated.

```python
# docs: skip
usable = clips.filter(
    (col("bytes").audio.trim_silence().list.len() > 16000)  # at least a second of sound
    & (col("bytes").audio.zero_crossing_rate() < 0.5)       # not pure noise
)
```

For classical speech models (and many audio classifiers) that want the compact **MFCC**
feature instead, {py:meth}`.audio.mfcc(rate, n_mfcc=13) <batcher.plan.expr_ir.audio._AudioNamespace.mfcc>` runs the whole
`mel → AmplitudeToDB → DCT` chain natively. Its output numerically matches
`torchaudio.transforms.MFCC`:

```python
# docs: skip
mfccs = clips.with_columns(m=col("bytes").audio.mfcc(16000, n_mfcc=13))
```

The clip below is a synthesized stereo WAV, so this runs with no files and no model.

```python
import io
import math
import struct
import wave

import batcher as bt
from batcher import col


def wav(seconds, rate, freq=440.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        n = int(seconds * rate)
        samples = (int(8000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n))
        out.writeframes(b"".join(struct.pack("<hh", s, s) for s in samples))
    return buf.getvalue()


calls = bt.from_pydict(
    {
        "call_id": [1, 2, 3],
        "clip": [wav(1.0, 8000), wav(0.5, 8000), wav(0.02, 8000)],  # the third is a blip
    }
)

prepared = calls.with_columns(
    audio=col("clip").audio.resample(16000),
).with_columns(samples=col("audio").list.len())

print(prepared.select("call_id", "samples").to_pydict())
# {'call_id': [1, 2, 3], 'samples': [16000, 8000, 320]}
```

A one-second clip at 8 kHz came back as 16,000 samples. The resample happened in Rust, on
whole batches, without a `librosa` import anywhere in the pipeline.
:::
::::

Which call to reach for:

| Call | Gives you | Use it when |
| --- | --- | --- |
| `bt.read.audio(path)` | paths plus header metadata: duration, channels, rate | you are sizing the job, or dropping obviously wrong files before a decode |
| `bt.read.audio(path, decode=True, sample_rate=...)` | a decoded, resampled waveform column | the source is a directory of files |
| `col(x).audio.to_waveform()` | mono samples, channels averaged | the bytes are already a column and the rate is already right |
| `col(x).audio.resample(rate)` | mono samples at the rate the model wants | the model has a fixed input rate, which it always does |

## Drop the clips that are not worth transcribing

:::{tip}
Every corpus of call recordings contains silence and 20-millisecond blips from a dropped
connection. Each one costs a full Whisper forward pass and produces nothing.
Filter on the decoded waveform. It is an ordinary list column, so `list.len` gives you the duration
and the whole thing is a scan.
:::

```python
long_enough = prepared.filter(col("samples") >= 16000 * 0.1)  # at least 100 ms at 16 kHz
print(long_enough.to_pydict()["call_id"])
# [1, 2]
```

That filter runs before the GPU stage, so the model never sees the blip. On a real corpus
this routinely removes a double-digit percentage of the rows, and it is free.

## Transcribe

A model stage is a class: the constructor loads the weights once per worker, `__call__`
runs the forward pass on each batch.

:::{warning}
Pass the class. An instance or a plain function reloads the model on every batch, which on a
Whisper-sized model is several seconds of overhead against a forward pass measured in hundreds
of milliseconds.
:::

```python
# docs: skip
import numpy as np
import pyarrow as pa


class Whisper:
    def __init__(self):
        import torch
        from transformers import pipeline

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device="cuda",
            torch_dtype=torch.float16,
        )

    def __call__(self, batch):
        waveforms = [np.asarray(w, dtype="float32") for w in batch.column("audio").to_pylist()]
        results = self.pipe(waveforms, batch_size=len(waveforms))
        texts = [r["text"] for r in results]
        return batch.append_column("transcript", pa.array(texts))


transcribed = long_enough.ml.infer(
    Whisper,
    output_columns=["call_id", "clip", "audio", "samples", "transcript"],
    batch_size=16,
    num_gpus=1,
    concurrency=4,
    model_memory_gb=2.0,
    max_errored_rows=500,  # a truncated file costs one row, not the job
)
transcribed.select("call_id", "transcript").write.parquet("s3://bucket/transcripts.parquet")
```

:::{important}
`max_errored_rows` is the difference between a six-hour job that finishes and one that dies
at hour five on a truncated MP3. A batch whose `fn` raises is bisected, the offending rows
are dropped up to that budget, and the rest of the batch goes through. Beyond the budget the
error propagates, so a genuine bug on clean data still fails fast.
:::

Batcher's audio pipeline (torchaudio mel features into a ResNet-18) runs at **38,546 clip/s**
on 8xT4, and none of that comes from the model. It comes from the decode running in the data
plane and the CPU stage overlapping the GPU stage instead of taking turns.

## Long recordings

Whisper's window is 30 seconds. An hour-long recording has to be windowed before it reaches
the model, and windows are rows: slice the waveform, `explode` into one row per window, keep
the offset so you can stitch the transcript back together in order.

```python
# docs: skip
from batcher import col

windows = (
    prepared.with_columns(window=col("audio").list.slice(0, 16000 * 30))
    .explode("window")
    .with_row_index("window_id")
)
```

Then transcribe the windows and re-aggregate by `call_id`. The transcription stage does not
know or care that a row is a window, and that is the property that makes any of this
composable.

## What to check

:::{dropdown} The four things to check when the GPUs are still waiting
- The decode is an engine expression or `read.audio(decode=True)`, not a Python loop.
- The rate conversion happens once, in the engine, not per batch in the model's `__call__`.
- Short and silent clips are filtered *before* the GPU stage.
- The model stage got a class, so the weights load once per worker.
:::

## See also

- {doc}`Multimodal </ml/preparing/multimodal/index>`: `.audio` expressions, tensor columns, blob offload.
- {doc}`Inference </ml/inference/inference>`: pools, stage overlap, and adaptive batch sizing.
- {doc}`GPU scheduling </ml/inference/gpu>`: `num_gpus`, `concurrency`, and `model_memory_gb`.
- {doc}`Image classification </cookbook/ml/pipelines/multimodal/image-classification>`: the same decode → model shape, for pixels.
- {doc}`Image captioning </cookbook/ml/pipelines/multimodal/image-captioning>`: the same shape again, with a vision-language model.
- {doc}`ML API reference </api/models/ml>`: {py:meth}`bt.read.audio <batcher.api.io_namespace.reader.Reader.audio>`, {py:meth}`ds.ml.infer <batcher.api.dataset.ml.DatasetML.infer>`, `max_errored_rows`.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: where the 38,546 clip/s on the
  audio pipeline comes from.
- {doc}`GPU execution </architecture/deep-dives/distribution/gpu-execution>`: the CPU/GPU overlap that produces it.
- {doc}`HuggingFace integration </integrations/compute/huggingface>`: loading Whisper and friends.
