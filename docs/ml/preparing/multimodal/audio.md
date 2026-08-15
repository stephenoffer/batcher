# Preparing an audio corpus

This page covers what a speech or audio pipeline does to a clip between reading its bytes and handing it to a model: measuring how it was recorded, putting every row on the same footing, and turning it into the features a model consumes. Each step is an expression over a binary column, so a corpus of clips never becomes a per-file Python loop over a million floats.

## Measuring how a corpus was recorded

Recording quality is the axis a scraped audio corpus varies on most and reports least. Six measures reduce a clip to one number, so triaging is an ordinary predicate:

```python
import base64
import batcher as bt

clip = base64.b64decode(
    "UklGRuQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YcAAAAAAQADAAEAAwABAAMAAQADA"
    "AEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADA"
    "AEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADA"
    "AEAAwABAAMAAQADAAEAAwABAAMA="
)
clips = bt.from_pydict({"bytes": [clip]})

print(
    clips.select(
        level=bt.col("bytes").audio.rms(),
        loud=bt.col("bytes").audio.dbfs().round(2),
        hot=bt.col("bytes").audio.clipping_ratio(),
    ).to_pydict()
)
```

| Method | Reports | Why it is separate from the others |
|---|---|---|
| `rms()` | root-mean-square amplitude in `0..1` | tracks perceived loudness; a recording with one door slam has a peak of 1.0 and an RMS that still says quiet |
| `dbfs()` | the same level in decibels below full scale | the unit every audio tool states a threshold in |
| `peak_dbfs()` | the loudest single sample, in dBFS | paired with `dbfs()` it is the crest factor, which separates a compressed broadcast recording from a natural one |
| `clipping_ratio(threshold)` | fraction of samples at the rail | distortion no normalization can undo, and invisible to every other measure because normalizing makes it *look* well-levelled |
| `silence_ratio(threshold_db)` | fraction of samples below a floor | finds the recordings that are mostly dead air |
| `zero_crossing_rate()` | fraction of adjacent pairs that change sign | the classic voiced/unvoiced descriptor |

`dbfs()` and `peak_dbfs()` return null for digital silence rather than negative infinity. That is deliberate: an infinity compares less than every threshold, so a silent clip would pass every "quieter than X" filter *and* every "louder than X" one written with a negated comparison.

## Putting every clip on the same footing

Clips from different sources differ in level, length and sample rate, and a model sees each of those as a different distribution rather than a different recording.

`rms_normalize(target_db=-20)` matches loudness. It is usually the one you want over `peak_normalize()`, which equalizes the *maximum* — so a clip with one loud click stays quiet everywhere else. The gain is capped so the result cannot clip, which means a whisper is lifted toward the target rather than driven into the rails.

`pad_or_trim(duration_secs, rate)` is the operation that makes a clip corpus batchable at all. Whisper requires exactly 30 seconds of 16 kHz audio and every other fixed-input audio model requires something like it, so without it a pipeline either loops in Python or hands the model rows of unequal length:

```python
# docs: skip
from batcher import col

fixed = clips.with_columns(
    audio=col("bytes").audio.trim_silence().audio.pad_or_trim(30.0, 16000)
)
```

`slice(offset_secs, duration_secs)` extracts a region, measured against the clip's own sample rate. A window past the end of the recording yields an empty list rather than null, because an empty region is a fact about the window and not a failure to read the clip.

`pre_emphasis(coefficient=0.97)` applies the first-order high-pass every classical ASR front end runs before framing, to flatten the spectral tilt of voiced speech.

These compose. `trim_silence()` hands back a waveform and `rms_normalize()` reads one, so a two-step clean is one expression rather than a decode, a round trip through Python, and a re-encode:

```python
# docs: skip
from batcher import col

cleaned = clips.select(
    audio=col("bytes").audio.trim_silence().audio.rms_normalize().audio.encode_wav(16000)
)
```

## Writing audio back out

Every waveform method hands back a `List<Float32>`, which is what a model wants and what nothing else can read. `encode_wav(rate=None)` closes the loop, producing a mono 16-bit PCM WAV container — the format every player, dataset loader and annotation tool accepts:

```python
# docs: skip
from batcher import col

cleaned = clips.select(
    uri=col("uri"),
    wav=col("bytes").audio.trim_silence().audio.rms_normalize().audio.encode_wav(16000),
)
cleaned.write.parquet("s3://bucket/cleaned/")
```

Without it, writing a trimmed and normalized corpus back to storage *as audio* meant encoding each row in Python.

## Spectral features

`mel_spectrogram` and `mfcc` are the speech front ends, matching `torchaudio`'s defaults so the result is model-drop-in. `spectrogram(rate, n_fft=, hop_length=)` is their unwarped sibling: a mel filterbank is tuned to human pitch perception, and a music, bioacoustic or machine-fault model wants the frequencies themselves.

Four descriptors reduce the whole clip to one number instead:

| Method | Reports |
|---|---|
| `spectral_centroid(rate)` | the energy-weighted mean frequency — the standard brightness descriptor |
| `spectral_rolloff(rate, percentile=0.85)` | the frequency below which most of the energy lies |
| `spectral_bandwidth(rate)` | the spread of frequencies about the centroid |
| `spectral_flatness(rate)` | geometric over arithmetic mean of the power spectrum: near 0 for a tone, near 1 for noise |

`spectral_rolloff` is the one worth reaching for first on an unknown corpus. It says where a recording's usable band *ends*, which is how an 8 kHz telephone recording upsampled to 16 kHz is caught: it has a full-rate header, ordinary loudness, and no energy above 4 kHz. Nothing else here can see that, and a model trained on the mixture learns the artifact.

All four average over frames and skip frames carrying no energy. A silent frame counted as "0 Hz" would drag the average toward DC and make a mostly-quiet recording look band-limited, which is the exact confusion the measure exists to resolve.

## Requirements and limitations

- Decoding is WAV/PCM and FLAC. Compressed formats are read through `ds.ml.decode_audio`, which uses `soundfile`.
- The methods that need no sample rate — the level and hygiene measures, `trim_silence`, `peak_normalize`, `rms_normalize`, `pre_emphasis`, and `encode_wav` — also accept a waveform column, so they chain without re-decoding. The ones defined against a rate (`resample`, `slice`, `pad_or_trim`, and the spectral front ends) need encoded bytes, because a waveform carries no sample rate; handed one, they say which method and what to do.
- The waveform methods return a *variable*-length list, including `pad_or_trim`, whose length is knowable but whose column type stays `list<float32>` so the engine's declared schema matches the column it describes.
- Multi-channel audio is averaged to mono at decode.

## See also

- {doc}`/ml/preparing/multimodal/decoding`: getting the bytes and decoding them.
- {doc}`/ml/preparing/multimodal/curating`: the same triage question for images.
- {doc}`/api/relational/expression-accessors`: the full `.audio` method list.
