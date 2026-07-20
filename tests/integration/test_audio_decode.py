"""Native audio decode (`.audio` accessor) — moves WAV/FLAC decode off the per-row
Python `map_batches` path into the Rust data plane (symphonia).

No DuckDB oracle for audio; we hand-encode a minimal PCM WAV and assert the decoded
metadata and waveform.
"""

from __future__ import annotations

import struct

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration


def _wav(sample_rate: int, samples: list[int]) -> bytes:
    """A minimal mono 16-bit PCM WAV."""
    data = b"".join(struct.pack("<h", s) for s in samples)
    fmt = struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
    )
    return header + data


def test_audio_decode_metadata():
    ds = bt.from_pydict({"a": [_wav(16000, [0, 100, -100, 0, 50]), None]})
    out = ds.select(d=col("a").audio.decode()).collect().to_pydict()["d"]
    assert out[0] == {
        "sample_rate": 16000,
        "channels": 1,
        "num_frames": 5,
        "duration_secs": 5 / 16000,
    }
    assert out[1] is None  # null bytes → null struct


def test_audio_to_waveform():
    # 16384/32768 = 0.5 normalized; -16384/32768 = -0.5.
    ds = bt.from_pydict({"a": [_wav(8000, [0, 16384, -16384]), b"not audio"]})
    out = ds.select(w=col("a").audio.to_waveform()).collect().to_pydict()["w"]
    assert [round(x, 3) for x in out[0]] == [0.0, 0.5, -0.5]
    assert out[1] is None  # undecodable bytes → null list


def test_audio_dataset_helper_takes_native_path():
    # The default (mono, native rate) `audio_dataset` must decode in the data plane
    # — no per-row Python `map_batches` — and match the native expression's output.
    from batcher.core.udf import has_map_batches
    from batcher.ml.decode import audio_dataset

    ds = bt.from_pydict({"bytes": [_wav(8000, [0, 16384, -16384]), None]})
    decoded = audio_dataset(ds)
    assert not has_map_batches(decoded._plan), "default audio_dataset must not use map_batches"
    out = decoded.collect().to_pydict()["waveform"]
    assert [round(x, 3) for x in out[0]] == [0.0, 0.5, -0.5]
    assert out[1] is None


def test_audio_resample_expression():
    # `.audio.resample` decodes + sinc-resamples natively; output length is the librosa
    # length, ceil(n * target / source). 400 frames at 8 kHz -> 800 at 16 kHz (upsample),
    # -> 200 at 4 kHz (downsample). Undecodable input -> null.
    samples = [int(16384 * (i % 7 - 3) / 3) for i in range(400)]
    ds = bt.from_pydict({"a": [_wav(8000, samples), b"not audio"]})
    up = ds.select(w=col("a").audio.resample(16000)).collect().to_pydict()["w"]
    assert len(up[0]) == 800
    assert up[1] is None
    down = ds.select(w=col("a").audio.resample(4000)).collect().to_pydict()["w"]
    assert len(down[0]) == 200
    # Resampling to the source rate is an exact passthrough.
    same = ds.select(w=col("a").audio.resample(8000)).collect().to_pydict()["w"]
    assert len(same[0]) == 400


def test_audio_dataset_resample_takes_native_path():
    # An explicit mono sample_rate now resamples natively (Rust sinc) — no per-row Python.
    from batcher.core.udf import has_map_batches
    from batcher.ml.decode import audio_dataset

    ds = bt.from_pydict({"bytes": [_wav(8000, [int(16384 * (i % 5 - 2) / 2) for i in range(200)])]})
    decoded = audio_dataset(ds, sample_rate=16000)
    assert not has_map_batches(decoded._plan), "mono resample must take the native path"
    out = decoded.collect().to_pydict()["waveform"]
    assert len(out[0]) == 400  # 200 frames at 8 kHz -> 400 at 16 kHz


def _wav_from_floats(sample_rate: int, floats: list[float]) -> bytes:
    """A mono 16-bit PCM WAV from float samples in [-1, 1] (rounded to int16)."""
    ints = [max(-32768, min(32767, round(f * 32767))) for f in floats]
    return _wav(sample_rate, ints)


def test_mel_spectrogram_shape_and_reshape():
    import math

    sr, n_fft, hop, n_mels = 16000, 400, 160, 80
    n = 4000  # 0.25 s
    sig = [0.3 * math.sin(2 * math.pi * 440 * i / sr) for i in range(n)]
    ds = bt.from_pydict({"a": [_wav_from_floats(sr, sig), b"not audio"]})
    mel = col("a").audio.mel_spectrogram(sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    out = ds.select(m=mel).collect().to_pydict()["m"]
    # center=True: n_frames = 1 + n/hop = 1 + 4000/160 = 26; row length = 80 * 26.
    n_frames = 1 + n // hop
    assert len(out[0]) == n_mels * n_frames
    assert out[1] is None  # undecodable → null


def test_mel_spectrogram_matches_torchaudio():
    """The load-bearing correctness check: numerically match torchaudio's
    MelSpectrogram (the oracle a wrong window/padding/filterbank convention fails)."""
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    import math

    import numpy as np

    sr, n_fft, hop, n_mels = 16000, 400, 160, 80
    n = 8000
    # A couple of tones + a little broadband so every mel band sees some energy.
    sig = [
        0.5 * math.sin(2 * math.pi * 440 * i / sr)
        + 0.3 * math.sin(2 * math.pi * 2500 * i / sr)
        + 0.1 * math.sin(2 * math.pi * 6000 * i / sr)
        for i in range(n)
    ]
    wav = _wav_from_floats(sr, sig)

    # Batcher (decode + rate=sr passthrough + mel).
    out = (
        bt.from_pydict({"a": [wav]})
        .select(m=col("a").audio.mel_spectrogram(sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels))
        .collect()
        .to_pydict()["m"][0]
    )
    n_frames = 1 + n // hop
    got = np.asarray(out, dtype=np.float64).reshape(n_mels, n_frames)

    # Reconstruct the exact waveform symphonia decodes: the int16 PCM the WAV holds,
    # divided by 32768 (the standard i16→f32 convention). torchaudio 2.11 moved decoding
    # to the optional `torchcodec`, so we build the tensor directly rather than `.load()`.
    int16 = [max(-32768, min(32767, round(f * 32767))) for f in sig]
    waveform = torch.tensor([[v / 32768.0 for v in int16]], dtype=torch.float32)
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
        norm=None,
        mel_scale="htk",
        center=True,
        pad_mode="reflect",
    )
    exp = melspec(waveform)[0].numpy().astype(np.float64)

    assert exp.shape == got.shape, (exp.shape, got.shape)
    # Relative Frobenius error catches any convention mismatch (window/pad/filterbank/power);
    # f32 FFT accumulation order + PCM16 quantization keep genuine matches well under 2%.
    rel = np.linalg.norm(got - exp) / (np.linalg.norm(exp) + 1e-12)
    # A matching implementation lands ~1e-6 (only f32 FFT accumulation order + PCM16
    # quantization differ); any window/pad/filterbank/power convention bug is ~1-100%.
    assert rel < 5e-3, f"relative error {rel:.4f} too high — a convention likely differs"


def test_mfcc_shape_and_reshape():
    import math

    sr, n_fft, hop, n_mels, n_mfcc = 16000, 400, 160, 40, 13
    n = 4000
    sig = [0.3 * math.sin(2 * math.pi * 440 * i / sr) for i in range(n)]
    ds = bt.from_pydict({"a": [_wav_from_floats(sr, sig), b"not audio"]})
    feats = col("a").audio.mfcc(sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, n_mfcc=n_mfcc)
    out = ds.select(m=feats).collect().to_pydict()["m"]
    n_frames = 1 + n // hop
    assert len(out[0]) == n_mfcc * n_frames
    assert out[1] is None


def test_mfcc_matches_torchaudio():
    """MFCC numerically matches torchaudio.transforms.MFCC (the oracle for the whole
    mel → AmplitudeToDB → DCT chain)."""
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    import math

    import numpy as np

    sr, n_fft, hop, n_mels, n_mfcc = 16000, 400, 160, 40, 13
    n = 8000
    sig = [
        0.5 * math.sin(2 * math.pi * 440 * i / sr)
        + 0.3 * math.sin(2 * math.pi * 2500 * i / sr)
        + 0.1 * math.sin(2 * math.pi * 6000 * i / sr)
        for i in range(n)
    ]
    wav = _wav_from_floats(sr, sig)

    feats = col("a").audio.mfcc(sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, n_mfcc=n_mfcc)
    out = bt.from_pydict({"a": [wav]}).select(m=feats).collect().to_pydict()["m"][0]
    n_frames = 1 + n // hop
    got = np.asarray(out, dtype=np.float64).reshape(n_mfcc, n_frames)

    int16 = [max(-32768, min(32767, round(f * 32767))) for f in sig]
    waveform = torch.tensor([[v / 32768.0 for v in int16]], dtype=torch.float32)
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        dct_type=2,
        norm="ortho",
        log_mels=False,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop,
            "n_mels": n_mels,
            "center": True,
            "pad_mode": "reflect",
            "power": 2.0,
            "norm": None,
            "mel_scale": "htk",
        },
    )
    exp = mfcc(waveform)[0].numpy().astype(np.float64)

    assert exp.shape == got.shape, (exp.shape, got.shape)
    rel = np.linalg.norm(got - exp) / (np.linalg.norm(exp) + 1e-12)
    assert rel < 5e-3, f"relative error {rel:.4f} too high — a convention likely differs"
