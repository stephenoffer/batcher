"""The `.audio` expression namespace — lazy, batch-level audio decode.

`AudioFunc` lowers to ``{"e": "audio", "fn": ...}`` IR consumed by Rust
`Expr::Audio` (symphonia-backed). Like image decode, the interpreter is the oracle
and the JIT falls back; one implementation, so the tiers cannot diverge. This moves
audio decode off the per-row Python ``map_batches`` path into the native data plane.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError, require_int
from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.fn_names import AUDIO_FNS
from batcher.plan.expr_ir.node_base import IRNode, child, expr_node, scalar
from batcher.plan.ir_tags import ExprTag

__all__ = ["AudioFunc", "_AudioNamespace"]


@expr_node
class AudioFunc(IRNode):
    """An audio decode op over a binary (audio-bytes) sub-expression (via `.audio`).

    `decode` reads each clip's metadata; `to_waveform` decodes to a mono signal;
    `resample` decodes then band-limited-resamples that signal to `rate` Hz;
    `mel_spectrogram` produces the speech-model mel power-spectrogram front end; `mfcc`
    produces the classic MFCC feature. `trim_silence`, `peak_normalize` and
    `zero_crossing_rate` condition or describe the waveform itself.
    """

    tag = ExprTag.AUDIO
    vocab = AUDIO_FNS
    fn: str = scalar()
    input: Expr = child()
    rate: int | None = scalar(omit_none=True, default=None)
    # `mel_spectrogram` / `mfcc` — STFT/filterbank sizes. Omitted from the IR unless set, so
    # the other audio ops' wire shape is unchanged.
    n_fft: int | None = scalar(omit_none=True, default=None)
    hop_length: int | None = scalar(omit_none=True, default=None)
    n_mels: int | None = scalar(omit_none=True, default=None)
    n_mfcc: int | None = scalar(omit_none=True, default=None)  # `mfcc` only
    threshold_db: int | None = scalar(omit_none=True, default=None)  # `trim_silence` only


class _AudioNamespace:
    """Lazy audio decode: ``col("bytes").audio.decode()`` / ``.audio.to_waveform()``.

    Decoding runs in the Rust data plane over a binary column (symphonia-backed), so an
    audio pipeline never materializes samples in Python. Null or undecodable input —
    including bytes that are not audio at all — yields null rather than raising.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"clip": [b"not audio"]})
            >>> ds.select(meta=bt.col("clip").audio.decode()).to_pydict()
            {'meta': [None]}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.audio` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.audio accessor of col('c')>``."""
        return f"<.audio accessor of {self._e!r}>"

    def decode(self) -> AudioFunc:
        """Read each clip's metadata without materializing its samples.

        Returns:
            An expression evaluating to a struct ``{sample_rate, channels, num_frames,
            duration_secs}`` (WAV/FLAC); null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(m=bt.col("bytes").audio.decode()).to_pydict()  # doctest: +SKIP
                {'m': [{'sample_rate': 44100, 'channels': 2, ...}]}
        """
        return AudioFunc("decode", self._e)

    def to_waveform(self) -> AudioFunc:
        """Decode to a mono PCM signal by averaging channels.

        The training-ingest path: it produces a numeric column that feeds a model
        directly, with no per-row Python.

        Returns:
            An expression evaluating to a ``List<Float32>`` of channel-averaged
            samples; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(w=bt.col("bytes").audio.to_waveform())  # doctest: +SKIP
        """
        return AudioFunc("to_waveform", self._e)

    def resample(self, rate: int) -> AudioFunc:
        """Decode to mono and band-limited-resample to ``rate`` Hz.

        The audio-ML preprocessing step (models expect a fixed rate — 16 kHz for
        Whisper/wav2vec, 22 kHz for many audio models). Sinc resampling runs natively in
        the data plane over the whole batch, replacing a per-file Python ``librosa`` call.
        The output length is ``ceil(n * rate / source_rate)`` — the length ``librosa``
        produces — so a resampled frame count is reproducible and engine-independent.

        Args:
            rate: The target sample rate in Hz (must be positive).

        Returns:
            An expression evaluating to a ``List<Float32>`` of resampled mono samples;
            null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(w=bt.col("bytes").audio.resample(16000))  # doctest: +SKIP
        """
        return AudioFunc("resample", self._e, rate=rate)

    def trim_silence(self, threshold_db: int = -40) -> AudioFunc:
        """Decode to mono and drop the leading and trailing quiet.

        The first step of an ASR pipeline. Recorded clips carry seconds of room tone at each
        end, and every one of those samples is paid for twice — once in the spectrogram and
        again in the model's sequence length.

        Only the *ends* are trimmed. Interior pauses carry the timing an acoustic model reads,
        so an utterance with its pauses removed is not the same utterance. A clip that is quiet
        throughout trims to an empty list, which is how you filter silent recordings out:
        ``filter(col("bytes").audio.trim_silence().list.len() > 0)``.

        The threshold is in dBFS relative to full scale, so it is independent of the recording
        level. The default of -40 (1% of full scale) is the conventional floor: quiet enough to
        keep a soft consonant, loud enough to drop room tone.

        Args:
            threshold_db: The silence floor in dBFS. Must be at most 0, since 0 is full scale.

        Returns:
            An expression evaluating to a ``List<Float32>`` of the trimmed mono samples;
            null for null or undecodable input.

        Raises:
            PlanError: If `threshold_db` is above 0.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(w=bt.col("bytes").audio.trim_silence())  # doctest: +SKIP
        """
        threshold_db = require_int(threshold_db, func="audio.trim_silence", arg="threshold_db")
        if threshold_db > 0:
            raise PlanError(
                f"audio.trim_silence(): threshold_db is dBFS relative to full scale, so it "
                f"must be at most 0, got {threshold_db}"
            )
        return AudioFunc("trim_silence", self._e, threshold_db=threshold_db)

    def peak_normalize(self) -> AudioFunc:
        """Decode to mono and scale so the loudest sample sits at full scale.

        The level-matching step before batching clips from different sources. A model trained
        on normalized audio reads a quiet recording as a different distribution rather than a
        quieter one, so mismatched levels cost accuracy in a way that is invisible in the data.

        This is *peak* normalization, not loudness (LUFS) normalization: it equalizes the
        maximum sample, not the perceived level, so a clip containing one loud click stays
        quiet everywhere else. An all-zero clip is returned unchanged rather than amplified.

        Returns:
            An expression evaluating to a ``List<Float32>`` of the normalized mono samples;
            null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(w=bt.col("bytes").audio.peak_normalize())  # doctest: +SKIP
        """
        return AudioFunc("peak_normalize", self._e)

    def zero_crossing_rate(self) -> AudioFunc:
        """The fraction of adjacent samples that change sign (→ Float64).

        The cheapest useful descriptor of a waveform, and the classic voiced/unvoiced split: a
        vowel is low-frequency and crosses zero rarely, a fricative or noise crosses constantly.
        It separates speech from silence-with-hiss without computing a spectrogram, which makes
        it a good first-pass filter over a corpus nobody has curated.

        A clip shorter than two samples has no adjacent pair and yields null.

        Returns:
            An expression evaluating to a Float64 crossing rate in ``[0, 1]``; null for null
            or undecodable input, and for a clip shorter than two samples.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(z=bt.col("bytes").audio.zero_crossing_rate())  # doctest: +SKIP
        """
        return AudioFunc("zero_crossing_rate", self._e)

    def mel_spectrogram(
        self,
        rate: int,
        *,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
    ) -> AudioFunc:
        """Decode, resample, and compute the mel **power** spectrogram — the speech front end.

        The native front end for Whisper / wav2vec2 / HuBERT: decode to mono, resample to
        ``rate`` Hz, STFT (periodic Hann window, centered reflect padding), power spectrum,
        then an HTK-scale mel filterbank. Runs in the data plane over the whole batch,
        replacing a per-file Python ``torchaudio`` / ``librosa`` call. The output
        **numerically matches** ``torchaudio.transforms.MelSpectrogram`` defaults
        (``power=2.0``, ``norm=None``, ``mel_scale="htk"``, ``center=True``,
        ``pad_mode="reflect"``); the log/normalization step varies by model, so apply it
        downstream. The defaults are Whisper's (16 kHz, ``n_fft=400``, ``hop=160``,
        ``n_mels=80``).

        Args:
            rate: Target sample rate in Hz to resample to before the STFT (must be positive).
            n_fft: STFT window size (and FFT length).
            hop_length: Samples between successive STFT frames.
            n_mels: Number of mel filterbank bands.

        Returns:
            An expression evaluating to a ``List<Float32>`` of ``n_mels * n_frames`` values
            in row-major ``(n_mels, n_frames)`` order (reshape by ``n_mels``); null for null
            or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> mel = bt.col("bytes").audio.mel_spectrogram(16000, n_mels=80)  # doctest: +SKIP
                >>> ds.select(m=mel)  # doctest: +SKIP
        """
        return AudioFunc(
            "mel_spectrogram",
            self._e,
            rate=rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def mfcc(
        self,
        rate: int,
        *,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        n_mfcc: int = 40,
    ) -> AudioFunc:
        """Decode, resample, and compute MFCCs — the classic compact speech feature.

        Mel-Frequency Cepstral Coefficients: the mel power spectrogram, converted to dB
        (`AmplitudeToDB`), then an orthonormal DCT-II keeping the first ``n_mfcc``
        coefficients. Runs natively in the data plane, replacing a per-file Python
        ``torchaudio``/``librosa`` call. The output **numerically matches**
        ``torchaudio.transforms.MFCC`` defaults. The defaults here (128 mels, 40 coeffs)
        are torchaudio's.

        Args:
            rate: Target sample rate in Hz to resample to before the STFT (must be positive).
            n_fft: STFT window size (and FFT length).
            hop_length: Samples between successive STFT frames.
            n_mels: Number of mel filterbank bands.
            n_mfcc: Number of cepstral coefficients to keep (must be ``<= n_mels``).

        Returns:
            An expression evaluating to a ``List<Float32>`` of ``n_mfcc * n_frames`` values
            in row-major ``(n_mfcc, n_frames)`` order (reshape by ``n_mfcc``); null for null
            or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.audio("s3://bucket/clips/")  # doctest: +SKIP
                >>> feats = bt.col("bytes").audio.mfcc(16000, n_mfcc=13)  # doctest: +SKIP
                >>> ds.select(m=feats)  # doctest: +SKIP
        """
        return AudioFunc(
            "mfcc",
            self._e,
            rate=rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
        )
