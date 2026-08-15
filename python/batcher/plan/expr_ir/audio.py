"""The `.audio` expression namespace — lazy, batch-level audio decode.

`AudioFunc` lowers to ``{"e": "audio", "fn": ...}`` IR consumed by Rust
`Expr::Audio` (symphonia-backed). Like image decode, the interpreter is the oracle
and the JIT falls back; one implementation, so the tiers cannot diverge. This moves
audio decode off the per-row Python ``map_batches`` path into the native data plane.
"""

from __future__ import annotations

import base64

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
    # `trim_silence`/`silence_ratio`'s floor; `rms_normalize` reuses it as its target level.
    threshold_db: int | None = scalar(omit_none=True, default=None)
    # The one fractional knob the level and spectral ops take, named per op: `pre_emphasis`'s
    # coefficient, `clipping_ratio`'s full-scale fraction, `spectral_rolloff`'s percentile.
    # One slot rather than three, because an op reads exactly one and three would be two
    # nulls in every audio plan.
    factor: float | None = scalar(omit_none=True, default=None)
    # `slice` only: where the window starts, in seconds.
    offset_secs: float | None = scalar(omit_none=True, default=None)
    # `slice`/`pad_or_trim`: how long the window is, in seconds.
    duration_secs: float | None = scalar(omit_none=True, default=None)


# A 96-sample, 8 kHz mono WAV alternating between +0.5 and -0.5 full scale. Exported so the
# doctests here (and the docs) have real audio to decode without a fixture file: its RMS is
# exactly 0.5, its peak exactly 0.5, and every sample crosses zero, which makes the level
# and waveform measures show round numbers rather than encoder noise.
_WAV_ALTERNATING = base64.b64decode(
    "UklGRuQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YcAAAAAAQADAAEAAwABAAMAAQADAAEAAwABA"
    "AMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADA"
    "AEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABA"
    "AMAAQADAAEAAwABAAMAAQADAAEAAwABAAMAAQADAAEAAwABAAMA="
)


def _stft_args(
    func: str, rate: int, n_fft: int, hop_length: int, n_mels: int | None = None
) -> dict[str, int]:
    """Coerce and bound-check the STFT framing every spectral op shares.

    Both docstrings already promised these bounds and neither enforced one, so the engine
    caught them instead -- which means a mistyped sample rate surfaced as a `RuntimeError`
    from the data plane after the scan had run, and in a distributed job after the work had
    been scheduled. `.video` validates at plan build; this is `.audio` doing the same, on
    the arguments both spectral ops take rather than once per op.

    Args:
        func: Dotted method name for the message, such as ``"audio.mfcc"``.
        rate: Target sample rate in Hz.
        n_fft: STFT window size.
        hop_length: Samples between successive STFT frames.
        n_mels: Number of mel filterbank bands, for the two ops that warp to mel. Omitted
            by the linear spectrogram and the four descriptors, which read the frequency
            bins themselves.

    Returns:
        The arguments as the IR keywords they become, as plain `int`s.

    Raises:
        PlanError: If any of them is not an integer of at least 1.
    """
    args = {
        "rate": require_int(rate, func=func, arg="rate", minimum=1),
        "n_fft": require_int(n_fft, func=func, arg="n_fft", minimum=1),
        "hop_length": require_int(hop_length, func=func, arg="hop_length", minimum=1),
    }
    if n_mels is not None:
        args["n_mels"] = require_int(n_mels, func=func, arg="n_mels", minimum=1)
    return args


def _unit(func: str, value: float) -> float:
    """Reject a fraction outside 0..1 at plan build, where it names the caller's method."""
    if not 0.0 <= value <= 1.0:
        raise PlanError(f"audio.{func}(): value must be in 0..1, got {value}")
    return float(value)


def _positive_secs(func: str, what: str, secs: float) -> float:
    """A duration must be a positive, finite number of seconds."""
    if not (secs > 0 and secs < float("inf")):
        raise PlanError(f"audio.{func}(): {what} must be a positive number of seconds, got {secs}")
    return float(secs)


class _AudioNamespace:
    """Lazy audio decode: ``col("bytes").audio.decode()`` / ``.audio.to_waveform()``.

    Decoding runs in the Rust data plane over a binary column (symphonia-backed), so an
    audio pipeline never materializes samples in Python. Null or undecodable input --
    including bytes that are not audio at all -- yields null rather than raising.

    The methods **compose**. Every waveform method hands back a ``List<Float32>``, and the
    ones that need no sample rate read one back, so
    ``.audio.trim_silence().audio.rms_normalize().audio.encode_wav(16000)`` is a single
    expression and each step after the first costs no second decode. The methods that are
    defined against a sample rate -- :meth:`resample`, :meth:`slice`, :meth:`pad_or_trim`
    and the spectral front ends -- still need encoded bytes, because a waveform column
    carries no rate; they say so by name if handed one.

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

        Raises:
            PlanError: If `rate` is not positive.
        """
        rate = require_int(rate, func="audio.resample", arg="rate", minimum=1)
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

        Raises:
            PlanError: If `rate`, `n_fft`, `hop_length`, or `n_mels` is not positive.
        """
        return AudioFunc(
            "mel_spectrogram",
            self._e,
            **_stft_args("audio.mel_spectrogram", rate, n_fft, hop_length, n_mels),
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

        Raises:
            PlanError: If `rate`, `n_fft`, `hop_length`, `n_mels`, or `n_mfcc` is not
                positive, or if `n_mfcc` exceeds `n_mels`.
        """
        args = _stft_args("audio.mfcc", rate, n_fft, hop_length, n_mels)
        n_mfcc = require_int(n_mfcc, func="audio.mfcc", arg="n_mfcc", minimum=1)
        if n_mfcc > args["n_mels"]:
            # The DCT keeps the first `n_mfcc` of `n_mels` coefficients, so asking for more
            # coefficients than there are bands has no answer. torchaudio raises here too.
            raise PlanError(
                f"audio.mfcc(): n_mfcc ({n_mfcc}) must not exceed n_mels ({args['n_mels']})"
            )
        return AudioFunc("mfcc", self._e, n_mfcc=n_mfcc, **args)

    # ---- level and hygiene measures ---------------------------------------
    def rms(self) -> AudioFunc:
        """Measure the root-mean-square amplitude of the clip.

        The level measure that tracks perceived loudness, unlike the peak: a recording with
        one door slam in it has a peak of 1.0 and an RMS that still says "quiet". Use
        :meth:`dbfs` for the same number in the unit audio tools state levels in.

        Returns:
            An expression evaluating to a Float64 in ``0..1``; null for null or undecodable
            input, and null for a clip that decoded to no samples.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> ds.select(level=bt.col("clip").audio.rms()).to_pydict()
                {'level': [0.5]}
        """
        return AudioFunc("rms", self._e)

    def dbfs(self) -> AudioFunc:
        """Measure the RMS level in decibels relative to full scale.

        The unit every audio tool states a level in, so a threshold ported from one is
        meaningful here. Half full scale is about ``-6``, and speech typically sits between
        ``-30`` and ``-12``.

        Returns:
            An expression evaluating to a negative Float64; null for null or undecodable
            input, and null for digital silence rather than negative infinity — an infinity
            passes every ``< threshold`` filter written to find quiet clips.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> ds.select(db=bt.col("clip").audio.dbfs().round(2)).to_pydict()
                {'db': [-6.02]}
        """
        return AudioFunc("dbfs", self._e)

    def peak_dbfs(self) -> AudioFunc:
        """Measure the loudest single sample, in decibels relative to full scale.

        Paired with :meth:`dbfs` this is the crest factor, which separates a compressed,
        broadcast-loud recording from a natural one.

        Returns:
            An expression evaluating to a negative Float64; null for null or undecodable
            input, and null for digital silence.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> ds.select(pk=bt.col("clip").audio.peak_dbfs().round(2)).to_pydict()
                {'pk': [-6.02]}
        """
        return AudioFunc("peak_dbfs", self._e)

    def clipping_ratio(self, threshold: float = 0.99) -> AudioFunc:
        """Measure the fraction of samples at or above `threshold` of full scale.

        The corpus-hygiene measure for audio. A clip recorded too hot is distorted in a way
        no level normalization can undo, and it is invisible to every other measure here
        because normalizing makes it *look* well-levelled.

        Args:
            threshold: The fraction of full scale a sample must reach to count as clipped,
                ``0..1``. The default catches samples the recorder clamped.

        Returns:
            An expression evaluating to a Float64 in ``0..1``; null for null or undecodable
            input, and null for a clip with no samples.

        Raises:
            PlanError: If `threshold` is outside ``0..1``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> ds.select(hot=bt.col("clip").audio.clipping_ratio()).to_pydict()
                {'hot': [0.0]}
        """
        return AudioFunc("clipping_ratio", self._e, factor=_unit("clipping_ratio", threshold))

    def silence_ratio(self, threshold_db: int = -40) -> AudioFunc:
        """Measure the fraction of samples quieter than `threshold_db`.

        Where :meth:`trim_silence` removes the ends, this measures the whole clip, so a
        recording that is mostly dead air is one predicate away.

        Args:
            threshold_db: The silence floor in dBFS (negative). ``-40`` is 1% of full scale,
                roughly where room tone sits and speech does not.

        Returns:
            An expression evaluating to a Float64 in ``0..1``; null for null or undecodable
            input, and null for a clip with no samples.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> ds.select(q=bt.col("clip").audio.silence_ratio()).to_pydict()
                {'q': [0.0]}
        """
        return AudioFunc(
            "silence_ratio",
            self._e,
            threshold_db=require_int(threshold_db, func="audio.silence_ratio", arg="threshold_db"),
        )

    # ---- waveform shaping -------------------------------------------------
    def rms_normalize(self, target_db: int = -20) -> AudioFunc:
        """Scale the waveform so its RMS level sits at `target_db`.

        The loudness-matching counterpart of :meth:`peak_normalize`, and usually the one you
        want: peak normalization equalizes the *maximum*, so a clip with one loud click
        stays quiet everywhere else. The gain is capped so the result cannot clip, which
        means a very quiet recording is lifted toward the target rather than driven into the
        rails. A digitally silent clip is returned unchanged.

        Args:
            target_db: The RMS level to reach, in dBFS (negative). ``-20`` is the broadcast
                convention for speech.

        Returns:
            An expression evaluating to a mono ``List<Float32>``; null for null or
            undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> level = bt.col("clip").audio.rms_normalize().list.len()
                >>> ds.select(n=level).to_pydict()
                {'n': [96]}
        """
        return AudioFunc(
            "rms_normalize",
            self._e,
            threshold_db=require_int(target_db, func="audio.rms_normalize", arg="target_db"),
        )

    def pre_emphasis(self, coefficient: float = 0.97) -> AudioFunc:
        """Apply the first-order high-pass ``y[n] = x[n] - a * x[n-1]``.

        The standard filter every classical ASR front end applies before framing, to flatten
        the spectral tilt of voiced speech. ``0`` is the identity.

        Args:
            coefficient: The filter coefficient ``a``, ``0..1``.

        Returns:
            An expression evaluating to a mono ``List<Float32>`` of the same length as the
            input; null for null or undecodable input.

        Raises:
            PlanError: If `coefficient` is outside ``0..1``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> flat = bt.col("clip").audio.pre_emphasis()
                >>> ds.select(n=flat.list.len()).to_pydict()
                {'n': [96]}
        """
        return AudioFunc("pre_emphasis", self._e, factor=_unit("pre_emphasis", coefficient))

    def pad_or_trim(self, duration_secs: float, rate: int) -> AudioFunc:
        """Force every clip to exactly `duration_secs` at `rate` Hz.

        The op that makes a clip corpus batchable. Whisper requires exactly 30 seconds of
        16 kHz audio and every other fixed-input audio model requires something like it, so
        without this a pipeline either loops in Python or hands the model rows of unequal
        length. A longer clip is truncated and a shorter one zero-padded at the end, which
        is what the reference implementations do.

        Because the length is a query parameter rather than a property of the data, the
        output column has a width the planner can know before a byte is read.

        Args:
            duration_secs: The exact length every row must have, in seconds.
            rate: The sample rate to resample to first, in Hz.

        Returns:
            An expression evaluating to a mono ``List<Float32>`` of exactly
            ``duration_secs * rate`` samples; null for null or undecodable input.

        Raises:
            PlanError: If `duration_secs` or `rate` is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> fixed = bt.col("clip").audio.pad_or_trim(0.5, 8000)
                >>> ds.select(n=fixed.list.len()).to_pydict()
                {'n': [4000]}
        """
        return AudioFunc(
            "pad_or_trim",
            self._e,
            rate=require_int(rate, func="audio.pad_or_trim", arg="rate", minimum=1),
            duration_secs=_positive_secs("pad_or_trim", "duration_secs", duration_secs),
        )

    def slice(self, offset_secs: float, duration_secs: float) -> AudioFunc:
        """Extract the region of the clip starting at `offset_secs`, `duration_secs` long.

        Measured against the clip's own sample rate, so the window names the time it says it
        does. Pair it with :meth:`decode`'s ``duration_secs`` to cut a long recording into
        equal segments without leaving the engine.

        Args:
            offset_secs: Where the window starts, in seconds from the beginning.
            duration_secs: How long the window is, in seconds.

        Returns:
            An expression evaluating to a mono ``List<Float32>``; null for null or
            undecodable input, and an **empty** list for a window past the end of the clip —
            an empty region is a fact about the window, not a failure to read the clip.

        Raises:
            PlanError: If `offset_secs` is negative or `duration_secs` is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> head = bt.col("clip").audio.slice(0.0, 0.005)
                >>> ds.select(n=head.list.len()).to_pydict()
                {'n': [40]}
        """
        if offset_secs < 0:
            raise PlanError(f"audio.slice(): offset_secs must be >= 0, got {offset_secs}")
        return AudioFunc(
            "slice",
            self._e,
            offset_secs=float(offset_secs),
            duration_secs=_positive_secs("slice", "duration_secs", duration_secs),
        )

    def encode_wav(self, rate: int | None = None) -> AudioFunc:
        """Encode the waveform as a mono 16-bit PCM WAV container.

        The op that closes the loop. Every other waveform method here hands back a
        ``List<Float32>``, which is what a model wants and what nothing else can read, so
        writing a trimmed, normalized corpus back to storage as *audio* meant encoding each
        row in Python. 16-bit PCM because it is the format every player, dataset loader and
        annotation tool accepts.

        It is the one method here that also accepts a **waveform** column, which is what
        closes the loop: every other waveform method hands back a ``List<Float32>``, so a
        trimmed and level-matched clip could otherwise leave the engine only as a list of
        floats. Applied to a waveform, `rate` is required and states what the samples
        already are — a waveform carries no sample rate, so there is nothing to resample
        *from*, and guessing would make the clip play at the wrong speed.

        Args:
            rate: For an encoded input, resample to this rate before encoding; omit to keep
                the clip's own rate. For a waveform input, the rate its samples are at.

        Returns:
            An expression evaluating to Binary WAV bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `rate` is given and not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> quiet = bt.col("clip").audio.encode_wav(4000)
                >>> ds.select(m=quiet.audio.decode().struct.field("sample_rate")).to_pydict()
                {'m': [4000]}

                >>> cleaned = bt.col("clip").audio.trim_silence().audio.encode_wav(8000)
                >>> ds.select(n=cleaned.audio.decode().struct.field("num_frames")).to_pydict()
                {'n': [96]}
        """
        if rate is None:
            return AudioFunc("encode_wav", self._e)
        return AudioFunc(
            "encode_wav",
            self._e,
            rate=require_int(rate, func="audio.encode_wav", arg="rate", minimum=1),
        )

    # ---- spectral ---------------------------------------------------------
    def spectrogram(self, rate: int, *, n_fft: int = 400, hop_length: int = 160) -> AudioFunc:
        """Compute the linear power spectrogram.

        The mel spectrogram's unwarped sibling. A mel filterbank is tuned to human pitch
        perception and to the speech models that consume it; a music, bioacoustic or
        machine-fault model wants the frequencies themselves, and the mel warp throws away
        exactly the high-frequency resolution those depend on.

        Args:
            rate: The sample rate to resample to before analysis, in Hz.
            n_fft: The STFT window size in samples. The default is the 25 ms window at
                16 kHz that the speech stack uses.
            hop_length: The stride between frames in samples (10 ms at 16 kHz).

        Returns:
            An expression evaluating to a ``List<Float32>`` of ``(n_fft/2+1) * n_frames`` in
            row-major ``(freq, frame)`` order, the same axis convention
            :meth:`mel_spectrogram` uses; null for null or undecodable input.

        Raises:
            PlanError: If any argument is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> spec = bt.col("clip").audio.spectrogram(8000, n_fft=64, hop_length=32)
                >>> ds.select(n=spec.list.len()).to_pydict()
                {'n': [132]}
        """
        return AudioFunc(
            "spectrogram", self._e, **_stft_args("audio.spectrogram", rate, n_fft, hop_length)
        )

    def spectral_centroid(self, rate: int, *, n_fft: int = 400, hop_length: int = 160) -> AudioFunc:
        """Measure the energy-weighted mean frequency, in Hz.

        The standard "brightness" descriptor, and the cheapest way to separate speech from
        music from noise without a model. Averaged over frames, skipping frames with no
        energy — a silent frame counted as "0 Hz" would drag the average toward DC and make
        a mostly-quiet recording look band-limited.

        Args:
            rate: The sample rate to resample to before analysis, in Hz.
            n_fft: The STFT window size in samples.
            hop_length: The stride between frames in samples.

        Returns:
            An expression evaluating to a Float64 in Hz; null for null or undecodable input,
            and null for a clip with no frames carrying energy.

        Raises:
            PlanError: If any argument is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> hz = bt.col("clip").audio.spectral_centroid(8000, n_fft=64, hop_length=32)
                >>> ds.select(bright=(hz > 3000)).to_pydict()
                {'bright': [True]}
        """
        return AudioFunc(
            "spectral_centroid",
            self._e,
            **_stft_args("audio.spectral_centroid", rate, n_fft, hop_length),
        )

    def spectral_rolloff(
        self, rate: int, *, percentile: float = 0.85, n_fft: int = 400, hop_length: int = 160
    ) -> AudioFunc:
        """Measure the frequency below which `percentile` of the spectral energy lies.

        The descriptor worth reaching for first on an unknown speech corpus. Where the
        centroid reports the middle of the spectrum, this reports its *edge*, which is how
        an 8 kHz telephone recording upsampled to 16 kHz is caught: it has a full-rate
        header, ordinary loudness, and no energy above 4 kHz. Nothing else here can see
        that, and a model trained on the mixture learns the artifact.

        Args:
            rate: The sample rate to resample to before analysis, in Hz.
            percentile: The fraction of energy below the reported frequency, ``0..1``.
            n_fft: The STFT window size in samples.
            hop_length: The stride between frames in samples.

        Returns:
            An expression evaluating to a Float64 in Hz; null for null or undecodable input,
            and null for a clip with no frames carrying energy.

        Raises:
            PlanError: If `percentile` is outside ``0..1`` or any size is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> edge = bt.col("clip").audio.spectral_rolloff(8000, n_fft=64, hop_length=32)
                >>> ds.select(full_band=(edge > 3000)).to_pydict()
                {'full_band': [True]}
        """
        return AudioFunc(
            "spectral_rolloff",
            self._e,
            factor=_unit("spectral_rolloff", percentile),
            **_stft_args("audio.spectral_rolloff", rate, n_fft, hop_length),
        )

    def spectral_bandwidth(
        self, rate: int, *, n_fft: int = 400, hop_length: int = 160
    ) -> AudioFunc:
        """Measure the energy-weighted spread of frequencies about the centroid, in Hz.

        Args:
            rate: The sample rate to resample to before analysis, in Hz.
            n_fft: The STFT window size in samples.
            hop_length: The stride between frames in samples.

        Returns:
            An expression evaluating to a Float64 in Hz; null for null or undecodable input,
            and null for a clip with no frames carrying energy.

        Raises:
            PlanError: If any argument is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> w = bt.col("clip").audio.spectral_bandwidth(8000, n_fft=64, hop_length=32)
                >>> ds.select(narrow=(w < 2000)).to_pydict()
                {'narrow': [True]}
        """
        return AudioFunc(
            "spectral_bandwidth",
            self._e,
            **_stft_args("audio.spectral_bandwidth", rate, n_fft, hop_length),
        )

    def spectral_flatness(self, rate: int, *, n_fft: int = 400, hop_length: int = 160) -> AudioFunc:
        """Measure spectral flatness — how noise-like rather than tonal the clip is.

        The ratio of the geometric to the arithmetic mean of the power spectrum: a pure tone
        is near 0, white noise near 1. It is what finds the dead channels and hiss-only
        recordings that every level measure reports as ordinary audio.

        Args:
            rate: The sample rate to resample to before analysis, in Hz.
            n_fft: The STFT window size in samples.
            hop_length: The stride between frames in samples.

        Returns:
            An expression evaluating to a Float64 in ``0..1``; null for null or undecodable
            input, and null for a clip with no frames carrying energy.

        Raises:
            PlanError: If any argument is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.audio import _WAV_ALTERNATING
                >>> ds = bt.from_pydict({"clip": [_WAV_ALTERNATING]})
                >>> f = bt.col("clip").audio.spectral_flatness(8000, n_fft=64, hop_length=32)
                >>> ds.select(tonal=(f < 0.1)).to_pydict()
                {'tonal': [True]}
        """
        return AudioFunc(
            "spectral_flatness",
            self._e,
            **_stft_args("audio.spectral_flatness", rate, n_fft, hop_length),
        )
