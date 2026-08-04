"""The `.video` expression namespace — lazy, batch-level video decode.

`VideoFunc` lowers to ``{"e": "video", "fn": ...}`` IR consumed by Rust
`Expr::Video` (FFmpeg-backed). Decode requires building the engine with the
``video`` cargo feature (system FFmpeg); without it, evaluating a `.video` op
raises a clear error. Like image/audio, the interpreter is the oracle and the JIT
falls back, so the tiers cannot diverge.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.fn_names import VIDEO_FNS
from batcher.plan.expr_ir.node_base import IRNode, child, expr_node, scalar
from batcher.plan.ir_tags import ExprTag

__all__ = ["VideoFunc", "_VideoNamespace"]


@expr_node
class VideoFunc(IRNode):
    """A video decode op over a binary (video-bytes) sub-expression (via `.video`).

    `decode` reads each clip's metadata; `frames`, `thumbnail`, and `frame_at` turn a
    clip into pixels. Requires the engine's ``video`` feature.
    """

    tag = ExprTag.VIDEO
    vocab = VIDEO_FNS
    fn: str = scalar()
    input: Expr = child()
    # The sampling ops only. Omitted from the IR unless set, so `decode`'s wire shape is
    # byte-identical to what it was before the sampling ops existed.
    num_frames: int | None = scalar(omit_none=True, default=None)
    width: int | None = scalar(omit_none=True, default=None)
    height: int | None = scalar(omit_none=True, default=None)
    second: float | None = scalar(omit_none=True, default=None)


class _VideoNamespace:
    """Lazy video decode: ``col("bytes").video.decode()`` / ``.video.frames(8, 224, 224)``.

    Decoding runs in the Rust data plane over a binary column (FFmpeg-backed), so a video
    pipeline never materializes frames in Python. Evaluating a `.video` op needs the
    engine built with the ``video`` cargo feature; building the expression does not.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.col("clip").video.decode().to_ir()["fn"]
            'decode'
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.video` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.video accessor of col('c')>``."""
        return f"<.video accessor of {self._e!r}>"

    def decode(self) -> VideoFunc:
        """Read each clip's metadata without materializing its frames.

        Requires the engine built with the ``video`` cargo feature (system FFmpeg);
        without it, evaluating this op raises a clear error rather than returning null.

        Returns:
            An expression evaluating to a struct ``{width, height, num_frames,
            duration_secs, fps}``; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.read.video("s3://bucket/clips/")  # doctest: +SKIP
                >>> ds.select(m=bt.col("bytes").video.decode()).to_pydict()  # doctest: +SKIP
                {'m': [{'width': 1920, 'height': 1080, ...}]}
        """
        return VideoFunc("decode", self._e)

    def frames(self, num_frames: int, width: int, height: int) -> VideoFunc:
        """Sample `num_frames` evenly-spaced frames, each resized to ``(width, height)``.

        The video training-ingest path, and the counterpart of
        :meth:`~batcher.plan.expr_ir.image._ImageNamespace.to_tensor` for clips. Because
        both the frame count and the frame size are fixed, every row has the same
        ``(num_frames, height, width, 3)`` shape whatever the source clip's resolution or
        length was — so the result is a fixed-shape tensor column a video model consumes
        directly, with no per-row Python anywhere in the pipeline.

        Frames are the ones ``numpy.linspace(0, num_frames - 1, n)`` names, matching the
        reference preprocessing of the common video models, and they are found by decoding
        the clip in order rather than by seeking, so the *n*-th frame really is the *n*-th.
        The clip is never decoded past the last wanted frame, and only the wanted frames
        are kept, so peak memory is the output plus one frame rather than the whole clip.

        A clip with fewer than `num_frames` frames repeats frames rather than yielding a
        shorter row, since a ragged row would not be a fixed-shape tensor.

        Args:
            num_frames: How many frames to sample from each clip.
            width: Width each frame is resized to, in pixels.
            height: Height each frame is resized to, in pixels.

        Returns:
            An expression evaluating to a ``FixedSizeList<u8>`` of
            ``num_frames * height * width * 3`` RGB8 samples; null for null or
            undecodable input, and for a clip whose frames cannot all be decoded.

        Raises:
            PlanError: If `num_frames`, `width`, or `height` is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> clips = bt.read.video("s3://bucket/clips/")  # doctest: +SKIP
                >>> tensors = clips.select(  # doctest: +SKIP
                ...     x=bt.col("bytes").video.frames(8, 224, 224)
                ... )
        """
        if num_frames <= 0:
            raise PlanError(f"video.frames(): num_frames must be positive, got {num_frames}")
        if width <= 0 or height <= 0:
            raise PlanError(
                f"video.frames(): width and height must be positive, got {width}x{height}"
            )
        return VideoFunc("frames", self._e, num_frames=num_frames, width=width, height=height)

    def thumbnail(self, max_size: int) -> VideoFunc:
        """Take one representative frame from each clip, as PNG bytes.

        The frame is the one halfway through the clip, not the first: the first frame of
        a real clip is very often black, a title card, or a fade-in, which makes a corpus
        of first-frame thumbnails useless for exactly the review and curation work
        thumbnails exist for.

        Scaled so its longest side is `max_size`, keeping the clip's aspect ratio and never
        upscaling — the same operation, and the same rule, as
        :meth:`~batcher.plan.expr_ir.image._ImageNamespace.thumbnail`. Across the whole
        media surface, an op that hands back an **encoded still** takes a longest side and
        keeps the shape, while an op that hands back a **tensor** takes exact dimensions,
        because a tensor feeds a model that needs every row the same and a still is looked
        at, where a squashed 16:9 frame is a distortion nothing downstream can see.

        The clip is seeked to the keyframe before the midpoint and decoded forward from
        there, so the cost is bounded by the keyframe interval rather than by the clip's
        length — which is what makes thumbnailing a large corpus affordable.

        Args:
            max_size: Length of the longest side of the thumbnail, in pixels.

        Returns:
            An expression evaluating to PNG bytes; null for null or undecodable input.

        Raises:
            PlanError: If `max_size` is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> clips = bt.read.video("s3://bucket/clips/")  # doctest: +SKIP
                >>> contact_sheet = clips.select(  # doctest: +SKIP
                ...     path=bt.col("path"), thumb=bt.col("bytes").video.thumbnail(320)
                ... )
        """
        if max_size <= 0:
            raise PlanError(f"video.thumbnail(): max_size must be positive, got {max_size}")
        return VideoFunc("thumbnail", self._e, width=max_size)

    def frame_at(self, second: float, max_size: int) -> VideoFunc:
        """Take the frame shown at `second`, as PNG bytes.

        The random-access counterpart of :meth:`frames`, for when a row already carries a
        timestamp — a detection, a caption, a scene boundary — and the pipeline needs the
        picture at it. The clip is seeked to the keyframe before `second` and decoded
        forward from there, so reading a frame ten minutes in costs about what reading one
        ten seconds in does.

        The frame returned is the one a player displays at `second`: the last frame whose
        presentation time is at or before it. Address a frame by *index* rather than by
        time with :meth:`frames`, which decodes in order.

        Scaled so its longest side is `max_size`, keeping the clip's aspect ratio and
        never upscaling, as :meth:`thumbnail` does.

        Args:
            second: Offset from the start of the clip, in seconds.
            max_size: Length of the longest side of the still, in pixels.

        Returns:
            An expression evaluating to PNG bytes; null for null or undecodable input, and
            for a `second` past the end of a clip whose duration is known.

        Raises:
            PlanError: If `second` is negative or `max_size` is not positive.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> hits = bt.read.video("s3://bucket/clips/")  # doctest: +SKIP
                >>> stills = hits.select(  # doctest: +SKIP
                ...     still=bt.col("bytes").video.frame_at(3.5, 640)
                ... )
        """
        if second < 0:
            raise PlanError(f"video.frame_at(): second must be non-negative, got {second}")
        if max_size <= 0:
            raise PlanError(f"video.frame_at(): max_size must be positive, got {max_size}")
        return VideoFunc("frame_at", self._e, second=float(second), width=max_size)
