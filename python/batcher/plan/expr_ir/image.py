"""The `.image` expression namespace — lazy, batch-level image decode.

`ImageFunc` lowers to ``{"e": "image", "fn": ...}`` IR consumed by Rust
`Expr::Image`. Decoding is library-backed, so the interpreter is the oracle and
the JIT falls back; there is one implementation, so the tiers cannot diverge.
"""

from __future__ import annotations

import base64

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.fn_names import IMAGE_FNS
from batcher.plan.expr_ir.node_base import IRNode, child, expr_node, scalar
from batcher.plan.ir_tags import ExprTag

__all__ = ["ImageCrop", "ImageFunc", "_ImageNamespace"]

# Container formats `.image.encode` can write; mirrors `bc-expr`'s `ENCODE_FORMATS`.
# WebP is readable but not writable by the underlying decoder, so it is deliberately
# absent: accepting the name at plan build and failing at run time would be worse than
# rejecting it here.
_IMAGE_FORMATS = frozenset({"png", "jpeg", "bmp", "gif"})

# Color modes `.image.convert` can produce; mirrors `bc-expr`'s `COLOR_MODES`, and the
# same vocabulary `.image.decode()` reports — so a mode read off `decode` can be handed
# straight back to `convert`.
_IMAGE_MODES = frozenset({"L", "LA", "RGB", "RGBA"})


def _non_negative(func: str, value: float) -> float:
    """Reject a negative enhancement factor at plan build, where it names the method."""
    if value < 0:
        raise PlanError(f"image.{func}(): factor must be >= 0, got {value}")
    return float(value)


def _container(func: str, format: str, quality: int | None) -> dict[str, object]:
    """Validate an output container and quality, as the IR keywords they become.

    Every bytes-out op takes the same pair, so the check lives once. Rejecting a bad
    format here rather than in the engine is what turns a typo into a plan-build error
    naming the caller's own method, instead of a per-row failure a million rows into a
    scan.
    """
    if format not in _IMAGE_FORMATS:
        raise PlanError(
            f"image.{func}(): format must be one of {sorted(_IMAGE_FORMATS)}, got {format!r}"
        )
    if quality is not None and not 1 <= quality <= 100:
        raise PlanError(f"image.{func}(): quality must be in 1..100, got {quality}")
    args: dict[str, object] = {}
    # Only ever set when it differs from the engine's default, so an unchanged plan keeps
    # its exact wire shape: `format` is the stable contract, not a place to write "png".
    if format != "png":
        args["format"] = format
    if quality is not None:
        args["quality"] = quality
    return args


@expr_node
class ImageCrop(IRNode):
    """A crop whose window is four sub-expressions rather than four constants.

    Its own node rather than four more scalars on `ImageFunc`, because the distinction is
    real: every other image op's dimensions are part of its output *type* and so cannot
    vary per row, while a crop window is data. Cropping the box a detector predicted is
    what a vision pipeline is built around.
    """

    tag = ExprTag.IMAGE_CROP
    input: Expr = child()
    x: Expr = child()
    y: Expr = child()
    width: Expr = child()
    height: Expr = child()


@expr_node
class ImageFunc(IRNode):
    """An image decode op over a binary (image-bytes) sub-expression (via `.image`).

    `decode` reads each image's header facts (dimensions, channel count, color mode);
    `to_tensor` decodes, resizes to
    ``(width, height)``, and flattens to a fixed-size RGB8 pixel list; `to_tensor_f32`
    additionally scales/normalizes to a model-ready ``float32`` tensor; `center_crop`
    crops the centered region; `to_grayscale` converts to a single luma channel.
    """

    tag = ExprTag.IMAGE
    vocab = IMAGE_FNS
    fn: str = scalar()
    input: Expr = child()
    width: int | None = scalar(omit_none=True, default=None)
    height: int | None = scalar(omit_none=True, default=None)
    # `to_tensor_f32` only: per-channel normalization and channel-first layout. Omitted
    # from the IR unless set, so the other image ops' wire shape is unchanged.
    mean: list[float] | None = scalar(omit_none=True, default=None)
    std: list[float] | None = scalar(omit_none=True, default=None)
    channels_first: bool = scalar(omit_falsy=True, default=False)
    # The container every bytes-out op re-encodes into (`png` when absent). It used to be
    # `encode`'s alone, with `convert` borrowing it for a color mode -- which left `resize`,
    # `thumbnail` and `auto_orient` hard-wired to PNG, several times larger and slower than
    # the JPEG a photographic corpus arrived as. `convert` now has its own `mode`.
    format: str | None = scalar(omit_none=True, default=None)
    # `convert` only: the target color mode.
    mode: str | None = scalar(omit_none=True, default=None)
    # Encoder quality for the lossy containers, 1..100.
    quality: int | None = scalar(omit_none=True, default=None)
    # The one scalar knob the photometric and geometry ops take, named per op: `rotate`'s
    # degrees, `adjust_*`'s factor, `blur`/`sharpen`'s sigma, `posterize`'s bit count,
    # `solarize`'s threshold, `autocontrast`'s cutoff. One slot rather than six, because an
    # op reads exactly one and six would be five nulls in every image plan.
    factor: float | None = scalar(omit_none=True, default=None)
    # `letterbox`/`pad` only: the byte the leftover canvas is filled with.
    fill: int | None = scalar(omit_none=True, default=None)


# A 1x1 red PNG. Exported so the doctests here (and the docs) have a real image to
# decode without reaching for a fixture file.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

# A 2x2 RGB PNG: red, green / blue, white. A 1x1 image has no geometry to flip, no colour
# spread to measure and no histogram to equalize, so the ops added for curation and
# augmentation need a picture with at least two of everything to show anything at all.
_PNG_2X2 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGP4z8DAAMIM/4EAAB/uBfsL2WiLAAAAAElFTkSuQmCC"
)


class _ImageNamespace:
    """Lazy image decode: ``col("bytes").image.decode()`` / ``.image.to_tensor(224, 224)``.

    Decoding runs in the Rust data plane over a binary column, so an image pipeline
    never materializes pixels in Python. Null or undecodable input yields null.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.plan.expr_ir.image import _PNG_1X1
            >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
            >>> ds.select(dims=bt.col("img").image.decode()).to_pydict()
            {'dims': [{'width': 1, 'height': 1, 'channels': 4, 'mode': 'RGBA'}]}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.image` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.image accessor of col('c')>``."""
        return f"<.image accessor of {self._e!r}>"

    def decode(self) -> ImageFunc:
        """Read each image's header facts without materializing its pixels.

        Only the image header is parsed, so this succeeds — and is cheap — even for a
        file whose pixel data is truncated or corrupt. Use `to_tensor` when the pixels
        themselves must be valid.

        All four facts come from one header read, so asking for the mode costs nothing
        beyond asking for the width. Project the one you want with
        ``.struct.field("width")``. ``mode`` uses Pillow's vocabulary — ``L``, ``LA``,
        ``RGB``, ``RGBA`` — because that is what the other half of a multimodal pipeline
        speaks; bit depth is not part of the name, so a 16-bit RGB image is ``RGB`` with
        3 channels.

        Returns:
            An expression evaluating to a struct ``{width, height, channels, mode}``,
            the first three Int32 and ``mode`` Utf8; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> ds.select(d=bt.col("img").image.decode()).to_pydict()
                {'d': [{'width': 1, 'height': 1, 'channels': 4, 'mode': 'RGBA'}]}

                >>> ds.select(w=bt.col("img").image.decode().struct.field("width")).to_pydict()
                {'w': [1]}
        """
        return ImageFunc("decode", self._e)

    def crop(
        self,
        x: int | Expr,
        y: int | Expr,
        width: int | Expr,
        height: int | Expr,
    ) -> ImageCrop:
        """Cut the window at ``(x, y)`` out of each image, as PNG bytes.

        The arbitrary-offset counterpart of :meth:`center_crop`, and the shape a detection
        pipeline needs: pull a bounding box out of a frame and keep it as an *image* rather
        than as a tensor.

        **Each bound may be a column.** That is what makes this the operation a vision
        pipeline is built around rather than a fixed-window utility: the boxes a detector
        predicts are data, one per row, so cutting them out of their frames needs the
        window to vary per row. Mixing constants and columns is fine — a fixed-size patch
        at a per-row position is ``crop(col("cx"), col("cy"), 64, 64)``.

        A window that runs past an edge is **clipped**, so the result can be smaller than
        requested. That is the opposite of :meth:`center_crop`, which zero-pads, and the
        difference is deliberate: `center_crop` feeds a model that needs a fixed input
        size, while a cropped image is something a person or another tool will look at,
        and inventing black pixels there would be inventing data. A window that starts
        past the image entirely is null, as is one whose bounds are null, negative, or
        non-positive in extent — a box the caller could not supply is a row with no
        answer, not a reason to fail the batch.

        Because the window varies per row, the result is encoded bytes rather than a
        fixed-shape tensor: rows genuinely differ in size. Feed a model by following it
        with :meth:`to_tensor` or :meth:`letterbox`.

        Args:
            x: Left edge of the window, in pixels from the left of the image.
            y: Top edge of the window, in pixels from the top of the image.
            width: Window width in pixels.
            height: Window height in pixels.

        Returns:
            An expression evaluating to PNG bytes of the cropped region; null for null,
            undecodable, or entirely-out-of-bounds input, and for an unusable window.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> region = bt.col("img").image.crop(0, 0, 1, 1)
                >>> ds.select(d=region.image.decode().struct.field("width")).to_pydict()
                {'d': [1]}

                >>> # A detector's boxes, cut out of the frames they were found in.
                >>> patch = bt.col("frame").image.crop(
                ...     bt.col("box_x"), bt.col("box_y"), bt.col("box_w"), bt.col("box_h")
                ... )
        """
        from batcher.plan.expr_ir.constructors import lit

        bounds = [b if isinstance(b, Expr) else lit(int(b)) for b in (x, y, width, height)]
        return ImageCrop(self._e, *bounds)

    def encode(self, format: str, *, quality: int | None = None) -> ImageFunc:
        """Re-encode each image in `format`, pixels unchanged.

        Normalizes a mixed-format corpus onto one codec, or trades a PNG for a smaller
        JPEG. Because JPEG has no alpha channel, an RGBA source is flattened to RGB rather
        than failing the row, which would otherwise drop every transparent image.

        Args:
            format: One of ``"png"``, ``"jpeg"``, ``"bmp"``, or ``"gif"``. WebP is
                readable but not writable, so it is not offered.
            quality: Encoder quality in ``1..100`` for the lossy containers. Ignored by
                the lossless ones. Defaults to the encoder's own (75 for JPEG).

        Returns:
            An expression evaluating to the re-encoded bytes; null for null or
            undecodable input.

        Raises:
            PlanError: If `format` is not a writable format, or `quality` is out of range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> jpeg = bt.col("img").image.encode("jpeg")
                >>> ds.select(m=jpeg.image.decode().struct.field("mode")).to_pydict()
                {'m': ['RGB']}
        """
        # Unlike its neighbours, `encode` names the container as its whole purpose, so it
        # is written even when it is the default -- the plan should say what was asked for.
        return ImageFunc(
            "encode", self._e, **{**_container("encode", format, quality), "format": format}
        )

    def convert(self, mode: str, *, format: str = "png", quality: int | None = None) -> ImageFunc:
        """Convert each image to color `mode`, re-encoded.

        The general form of :meth:`to_grayscale`, which is ``"L"`` plus a resize. This
        changes only the channels, which is what normalizing a corpus that mixes RGB and
        RGBA needs before a model that wants one of them.

        The mode names are the ones :meth:`decode` reports, so a mode read off one can be
        handed straight back to the other. Grayscale uses Rec. 601 luma — the same
        weighting :meth:`to_grayscale` and :meth:`dhash` use, so the three cannot disagree
        about what grey means.

        Args:
            mode: One of ``"L"`` (grayscale), ``"LA"`` (grayscale + alpha), ``"RGB"``, or
                ``"RGBA"``.
            format: Container to write. Note that ``"jpeg"`` carries no alpha, so it
                flattens ``"LA"``/``"RGBA"`` back to three channels.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to image bytes in `mode`; null for null or
            undecodable input.

        Raises:
            PlanError: If `mode` is not one of the four, or the container is not writable.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> grey = bt.col("img").image.convert("L")
                >>> ds.select(m=grey.image.decode().struct.field("mode")).to_pydict()
                {'m': ['L']}
        """
        if mode not in _IMAGE_MODES:
            raise PlanError(
                f"image.convert(): mode must be one of {sorted(_IMAGE_MODES)}, got {mode!r}"
            )
        return ImageFunc("convert", self._e, mode=mode, **_container("convert", format, quality))

    def to_tensor(self, width: int, height: int) -> ImageFunc:
        """Decode and resize to ``(width, height)``, flattened to RGB8 pixels.

        The training-ingest path: it produces a fixed-size numeric column that feeds
        a model directly, with no per-row Python.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            An expression evaluating to a ``FixedSizeList<u8>`` of ``height * width * 3``
            RGB8 samples; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> len(ds.select(t=bt.col("img").image.to_tensor(2, 2)).to_pydict()["t"][0])
                12
        """
        return ImageFunc("to_tensor", self._e, width=width, height=height)

    def to_tensor_f32(
        self,
        width: int,
        height: int,
        *,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        channels_first: bool = False,
    ) -> ImageFunc:
        """Decode, resize, and normalize to a model-ready ``float32`` tensor.

        The full torchvision ``ToTensor`` + ``Normalize`` preprocessing in one native
        pass: decode, resize to ``(width, height)``, scale to ``[0, 1]`` (``pixel/255``),
        then optionally apply per-channel ``(x - mean) / std``, laid out ``HWC`` (default)
        or ``CHW`` (``channels_first=True``). Because it runs in the engine, an image
        pipeline never exits to a per-batch Python UDF for the ``/255`` / standardize /
        permute — the output is a fixed-shape-tensor column ready for the model.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.
            mean: Optional per-channel means (3 values, RGB) subtracted after scaling;
                defaults to no shift.
            std: Optional per-channel standard deviations (3 values, RGB) divided after
                the mean shift; defaults to no scaling. Values must be non-zero.
            channels_first: Emit ``CHW`` (channel, height, width) instead of the default
                ``HWC`` — the layout most torch models expect.

        Returns:
            An expression evaluating to a ``FixedSizeList<float32>`` of
            ``height * width * 3`` samples (a fixed-shape-tensor column); null for null
            or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> t = bt.col("img").image.to_tensor_f32(
                ...     2, 2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                ...     channels_first=True,
                ... )
                >>> len(ds.select(t=t).to_pydict()["t"][0])
                12
        """
        return ImageFunc(
            "to_tensor_f32",
            self._e,
            width=width,
            height=height,
            mean=mean,
            std=std,
            channels_first=channels_first,
        )

    def to_grayscale(self, width: int, height: int) -> ImageFunc:
        """Decode, resize to ``(width, height)``, and convert to a single luma channel.

        The color-convert step for models that take 1-channel input (many document,
        medical, and depth models). Uses the standard Rec.601 luminance
        (``0.299 R + 0.587 G + 0.114 B``), matching PIL ``convert("L")``, in the data plane.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            An expression evaluating to a ``FixedSizeList<u8>`` of ``height * width``
            luminance samples, shape ``(height, width, 1)``; null for null or undecodable
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> len(ds.select(t=bt.col("img").image.to_grayscale(2, 2)).to_pydict()["t"][0])
                4
        """
        return ImageFunc("to_grayscale", self._e, width=width, height=height)

    def center_crop(self, width: int, height: int) -> ImageFunc:
        """Decode and center-crop to ``(width, height)`` RGB8 pixels.

        The second half of the standard vision inference transform (resize the short side,
        then center-crop to the model's input size). Runs natively in the data plane; when
        the image is smaller than the crop the border is zero-padded, matching torchvision
        ``CenterCrop``.

        Args:
            width: Crop width in pixels.
            height: Crop height in pixels.

        Returns:
            An expression evaluating to a ``FixedSizeList<u8>`` of ``height * width * 3``
            RGB8 samples (a fixed-shape-tensor column); null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> len(ds.select(t=bt.col("img").image.center_crop(2, 2)).to_pydict()["t"][0])
                12
        """
        return ImageFunc("center_crop", self._e, width=width, height=height)

    def dhash(self) -> ImageFunc:
        """Compute each image's 64-bit perceptual *difference hash*, for near-duplicate work.

        The image is reduced to a 9x8 grayscale thumbnail and each row's adjacent pixel
        pairs are compared, giving 64 bits of "is this pixel brighter than its right-hand
        neighbour". Because it encodes *gradients* rather than pixel values, the hash
        survives re-encoding, rescaling and brightness shifts — so a thumbnail and its
        original agree, while different pictures do not.

        The result is a plain integer, so the existing bitwise vocabulary does the rest:
        ``a.bitwise_xor(b).bit_count()`` is the Hamming distance between two hashes, and a
        threshold on it (``<= 5`` is a common choice) is a near-duplicate predicate. Group
        by the hash itself for exact-duplicate collapse.

        Returns:
            An expression evaluating to an Int64 hash; null for null or undecodable
            input. It is Int64 rather than UInt64 because the FFI boundary normalizes to
            `i64` and rejects a `u64` above `i64::MAX` — a UInt64 hash would be unusable
            for half of all images. The 64 bits are reinterpreted, not clamped, and XOR
            and bit-count are unaffected by how the sign bit is read.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1, _PNG_1X1]})
                >>> # The same image hashes the same, so exact dedup is a group-by.
                >>> ds.select(h=bt.col("img").image.dhash()).to_pydict()["h"][0] == (
                ...     ds.select(h=bt.col("img").image.dhash()).to_pydict()["h"][1]
                ... )
                True

                >>> # Hamming distance between two images' hashes.
                >>> d = bt.col("a").image.dhash().bitwise_xor(bt.col("b").image.dhash())
                >>> near_duplicate = d.bit_count() <= 5
        """
        return ImageFunc("dhash", self._e)

    def brightness(self) -> ImageFunc:
        """The image's mean luma, normalized to ``[0, 1]`` (→ Float64).

        The blank-image detector for a scraped corpus. A placeholder tile, a blown-out scan,
        and the grey box a CDN serves for a missing asset all decode perfectly and teach a
        model nothing; each sits at an extreme of this scale while a photograph of anything
        lands in the middle. Filtering both ends removes a class of row nothing upstream
        catches.

        Measured on a downsampled luma plane, so the cost per image does not depend on its
        resolution — a corpus mixing thumbnails and 50-megapixel scans is not dominated by the
        scans.

        Returns:
            An expression evaluating to a Float64 in ``[0, 1]``; null for null or undecodable
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> 0.0 <= ds.select(b=bt.col("img").image.brightness()).to_pydict()["b"][0] <= 1.0
                True

                >>> # Drop the blank and blown-out ends of a corpus.
                >>> usable = (bt.col("img").image.brightness() > 0.05) & (
                ...     bt.col("img").image.brightness() < 0.95
                ... )
        """
        return ImageFunc("brightness", self._e)

    def sharpness(self) -> ImageFunc:
        """The variance of the image's Laplacian, normalized to ``[0, 1]`` (→ Float64).

        The standard focus measure, and the way to find the blurred tail of an image corpus. A
        sharp image has strong second derivatives at its edges and so a high variance; a
        blurred or empty one has almost none.

        Values are small in absolute terms — a well-focused photograph lands around 0.01 to
        0.05 — so choose the threshold from a histogram of your own images rather than from a
        remembered number. It measures *detail*, not quality: a brick wall outscores a
        portrait, and a noisy image outscores a clean one. Use it to find the blurred tail, not
        to rank images against each other.

        The image is downsampled before measuring, deliberately: at full resolution sensor
        noise reads as high-frequency detail and a blurry large photograph scores like a sharp
        one, which is the failure mode of a naive Laplacian variance.

        Returns:
            An expression evaluating to a Float64 in ``[0, 1]``; null for null or undecodable
            input, and for an image too small to have an interior pixel.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> # A 1x1 image has no interior pixel, so it has no second derivative.
                >>> ds.select(s=bt.col("img").image.sharpness()).to_pydict()["s"]
                [None]
        """
        return ImageFunc("sharpness", self._e)

    def thumbnail(
        self, max_size: int, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Scale so the longest side is `max_size`, keeping the aspect ratio.

        The aspect-preserving counterpart of :meth:`resize`, and the one to reach for when
        the output is for a person rather than a model. `resize` takes both dimensions, so
        it stretches anything not already at the target ratio: a corpus of mixed portrait
        and landscape photographs run through ``resize(256, 256)`` comes out squashed, and
        nothing about the shape of the result says so.

        Never *up*scales. Enlarging a small image to reach `max_size` invents detail and
        costs bytes, which is also what Pillow's ``Image.thumbnail`` does, so a corpus
        already normalized against that stays comparable.

        Args:
            max_size: Length of the longest side of the result, in pixels.
            format: Container to write. One of ``"png"``, ``"jpeg"``, ``"bmp"``, ``"gif"``.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `format` is not writable or `quality` is out of range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> small = bt.col("img").image.thumbnail(256)
                >>> ds.select(d=small.image.decode()).to_pydict()["d"][0]["width"]
                1
        """
        return ImageFunc(
            "thumbnail", self._e, width=max_size, **_container("thumbnail", format, quality)
        )

    def letterbox(self, width: int, height: int, *, fill: int = 114) -> ImageFunc:
        """Fit onto a ``(width, height)`` canvas keeping the aspect ratio, padding the rest.

        The standard object-detection preprocessing, and the reason neither
        :meth:`to_tensor` nor :meth:`center_crop` covers it. `to_tensor` stretches, which
        moves every box a model predicts off its object; `center_crop` throws the border
        away, which is where the missed detections live. Letterboxing does neither: the
        whole image survives at its true aspect ratio, and the leftover canvas is a
        constant the model learns to ignore.

        The image is centered on the canvas, so the padding is split evenly between the two
        sides. An off-centre paste would bias every coordinate a model predicts, in a way
        that is easy to miss and hard to trace back.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            fill: Byte value the leftover canvas is filled with, ``0``-``255``. The default
                ``114`` is the YOLO family's grey, so a model trained against that
                preprocessing sees the padding it expects.

        Returns:
            An expression evaluating to a ``FixedSizeList<u8>`` of ``height * width * 3``
            RGB8 samples (a fixed-shape-tensor column); null for null or undecodable input.

        Raises:
            PlanError: If `fill` is outside ``0``-``255``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> boxed = bt.col("img").image.letterbox(4, 4)
                >>> len(ds.select(t=boxed).to_pydict()["t"][0])
                48
        """
        if not 0 <= fill <= 255:
            raise PlanError(f"image.letterbox(): fill must be in 0..=255, got {fill}")
        return ImageFunc("letterbox", self._e, width=width, height=height, fill=fill)

    def auto_orient(self, *, format: str = "png", quality: int | None = None) -> ImageFunc:
        """Apply each image's EXIF orientation, re-encoded.

        A camera almost never rotates its sensor data. It records which way up it was held
        in the EXIF ``Orientation`` tag and leaves the pixels as the sensor read them, so a
        portrait phone photo is *stored* landscape with a "rotate 90" note attached. Every
        viewer, phone gallery, and browser honours that note, as does ``cv2.imread`` and
        anything built on ``PIL.ImageOps.exif_transpose``.

        The decoder behind this namespace does not. So a corpus of phone photographs
        decodes a quarter turn from what the rest of a pipeline sees — the right shape,
        real pixels, the wrong image — and nothing downstream can tell. Insert this before
        the decode ops and the two agree:

        ``col("bytes").image.auto_orient().image.to_tensor(224, 224)``

        It is a separate operation rather than a changed default because flipping the
        default would rotate the output of pipelines that already compensate. Use
        :meth:`exif_orientation` to find out whether a corpus needs it at all.

        The result is PNG, which carries no EXIF, so the rotation cannot be applied twice.
        An image that is already upright, or in a format that cannot carry orientation, is
        re-encoded unchanged.

        Args:
            format: Container to write. One of ``"png"``, ``"jpeg"``, ``"bmp"``, ``"gif"``.
                The result carries no EXIF whichever is chosen, so the orientation cannot
                be applied twice.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to PNG bytes; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> upright = bt.col("img").image.auto_orient()
                >>> ds.select(d=upright.image.decode().struct.field("width")).to_pydict()
                {'d': [1]}
        """
        return ImageFunc("auto_orient", self._e, **_container("auto_orient", format, quality))

    def exif_orientation(self) -> ImageFunc:
        """Read each image's EXIF orientation code, 1 through 8 (→ Int32).

        The diagnostic half of :meth:`auto_orient`. Whether a corpus needs orienting is
        otherwise invisible — a rotated decode is a valid image of the right size — so this
        is how you find out, and how you measure how much of a corpus is affected:

        ``ds.filter(bt.col("bytes").image.exif_orientation() != 1).count()``

        The codes are the EXIF standard's: 1 upright, 2 mirrored, 3 rotated 180, 4 flipped
        vertically, 5-8 the quarter turns and their mirrors.

        Returns:
            An expression evaluating to an Int32 in ``1..8``; null for null or undecodable
            input. An image carrying no orientation reports ``1``, as does one in a format
            that cannot carry the tag — the code means "already upright", not "absent".

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> ds.select(o=bt.col("img").image.exif_orientation()).to_pydict()
                {'o': [1]}
        """
        return ImageFunc("exif_orientation", self._e)

    def resize(
        self, width: int, height: int, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Resize the image and re-encode it, stretching to the exact size asked for.

        Use :meth:`thumbnail` when the aspect ratio must survive: this takes both
        dimensions and so squashes anything not already at the target ratio, which no
        shape assertion downstream can see.

        `format` matters more than it looks. A photographic corpus arrives as JPEG, and
        re-encoding it as PNG is both slower to write and several times larger — so a
        resize step that was meant to shrink a dataset used to inflate it instead. Pass
        the container the corpus should stay in.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.
            format: Container to write. One of ``"png"``, ``"jpeg"``, ``"bmp"``,
                ``"gif"``.
            quality: Encoder quality in ``1..100`` for the lossy containers. Ignored by
                the lossless ones. Defaults to the encoder's own (75 for JPEG).

        Returns:
            An expression evaluating to Binary image bytes; null for null or
            undecodable input.

        Raises:
            PlanError: If `format` is not writable or `quality` is out of range.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> small = bt.col("img").image.resize(2, 2)
                >>> ds.select(d=small.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}

                >>> jpeg = bt.col("img").image.resize(8, 8, format="jpeg", quality=60)
                >>> ds.select(f=jpeg.image.format()).to_pydict()
                {'f': ['jpeg']}
        """
        return ImageFunc(
            "resize",
            self._e,
            width=width,
            height=height,
            **_container("resize", format, quality),
        )

    # ---- geometry ---------------------------------------------------------
    def rotate(self, degrees: int, *, format: str = "png", quality: int | None = None) -> ImageFunc:
        """Turn the image by a multiple of 90 degrees, re-encoded.

        Only right angles. A free rotation resamples every pixel and leaves a triangular
        border in a color nobody chose, where a quarter turn is a transposition that is
        exact and lossless — and "rotate this corpus upright" is what people actually
        want. Use :meth:`auto_orient` when the turn should come from the camera's own
        Exif tag rather than from a constant.

        Args:
            degrees: A multiple of 90. Negative and over-full-turn values are normalized,
                so ``-90`` and ``270`` are the same rotation.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `degrees` is not a multiple of 90, or the container is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> turned = bt.col("img").image.rotate(90)
                >>> ds.select(d=turned.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}
        """
        if degrees % 90 != 0:
            raise PlanError(
                "image.rotate(): degrees must be a multiple of 90 (a free rotation would "
                f"resample every pixel and pad the corners), got {degrees}"
            )
        return ImageFunc(
            "rotate",
            self._e,
            factor=float(degrees),
            **_container("rotate", format, quality),
        )

    def flip_horizontal(self, *, format: str = "png", quality: int | None = None) -> ImageFunc:
        """Mirror the image left-to-right, re-encoded.

        The single most-used training-time augmentation. It belongs here rather than in a
        loader because it must happen on the *image*, before the tensor step, so a
        detector's boxes can be flipped alongside it.

        Args:
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> flipped = bt.col("img").image.flip_horizontal()
                >>> ds.select(d=flipped.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}
        """
        return ImageFunc(
            "flip_horizontal", self._e, **_container("flip_horizontal", format, quality)
        )

    def flip_vertical(self, *, format: str = "png", quality: int | None = None) -> ImageFunc:
        """Mirror the image top-to-bottom, re-encoded.

        Args:
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> flipped = bt.col("img").image.flip_vertical()
                >>> ds.select(d=flipped.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}
        """
        return ImageFunc("flip_vertical", self._e, **_container("flip_vertical", format, quality))

    def pad(
        self,
        width: int,
        height: int,
        *,
        fill: int = 0,
        format: str = "png",
        quality: int | None = None,
    ) -> ImageFunc:
        """Center the image on a ``(width, height)`` canvas without scaling it.

        The difference from :meth:`letterbox` is that nothing is resampled: every surviving
        pixel keeps its exact value. That is what an OCR, document or super-resolution
        pipeline needs, and what a scaling pad quietly destroys. A canvas smaller than the
        image crops it centrally rather than failing the row.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            fill: The byte value (``0..255``, applied to all three channels) the leftover
                canvas is filled with. Defaults to 0, black.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> padded = bt.col("img").image.pad(4, 4, fill=255)
                >>> ds.select(d=padded.image.decode()).to_pydict()
                {'d': [{'width': 4, 'height': 4, 'channels': 3, 'mode': 'RGB'}]}
        """
        return ImageFunc(
            "pad",
            self._e,
            width=width,
            height=height,
            fill=fill,
            **_container("pad", format, quality),
        )

    # ---- photometric ------------------------------------------------------
    def adjust_brightness(
        self, factor: float, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Scale every color channel by `factor`, clamped to the byte range.

        The `PIL.ImageEnhance.Brightness` convention, so an augmentation policy written
        against torchvision ports over unchanged: ``1.0`` is the identity, ``0.0`` black,
        ``2.0`` twice as bright.

        Args:
            factor: A non-negative multiplier.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `factor` is negative or the container is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> dark = bt.col("img").image.adjust_brightness(0.5)
                >>> ds.select(c=dark.image.mean_color()).to_pydict()
                {'c': [{'r': 64.0, 'g': 64.0, 'b': 64.0}]}
        """
        return ImageFunc(
            "adjust_brightness",
            self._e,
            factor=_non_negative("adjust_brightness", factor),
            **_container("adjust_brightness", format, quality),
        )

    def adjust_contrast(
        self, factor: float, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Push every channel away from the image's mean luma by `factor`.

        `PIL.ImageEnhance.Contrast`: ``1.0`` is the identity and ``0.0`` collapses the
        image to a flat field at its own average brightness.

        Args:
            factor: A non-negative multiplier.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `factor` is negative or the container is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> flat = bt.col("img").image.adjust_contrast(0.0)
                >>> ds.select(e=flat.image.entropy()).to_pydict()
                {'e': [0.0]}
        """
        return ImageFunc(
            "adjust_contrast",
            self._e,
            factor=_non_negative("adjust_contrast", factor),
            **_container("adjust_contrast", format, quality),
        )

    def adjust_saturation(
        self, factor: float, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Interpolate each pixel between its grey and its color by `factor`.

        `PIL.ImageEnhance.Color`: ``0.0`` is grayscale, ``1.0`` the identity, and anything
        above 1 more vivid. Unlike :meth:`convert` to ``"L"`` this keeps three channels, so
        it is an augmentation rather than a format change.

        Args:
            factor: A non-negative multiplier.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `factor` is negative or the container is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> grey = bt.col("img").image.adjust_saturation(0.0)
                >>> ds.select(g=grey.image.is_grayscale()).to_pydict()
                {'g': [True]}
        """
        return ImageFunc(
            "adjust_saturation",
            self._e,
            factor=_non_negative("adjust_saturation", factor),
            **_container("adjust_saturation", format, quality),
        )

    def adjust_hue(
        self, degrees: float, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Rotate every hue around the color wheel, leaving saturation and value alone.

        The color-jitter axis the other three adjustments cannot express, and the one a
        robustness sweep varies. Degrees wrap, so ``-30`` and ``330`` are the same shift.

        Args:
            degrees: The rotation in degrees.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> shifted = bt.col("img").image.adjust_hue(180)
                >>> ds.select(d=shifted.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}
        """
        return ImageFunc(
            "adjust_hue",
            self._e,
            factor=float(degrees),
            **_container("adjust_hue", format, quality),
        )

    def blur(
        self, sigma: float = 1.0, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Apply a Gaussian blur of standard deviation `sigma` pixels.

        Both an augmentation and a curation tool: blurring a copy and comparing
        :meth:`sharpness` separates images that carry fine detail from ones that are
        already soft.

        Args:
            sigma: The blur radius in pixels. ``0`` is the identity.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `sigma` is negative or the container is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> soft = bt.col("img").image.blur(2.0)
                >>> ds.select(d=soft.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}
        """
        return ImageFunc(
            "blur",
            self._e,
            factor=_non_negative("blur", sigma),
            **_container("blur", format, quality),
        )

    def sharpen(
        self, amount: float = 1.0, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Apply an unsharp mask of strength `amount`.

        The classical formula, ``image + amount * (image - blur(image))``, so ``0`` is the
        identity and larger values push edge contrast harder.

        Args:
            amount: The strength of the mask. ``0`` is the identity.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `amount` is negative or the container is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> crisp = bt.col("img").image.sharpen(1.5)
                >>> ds.select(d=crisp.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}
        """
        return ImageFunc(
            "sharpen",
            self._e,
            factor=_non_negative("sharpen", amount),
            **_container("sharpen", format, quality),
        )

    def invert(self, *, format: str = "png", quality: int | None = None) -> ImageFunc:
        """Take the photographic negative of each color channel, alpha untouched.

        Args:
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> negative = bt.col("img").image.invert()
                >>> ds.select(e=negative.image.entropy()).to_pydict()
                {'e': [2.0]}
        """
        return ImageFunc("invert", self._e, **_container("invert", format, quality))

    def posterize(
        self, bits: int = 4, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Reduce each color channel to its top `bits` bits.

        One of the AutoAugment/RandAugment primitives, and a cheap way to make a corpus's
        color quantization uniform. The low bits are masked off rather than rescaled, which
        is what `PIL.ImageOps.posterize` does, so ``bits=1`` leaves only 0 and 128.

        Args:
            bits: How many high bits to keep, ``1..8``. ``8`` is the identity.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `bits` is outside ``1..8`` or the container is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> flat = bt.col("img").image.posterize(1)
                >>> ds.select(c=flat.image.mean_color()).to_pydict()
                {'c': [{'r': 64.0, 'g': 64.0, 'b': 64.0}]}
        """
        if not 1 <= bits <= 8:
            raise PlanError(f"image.posterize(): bits must be in 1..8, got {bits}")
        return ImageFunc(
            "posterize",
            self._e,
            factor=float(bits),
            **_container("posterize", format, quality),
        )

    def solarize(
        self, threshold: int = 128, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Invert every channel value at or above `threshold`, leaving the rest alone.

        The other AutoAugment primitive.

        Args:
            threshold: The cutoff, ``0..255``. ``255`` leaves almost everything alone.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `threshold` is outside ``0..255`` or the container is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> solar = bt.col("img").image.solarize(128)
                >>> ds.select(c=solar.image.mean_color()).to_pydict()
                {'c': [{'r': 0.0, 'g': 0.0, 'b': 0.0}]}
        """
        if not 0 <= threshold <= 255:
            raise PlanError(f"image.solarize(): threshold must be in 0..255, got {threshold}")
        return ImageFunc(
            "solarize",
            self._e,
            factor=float(threshold),
            **_container("solarize", format, quality),
        )

    def equalize(self, *, format: str = "png", quality: int | None = None) -> ImageFunc:
        """Equalize each channel's histogram so the tonal range is used evenly.

        What rescues an under-exposed scan without a model in the loop.
        :meth:`autocontrast` is the gentler alternative: it stretches the range without
        redistributing within it.

        Args:
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> even = bt.col("img").image.equalize()
                >>> ds.select(d=even.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}
        """
        return ImageFunc("equalize", self._e, **_container("equalize", format, quality))

    def autocontrast(
        self, cutoff: float = 0.0, *, format: str = "png", quality: int | None = None
    ) -> ImageFunc:
        """Rescale each channel so its darkest and brightest surviving values hit 0 and 255.

        `PIL.ImageOps.autocontrast`. A channel with no range left to stretch — a solid
        field — is passed through untouched rather than divided by zero.

        Args:
            cutoff: The percent of each histogram tail to ignore before finding the
                extremes, ``0..49``. A few percent makes the stretch robust to a handful
                of stuck pixels.
            format: Container to write.
            quality: Encoder quality in ``1..100`` for the lossy containers.

        Returns:
            An expression evaluating to Binary image bytes; null for null or undecodable
            input.

        Raises:
            PlanError: If `cutoff` is outside ``0..49`` or the container is invalid.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> stretched = bt.col("img").image.autocontrast()
                >>> ds.select(d=stretched.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}
        """
        if not 0 <= cutoff <= 49:
            raise PlanError(f"image.autocontrast(): cutoff must be in 0..49, got {cutoff}")
        return ImageFunc(
            "autocontrast",
            self._e,
            factor=float(cutoff),
            **_container("autocontrast", format, quality),
        )

    # ---- perceptual hashes ------------------------------------------------
    def phash(self) -> ImageFunc:
        """Compute the 64-bit DCT perceptual hash, as an Int64.

        The most robust of the three fingerprints here. Where :meth:`dhash` compares
        adjacent pixels, this keeps the 8x8 lowest-frequency DCT coefficients of a 32x32
        luma reduction and thresholds them at their median — the standard ``pHash``. It
        survives re-encoding, heavy rescaling and moderate cropping far better, which is
        why a dedup pass over a scraped corpus usually confirms with this one and
        pre-filters with :meth:`ahash` or :meth:`dhash`.

        Two images are near-duplicates when few bits differ, so
        ``a.bitwise_xor(b).bit_count() <= 6`` is a similarity predicate the engine
        evaluates as ordinary integer arithmetic.

        Returns:
            An expression evaluating to an Int64 digest (the 64 bits reinterpreted, not
            clamped); null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> h = bt.col("img").image.phash()
                >>> ds.select(stable=(h == bt.col("img").image.phash())).to_pydict()
                {'stable': [True]}
        """
        return ImageFunc("phash", self._e)

    def ahash(self) -> ImageFunc:
        """Compute the 64-bit average hash, as an Int64.

        An 8x8 luma reduction thresholded at its own mean: the cheapest of the three
        fingerprints and the least discriminating. It exists because a Hamming pre-filter
        over a large corpus wants a hash that costs almost nothing, with :meth:`phash`
        confirming the survivors.

        Returns:
            An expression evaluating to an Int64 digest (the 64 bits reinterpreted, not
            clamped); null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> h = bt.col("img").image.ahash()
                >>> ds.select(stable=(h == bt.col("img").image.ahash())).to_pydict()
                {'stable': [True]}
        """
        return ImageFunc("ahash", self._e)

    # ---- curation measures ------------------------------------------------
    def entropy(self) -> ImageFunc:
        """Measure the Shannon entropy of the luma histogram, in bits.

        The curation measure that separates the cases the other two confuse.
        :meth:`brightness` cannot tell a mid-grey placeholder tile from a photograph of a
        foggy road, because both average to the middle; :meth:`sharpness` cannot tell a
        blank image from an out-of-focus one. This answers how much information is in the
        tonal distribution at all: a solid field is 0 whatever shade it is, a two-tone logo
        near 1, and a photograph of anything between 6 and 8.

        Returns:
            An expression evaluating to a Float64 in ``0..8``; null for null or
            undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> ds.select(e=bt.col("img").image.entropy()).to_pydict()
                {'e': [2.0]}
        """
        return ImageFunc("entropy", self._e)

    def colorfulness(self) -> ImageFunc:
        """Measure Hasler-Süsstrunk colorfulness — how much color the image actually has.

        The measure no luma statistic can express. A sepia-toned duplicate, a line drawing,
        a scanned page and a greyscale photograph stored as RGB all have ordinary
        brightness, sharpness and entropy, and all of them are the wrong training data for
        a model that is supposed to see color. Roughly 0 for anything grey and 15 or more
        for a vivid scene.

        Returns:
            An expression evaluating to a non-negative Float64; null for null or
            undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> vivid = ds.select(c=bt.col("img").image.colorfulness()).to_pydict()
                >>> vivid["c"][0] > 15
                True
        """
        return ImageFunc("colorfulness", self._e)

    def mean_color(self) -> ImageFunc:
        """Read the mean of each color channel as a struct ``{r, g, b}``.

        The cheapest color summary there is, and the one that makes "find every product
        shot on a white background" and "cluster this corpus by palette" ordinary
        expressions rather than an embedding model. A struct rather than three functions
        because all three come out of the same pass, exactly as :meth:`decode`'s four
        header facts do.

        Returns:
            An expression evaluating to a struct of three Float64 channel means in
            ``0..255``; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> ds.select(c=bt.col("img").image.mean_color()).to_pydict()
                {'c': [{'r': 127.5, 'g': 127.5, 'b': 127.5}]}

                >>> red = bt.col("img").image.mean_color().struct.field("r")
                >>> ds.select(r=red).to_pydict()
                {'r': [127.5]}
        """
        return ImageFunc("mean_color", self._e)

    def is_grayscale(self) -> ImageFunc:
        """Test whether every pixel satisfies ``R == G == B``.

        The fact no header carries. A corpus assembled from mixed sources is full of
        greyscale images *stored* as three identical channels: :meth:`decode` reports
        ``RGB``, :meth:`has_alpha` reports false, and nothing says that two thirds of every
        tensor is a copy. Finding them is what lets a pipeline route them to a one-channel
        model instead of paying three times the bandwidth for one channel of information.

        Returns:
            An expression evaluating to a Boolean; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> ds.select(g=bt.col("img").image.is_grayscale()).to_pydict()
                {'g': [False]}
        """
        return ImageFunc("is_grayscale", self._e)

    # ---- header-only facts ------------------------------------------------
    def aspect_ratio(self) -> ImageFunc:
        """Read width divided by height, from the header alone.

        The orientation and letterboxing decisions of a whole pipeline hang on this, and
        paying a full decode to learn it is what made people skip the check. One header
        read, like :meth:`decode`.

        Returns:
            An expression evaluating to a Float64; null for null or undecodable input, and
            null rather than infinity for a zero-height image.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> ds.select(a=bt.col("img").image.aspect_ratio()).to_pydict()
                {'a': [1.0]}
        """
        return ImageFunc("aspect_ratio", self._e)

    def has_alpha(self) -> ImageFunc:
        """Test whether the image carries an alpha channel, from the header alone.

        The flag that decides whether a corpus needs flattening before a model that takes
        three channels. Use :meth:`convert` to ``"RGB"`` to do the flattening.

        Returns:
            An expression evaluating to a Boolean; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> ds.select(a=bt.col("img").image.has_alpha()).to_pydict()
                {'a': [False]}
        """
        return ImageFunc("has_alpha", self._e)

    def format(self) -> ImageFunc:
        """Read the container format's name, sniffed from the magic bytes.

        From the bytes, never from the path. A corpus downloaded by content type is full of
        files whose extension and container disagree, and every one of them is a row that
        decodes fine and breaks whatever downstream step branched on the name. The name
        shares a vocabulary with :meth:`encode`, so a value read out of this column can be
        handed straight back to the encoder.

        Returns:
            An expression evaluating to a Utf8 container name such as ``"png"`` or
            ``"jpeg"``; null for null input or bytes matching no known container. The name
            is the one :meth:`encode` accepts and the one an image listing's ``format``
            column reports for the same bytes, so the three cannot disagree.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_2X2
                >>> ds = bt.from_pydict({"img": [_PNG_2X2]})
                >>> ds.select(f=bt.col("img").image.format()).to_pydict()
                {'f': ['png']}
        """
        return ImageFunc("format", self._e)
