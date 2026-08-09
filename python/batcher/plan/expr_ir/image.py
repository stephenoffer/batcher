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
    # `encode` only: the target container; `convert` reuses the slot for its color mode.
    # Omitted from the IR unless set, so every other image op's wire shape is unchanged.
    format: str | None = scalar(omit_none=True, default=None)
    # `letterbox` only: the byte the leftover canvas is filled with.
    fill: int | None = scalar(omit_none=True, default=None)


# A 1x1 red PNG. Exported so the doctests here (and the docs) have a real image to
# decode without reaching for a fixture file.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
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

    def encode(self, format: str) -> ImageFunc:
        """Re-encode each image in `format`, pixels unchanged.

        Normalizes a mixed-format corpus onto one codec, or trades a PNG for a smaller
        JPEG. Because JPEG has no alpha channel, an RGBA source is flattened to RGB rather
        than failing the row, which would otherwise drop every transparent image.

        Args:
            format: One of ``"png"``, ``"jpeg"``, ``"bmp"``, or ``"gif"``. WebP is
                readable but not writable, so it is not offered.

        Returns:
            An expression evaluating to the re-encoded bytes; null for null or
            undecodable input.

        Raises:
            PlanError: If `format` is not a writable format.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> jpeg = bt.col("img").image.encode("jpeg")
                >>> ds.select(m=jpeg.image.decode().struct.field("mode")).to_pydict()
                {'m': ['RGB']}
        """
        if format not in _IMAGE_FORMATS:
            raise PlanError(
                f"image.encode(): format must be one of {sorted(_IMAGE_FORMATS)}, got {format!r}"
            )
        return ImageFunc("encode", self._e, format=format)

    def convert(self, mode: str) -> ImageFunc:
        """Convert each image to color `mode`, re-encoded as PNG.

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

        Returns:
            An expression evaluating to PNG bytes in `mode`; null for null or undecodable
            input.

        Raises:
            PlanError: If `mode` is not one of the four.

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
        # `format` carries the target mode here and the target container for `encode`;
        # neither function uses the other's meaning, so they share the one string slot.
        return ImageFunc("convert", self._e, format=mode)

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

    def thumbnail(self, max_size: int) -> ImageFunc:
        """Scale so the longest side is `max_size`, keeping the aspect ratio (→ PNG bytes).

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

        Returns:
            An expression evaluating to PNG bytes; null for null or undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> small = bt.col("img").image.thumbnail(256)
                >>> ds.select(d=small.image.decode()).to_pydict()["d"][0]["width"]
                1
        """
        return ImageFunc("thumbnail", self._e, width=max_size)

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

    def auto_orient(self) -> ImageFunc:
        """Apply each image's EXIF orientation, re-encoded as PNG bytes.

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
        return ImageFunc("auto_orient", self._e)

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

    def resize(self, width: int, height: int) -> ImageFunc:
        """Resize the image and re-encode it as PNG bytes.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            An expression evaluating to Binary PNG bytes; null for null or
            undecodable input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> small = bt.col("img").image.resize(2, 2)
                >>> ds.select(d=small.image.decode()).to_pydict()
                {'d': [{'width': 2, 'height': 2, 'channels': 3, 'mode': 'RGB'}]}
        """
        return ImageFunc("resize", self._e, width=width, height=height)
