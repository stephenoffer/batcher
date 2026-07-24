"""The `.image` expression namespace — lazy, batch-level image decode.

`ImageFunc` lowers to ``{"e": "image", "fn": ...}`` IR consumed by Rust
`Expr::Image`. Decoding is library-backed, so the interpreter is the oracle and
the JIT falls back; there is one implementation, so the tiers cannot diverge.
"""

from __future__ import annotations

import base64

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.node_base import IRNode, child, expr_node, scalar
from batcher.plan.ir_tags import ExprTag

__all__ = ["ImageFunc", "_ImageNamespace"]

# Container formats `.image.encode` can write; mirrors `bc-expr`'s `ENCODE_FORMATS`.
# WebP is readable but not writable by the underlying decoder, so it is deliberately
# absent: accepting the name at plan build and failing at run time would be worse than
# rejecting it here.
_IMAGE_FORMATS = frozenset({"png", "jpeg", "bmp", "gif"})


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
    fn: str = scalar()
    input: Expr = child()
    width: int | None = scalar(omit_none=True, default=None)
    height: int | None = scalar(omit_none=True, default=None)
    # `to_tensor_f32` only: per-channel normalization and channel-first layout. Omitted
    # from the IR unless set, so the other image ops' wire shape is unchanged.
    mean: list[float] | None = scalar(omit_none=True, default=None)
    std: list[float] | None = scalar(omit_none=True, default=None)
    channels_first: bool = scalar(omit_falsy=True, default=False)
    # `crop` only: the window's top-left corner. `encode` only: the target container.
    # All three are omitted from the IR unless set, so every other image op's wire shape
    # is byte-identical to what it was.
    x: int | None = scalar(omit_none=True, default=None)
    y: int | None = scalar(omit_none=True, default=None)
    format: str | None = scalar(omit_none=True, default=None)


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

    def crop(self, x: int, y: int, width: int, height: int) -> ImageFunc:
        """Cut the window at ``(x, y)`` out of each image, as PNG bytes.

        The arbitrary-offset counterpart of :meth:`center_crop`, and the shape a detection
        pipeline needs: pull a bounding box out of a frame and keep it as an *image* rather
        than as a tensor.

        A window that runs past an edge is **clipped**, so the result can be smaller than
        requested. That is the opposite of :meth:`center_crop`, which zero-pads, and the
        difference is deliberate: `center_crop` feeds a model that needs a fixed input
        size, while a cropped image is something a person or another tool will look at,
        and inventing black pixels there would be inventing data. A window that starts
        past the image entirely is null.

        Args:
            x: Left edge of the window, in pixels from the left of the image.
            y: Top edge of the window, in pixels from the top of the image.
            width: Window width in pixels.
            height: Window height in pixels.

        Returns:
            An expression evaluating to PNG bytes of the cropped region; null for null,
            undecodable, or entirely-out-of-bounds input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.plan.expr_ir.image import _PNG_1X1
                >>> ds = bt.from_pydict({"img": [_PNG_1X1]})
                >>> region = bt.col("img").image.crop(0, 0, 1, 1)
                >>> ds.select(d=region.image.decode().struct.field("width")).to_pydict()
                {'d': [1]}
        """
        return ImageFunc("crop", self._e, width=width, height=height, x=x, y=y)

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
