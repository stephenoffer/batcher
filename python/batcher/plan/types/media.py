"""Output types for the multimodal expressions, where the shape is in the arguments.

Split from `infer` on a responsibility seam that file was about to cross: the scalar
inference there answers "what type does this arithmetic produce", while these answer "how
big is a decoded frame", which is a *sizing* question that happens to be phrased as a type.

Every image expression used to infer as `None`. That reads as harmless, since `None` means
"fall back" -- but the fallback for a width is a flat 64-byte prior, and these are the widest
columns the engine ever holds. A decode pipeline lives entirely in derived columns, so the
fallback applied to all of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.plan.expr_ir.image import ImageFunc

__all__ = ["imagefunc_type"]


# The image ops whose output is a decoded pixel tensor, and how many channels each
# produces. Their shape is not a property of the data -- it is the arguments -- so the
# width of a decoded image column is knowable before a single byte is read.
_IMAGE_TENSOR_CHANNELS = {
    "to_tensor": 3,
    "to_tensor_f32": 3,
    "center_crop": 3,
    "to_grayscale": 1,
}

# The image ops that hand back a still-encoded image, so the payload stays compressed.
_IMAGE_BINARY_FNS = frozenset({"resize", "crop", "encode", "convert"})


def imagefunc_type(expr: ImageFunc) -> pa.DataType | None:
    """The Arrow type an `.image.*` expression produces, or `None` when not certain.

    Every image expression previously inferred as `None`, which sounds harmless -- `None`
    means "fall back" -- but the fallback for a *width* is a flat 64-byte prior, and these
    are the widest columns the engine ever holds. Measured on a real decode pipeline:
    `select("id", "img")` after `.image.to_tensor(224, 224)` was costed at **16 bytes per
    row against a true 150,536**, and `select("img")` at 64 against 150,528. That is the
    ordinary shape of every image workload -- decode, then drop the compressed bytes -- and
    it was mis-sized by four orders of magnitude, in the direction that under-provisions the
    memory envelope and makes a build side look broadcastable.

    This is the same blind spot as an extension type hiding its storage from
    `plan.types.widths`, one step further down the plan: there the *source* column's type
    was unreadable, here the *derived* column's is, and a decode pipeline lives entirely in
    derived columns.

    Nothing here is inferred from a name. Each mapping was read off the engine's actual
    output: `to_tensor(w, h)` yields `fixed_shape_tensor(uint8, [h, w, 3])`, `to_tensor_f32`
    the `float32` form (`[3, h, w]` under `channels_first`), `to_grayscale` a single luma
    channel, `center_crop` the cropped RGB region, `dhash` an `int64` digest, and
    `resize`/`crop`/`encode`/`convert` a still-encoded `binary`. `decode` returns a struct of
    header facts and is deliberately left `None`: its exact field nullability is not worth
    asserting for a column that is a handful of bytes either way.

    Args:
        expr: The `ImageFunc` node to type.

    Returns:
        The output Arrow type, or `None` when it is not statically known.
    """
    if expr.fn == "dhash":
        return pa.int64()
    if expr.fn in _IMAGE_BINARY_FNS:
        return pa.binary()
    channels = _IMAGE_TENSOR_CHANNELS.get(expr.fn)
    if channels is None or expr.width is None or expr.height is None:
        return None
    value = pa.float32() if expr.fn == "to_tensor_f32" else pa.uint8()
    shape = (
        (channels, expr.height, expr.width)
        if expr.channels_first
        else (expr.height, expr.width, channels)
    )
    return pa.fixed_shape_tensor(value, shape)
