# Object detection

A detection pipeline has a second half that a classification pipeline does not. The model
returns boxes, and boxes are only useful once you have cut them out of the frames they were
found in — to crop a product from a shelf photo, to feed a face to a recognizer, to build a
review set of what the model claims it saw.

That second half is where these pipelines usually leave the engine. The boxes are data, one
per row, so cropping them looks like something only Python can do. It is not: every step
here is an expression, and the whole pipeline stays one lazy plan.

## Preprocess without moving the boxes

Detection models want a fixed input size, and the obvious way to get one is wrong.
{py:meth}`.image.to_tensor(w, h) <batcher.plan.expr_ir.image._ImageNamespace.to_tensor>`
stretches the image to fit, which moves every box the model predicts off its object by
however much the aspect ratio changed. The error is invisible in the output — the boxes are
well-formed, the shape is right, they are just in the wrong place.

{py:meth}`.image.letterbox(w, h) <batcher.plan.expr_ir.image._ImageNamespace.letterbox>`
scales the whole image to fit, centres it, and fills the remainder with a constant the model
learns to ignore. The aspect ratio survives, so the geometry the model predicts is the
geometry of the original.

```python
# docs: skip
import batcher as bt
from batcher import col

frames = bt.read.images("s3://bucket/shelves/")
model_input = frames.with_columns(x=col("bytes").image.letterbox(640, 640))
```

The default fill is `114`, the YOLO family's grey, so a model trained against that
preprocessing sees the padding it expects. Pass `fill=` for anything else.

:::{tip}
Put {py:meth}`.image.auto_orient() <batcher.plan.expr_ir.image._ImageNamespace.auto_orient>`
in front of it for a corpus of photographs. A camera records which way up it was held in
the EXIF tag rather than rotating the pixels, so a portrait phone photo is *stored*
landscape — and a detector run on the stored orientation is looking at a sideways world.
:::

## Run the detector

The model stage is an ordinary `infer`, with the model passed as a **class** so its weights
load once per worker rather than once per batch:

```python
# docs: skip
import pyarrow as pa


class Detector:
    def __init__(self):
        import torch

        self.torch = torch
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s").cuda().eval()

    def __call__(self, batch):
        images = batch.column("x").to_numpy_ndarray()
        x = self.torch.from_numpy(images).cuda().permute(0, 3, 1, 2).float().div(255)
        with self.torch.no_grad():
            results = self.model(x)
        # One row per image, each holding its boxes as a list of structs.
        boxes = [
            [
                {"x": int(b[0]), "y": int(b[1]), "w": int(b[2] - b[0]), "h": int(b[3] - b[1])}
                for b in image_boxes
            ]
            for image_boxes in results.xyxy
        ]
        return batch.append_column("boxes", pa.array(boxes))


detected = model_input.ml.infer(
    Detector,
    output_columns=["path", "bytes", "boxes"],
    batch_size=32,
    num_gpus=1,
    concurrency=4,
)
```

## Cut the boxes out

This is the step that used to need a Python loop. A detection is one row of `{x, y, w, h}`,
so {py:meth}`.image.crop <batcher.plan.expr_ir.image._ImageNamespace.crop>` takes its window
from those columns:

```python
# docs: skip
patches = (
    detected.explode("boxes")
    .with_columns(
        patch=col("bytes").image.crop(
            col("boxes").struct.field("x"),
            col("boxes").struct.field("y"),
            col("boxes").struct.field("w"),
            col("boxes").struct.field("h"),
        )
    )
    .filter(col("patch").is_not_null())
)
patches.select("path", "patch").write.parquet("s3://bucket/crops.parquet")
```

`explode` turns one row per image into one row per box, and the crop reads each row's own
window. Constants and columns mix freely, so a fixed-size patch around a per-row centre is
`crop(col("cx"), col("cy"), 64, 64)`.

A window that runs past an edge is clipped to what exists rather than padded — a crop is
something you look at, and inventing black pixels invents data. A window that is null,
negative, empty, or entirely outside the image nulls **that row only**, which is why the
`filter` above is a cheap tidy-up rather than a rescue: a detector that declines to predict
on some frames costs you those rows and nothing else.

The result is encoded bytes, because the boxes genuinely differ in size. Feed a second-stage
model by putting them back on one shape:

```python
# docs: skip
recognizer_input = patches.with_columns(x=col("patch").image.letterbox(224, 224))
```

## Build a review set

Detections are worth looking at before they are trusted, and the crops make that a query.
A contact sheet of the largest boxes per class, as thumbnails rather than tensors:

```python
# docs: skip
review = (
    patches.with_columns(
        area=col("boxes").struct.field("w") * col("boxes").struct.field("h"),
        thumb=col("patch").image.thumbnail(128),
    )
    .sort("area", descending=True)
    .limit(200)
    .select("path", "label", "thumb")
)
```

{py:meth}`.image.thumbnail <batcher.plan.expr_ir.image._ImageNamespace.thumbnail>` keeps the
aspect ratio and never upscales, so a small crop stays small rather than being blown up into
invented detail.

## Detections in video

The same shape works on clips, with one addition: a detection in a video carries a
*timestamp*, and
{py:meth}`.video.frame_at <batcher.plan.expr_ir.video._VideoNamespace.frame_at>` takes that
timestamp from a column. So a row that says "something at 3.5 seconds" can produce the
picture of it:

```python
# docs: skip
stills = events.with_columns(still=col("clip").video.frame_at(col("t"), 640))
```

Seeking costs about the same wherever in the clip the moment is, so a table of events
scattered through a long recording does not degrade into a full decode of it. Sample frames
for a *model* with {py:meth}`.video.frames <batcher.plan.expr_ir.video._VideoNamespace.frames>`
instead, which gives every row the same `(n, h, w, 3)` shape.

## Requirements and limitations

- The crop bounds must be integer columns. Cast a float box (`col("x").cast("int64")`)
  before cropping; the engine will not round on your behalf, because rounding a coordinate
  is a decision about half-pixels that belongs to the caller.
- Cropping decodes the source image once per box. When a frame has many boxes, `explode`
  means the same image is decoded once per row — cheap for JPEG thumbnails, worth measuring
  for large scans.
- `.video.frame_at` needs an engine built with the `video` cargo feature. See
  {doc}`/ml/preparing/multimodal/video`.

## See also

- {doc}`/ml/preparing/multimodal/decoding`: choosing between `to_tensor`, `letterbox`, and
  `thumbnail`, and what a bad row does.
- {doc}`/ml/preparing/multimodal/video`: sampling frames and pulling stills from clips.
- {doc}`image-classification`: the simpler shape, where the decode is the whole story.
