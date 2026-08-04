# Working with video

This page covers turning video clips into columns: sampling frames for a model, pulling a
single still out of a clip, and reading a clip's metadata without decoding it.

Video is the modality where the difference between decoding in the engine and decoding in
Python matters most, because a clip is large and a frame is larger. The {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>` accessor
runs in the data plane, so a video pipeline is a lazy expression like any other rather
than a loop over rows.

## What the accessor does

Four operations, covering the three things a pipeline asks of a clip:

| Call | Result | Use it for |
| --- | --- | --- |
| {py:meth}`.video.decode() <batcher.plan.expr_ir.video._VideoNamespace.decode>` | struct `{width, height, num_frames, duration_secs, fps}` | filtering a corpus before you decode any of it |
| {py:meth}`.video.frames(num_frames, width, height) <batcher.plan.expr_ir.video._VideoNamespace.frames>` | `(num_frames, height, width, 3)` uint8 tensor | feeding a video model |
| {py:meth}`.video.thumbnail(max_size) <batcher.plan.expr_ir.video._VideoNamespace.thumbnail>` | PNG bytes of the middle frame | review, curation, contact sheets |
| {py:meth}`.video.frame_at(second, max_size) <batcher.plan.expr_ir.video._VideoNamespace.frame_at>` | PNG bytes of the frame at `second` | a row that already carries a timestamp |

A clip that will not decode yields null rather than failing the batch, the same convention
the {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>` and {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>` accessors follow. Null is deliberate rather than a black frame:
zeros are indistinguishable from a legitimately black clip, so they put blank samples into
a training set with nothing to detect them by.

## Read the metadata before you decode

`decode()` parses the container header and stops. It never touches a frame, so it is cheap
enough to run over a whole corpus, and it is how you drop the clips you did not want before
paying to decode them:

```python
# docs: skip
import batcher as bt
from batcher import col

clips = bt.read.video("s3://bucket/clips/")
usable = clips.filter(
    (col("bytes").video.decode().struct.field("duration_secs") > 2.0)
    & (col("bytes").video.decode().struct.field("width") >= 640)
)
```

The engine computes the struct once and reuses it, so naming two fields off it costs one
header read rather than two.

## Sample frames for a model

`frames` is the video counterpart of {py:meth}`.image.to_tensor <batcher.plan.expr_ir.image._ImageNamespace.to_tensor>`. It samples evenly-spaced frames
and resizes each to a fixed size, which is what makes every row the same shape whatever
the source clip's resolution or length was:

```python
# docs: skip
import batcher as bt
from batcher import col

tensors = bt.read.video("s3://bucket/clips/").with_columns(
    x=col("bytes").video.frames(8, 224, 224)
)
```

The result is a fixed-shape tensor column of `(8, 224, 224, 3)` uint8, tagged with the
canonical `arrow.fixed_shape_tensor` extension type. The shape travels with the data, so
the column reaches a model already shaped and the planner knows the row is 1.2 MB wide
rather than guessing.

The frames are the indices `numpy.linspace(0, num_frames - 1, n)` names, which is what the
reference preprocessing of the common video models uses, and they are found by decoding
the clip in order — so the *n*-th frame really is the *n*-th. The clip is never decoded
past the last wanted frame, and only the wanted frames are kept, so peak memory is the
output plus one frame rather than the whole clip. That distinction is worth the sentence:
a minute of 1080p at 30 fps is about 11 GB decoded, for eight frames of output.

A clip with fewer frames than you asked for repeats frames rather than yielding a shorter
row, because a ragged row would not be a fixed-shape tensor.

{py:meth}`bt.read.video(..., decode=True, size=(h, w), num_frames=n) <batcher.api.io_namespace.reader.Reader.video>` is the same thing spelled as a
reader argument, and it appends the column as `frames`.

## Pull a single still

`thumbnail` and `frame_at` hand back PNG bytes rather than a tensor, because a still is
something a person or another tool looks at:

```python
# docs: skip
import batcher as bt
from batcher import col

# A contact sheet of the corpus.
sheet = bt.read.video("s3://bucket/clips/").select(
    path=col("path"), thumb=col("bytes").video.thumbnail(320)
)

# The picture at a timestamp a detection already gave you.
stills = detections.with_columns(still=col("clip").video.frame_at(3.5, 640))
```

Both scale the frame so its longest side is `max_size`, keeping the clip's aspect ratio
and never upscaling. That is the rule the whole media surface follows: an operation that
hands back an **encoded still** takes a longest side and keeps the shape, while one that
hands back a **tensor** takes exact dimensions. A tensor feeds a model that needs every
row identical; a still is looked at, and a squashed 16:9 frame is a distortion nothing
downstream can see. It is also what makes `.image.thumbnail` and `.video.thumbnail` the
same operation rather than two methods that share a name.

`thumbnail` takes the frame halfway through the clip, not the first one. The first frame of
a real clip is very often black, a title card, or a fade-in, which makes a corpus of
first-frame thumbnails useless for the review work thumbnails exist for.

Both seek to the keyframe before the target and decode forward from there, so the cost is
bounded by the keyframe interval rather than by how far into the clip the target is.
`frame_at` returns the frame a player displays at that instant. A `second` past the end of
a clip whose duration is known yields null rather than the last frame, because handing back
a frame under a timestamp that does not exist invents data.

The output is an image column, so the `.image` accessor reads it back:

```python
# docs: skip
from batcher import col

sheet.with_columns(dims=col("thumb").image.decode())
```

## Which decoder is running

Native video decode links the system FFmpeg, so it is an optional build. The engine
reports what it was compiled with:

```python
from batcher._internal.native import engine_features

print("video" in engine_features())
```

On a build that reports `True`, `.video` expressions and `bt.read.video(decode=True)` run
in the data plane, row-parallel, with no Python in the loop. On a build that reports
`False`:

- `bt.read.video(decode=True)` falls back to a per-row `PyAV` loop, which needs the
  `batcher-engine[video]` extra. It returns the same frames.
- A bare `.video` expression raises rather than falling back, because an expression names
  the native kernel directly and silently substituting a different implementation for it
  would be worse than saying so.

The fallback decodes `decode_concurrency` clips at once (PyAV releases the GIL inside the
codec, so the work genuinely overlaps). The cost is memory: peak residency is that many
clips rather than one, so lower it to `1` for GB-sized clips.

## Requirements and limitations

- Native decode requires an engine built with the `video` cargo feature and the system
  FFmpeg development libraries present at build time.
- The Python fallback requires `pip install 'batcher-engine[video]'`.
- `frames` decodes the clip up to its last sampled frame. Sampling the final frame of a
  long clip therefore costs a full decode; sampling from the first half does not.
- `thumbnail` and `frame_at` are keyframe-bounded, not free: a clip encoded as one long
  GOP decodes from its single keyframe to the target.
- FFmpeg reads from a path, so each clip is written to a temp file for the duration of its
  decode. Size the node's temp filesystem for `clip size x concurrent rows`.

## See also

- {doc}`/ml/preparing/multimodal/decoding`: fetching bytes, and decoding images and audio.
- {doc}`/ml/preparing/multimodal/curating`: dropping the rows that decode perfectly and
  teach a model nothing.
- {doc}`/api/relational/expression-accessors`: the full `.video`, `.image`, and `.audio`
  method reference.
