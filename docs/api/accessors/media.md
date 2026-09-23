# The .image, .audio, and .video namespaces

This page is the reference for the three accessors that read inside a binary media column: `.image` on an encoded image, `.audio` on an encoded waveform, and `.video` on a clip. Reach each one from an expression, such as {py:obj}`col("img").image <batcher.plan.expr_ir.core.Expr.image>`.

Decoding is an expression like any other, so the optimizer sees it. A filter on an image's dimensions runs against the header and never decodes the pixels, and a projection that drops the decoded column stops the decode from happening at all. That is why these are methods on `Expr` rather than a Python loop over files.

## The `.image` namespace

Methods on a binary column holding images, reached as `col("img").image`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.image

.. autoclass:: _ImageNamespace
   :no-members:
```

### Decoding and metadata

Read an image's header facts without decoding its pixels.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.image

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ImageNamespace.decode
   _ImageNamespace.format
   _ImageNamespace.aspect_ratio
   _ImageNamespace.has_alpha
   _ImageNamespace.exif_orientation
```

### Resizing, cropping, and model input

Change an image's geometry, or decode it into a tensor a model can read.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.image

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ImageNamespace.resize
   _ImageNamespace.thumbnail
   _ImageNamespace.letterbox
   _ImageNamespace.pad
   _ImageNamespace.center_crop
   _ImageNamespace.crop
   _ImageNamespace.rotate
   _ImageNamespace.flip_horizontal
   _ImageNamespace.flip_vertical
   _ImageNamespace.auto_orient
   _ImageNamespace.to_tensor
   _ImageNamespace.to_tensor_f32
   _ImageNamespace.to_grayscale
```

### Color, tone, and filters

Adjust an image's color and tone, filter it, or re-encode it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.image

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ImageNamespace.adjust_brightness
   _ImageNamespace.adjust_contrast
   _ImageNamespace.adjust_saturation
   _ImageNamespace.adjust_hue
   _ImageNamespace.autocontrast
   _ImageNamespace.equalize
   _ImageNamespace.invert
   _ImageNamespace.posterize
   _ImageNamespace.solarize
   _ImageNamespace.blur
   _ImageNamespace.sharpen
   _ImageNamespace.convert
   _ImageNamespace.encode
```

### Quality signals and perceptual hashes

Score an image's brightness, sharpness, and color, or hash it for near-duplicate detection.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.image

.. autosummary::
   :toctree: generated
   :nosignatures:

   _ImageNamespace.brightness
   _ImageNamespace.sharpness
   _ImageNamespace.colorfulness
   _ImageNamespace.entropy
   _ImageNamespace.mean_color
   _ImageNamespace.is_grayscale
   _ImageNamespace.phash
   _ImageNamespace.dhash
   _ImageNamespace.ahash
```

## The `.audio` namespace

Methods on a binary column holding audio, reached as `col("clip").audio`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.audio

.. autoclass:: _AudioNamespace
   :no-members:
```

### Decoding and resampling

Read a clip's metadata, decode it to a waveform, and cut or resample it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.audio

.. autosummary::
   :toctree: generated
   :nosignatures:

   _AudioNamespace.decode
   _AudioNamespace.to_waveform
   _AudioNamespace.resample
   _AudioNamespace.slice
   _AudioNamespace.pad_or_trim
   _AudioNamespace.trim_silence
   _AudioNamespace.encode_wav
```

### Levels and normalization

Measure a clip's loudness, clipping, and silence, and normalize its level.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.audio

.. autosummary::
   :toctree: generated
   :nosignatures:

   _AudioNamespace.rms
   _AudioNamespace.dbfs
   _AudioNamespace.peak_dbfs
   _AudioNamespace.clipping_ratio
   _AudioNamespace.silence_ratio
   _AudioNamespace.rms_normalize
   _AudioNamespace.peak_normalize
   _AudioNamespace.pre_emphasis
```

### Spectral features

Compute spectrograms and the spectral features a speech or audio model reads.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.audio

.. autosummary::
   :toctree: generated
   :nosignatures:

   _AudioNamespace.spectrogram
   _AudioNamespace.mel_spectrogram
   _AudioNamespace.mfcc
   _AudioNamespace.spectral_centroid
   _AudioNamespace.spectral_bandwidth
   _AudioNamespace.spectral_rolloff
   _AudioNamespace.spectral_flatness
   _AudioNamespace.zero_crossing_rate
```

## The `.video` namespace

Methods on a binary column holding video, reached as `col("clip").video`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.video

.. autoclass:: _VideoNamespace
   :no-members:
```

Read a clip's metadata, or sample frames from it as images.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.video

.. autosummary::
   :toctree: generated
   :nosignatures:

   _VideoNamespace.decode
   _VideoNamespace.thumbnail
   _VideoNamespace.frame_at
   _VideoNamespace.frames
```

## See also

- {doc}`index`: the other accessor namespaces, and which column kind each one attaches to.
- {doc}`/api/relational/expression-accessors`: the same methods with a runnable example per namespace.
- {doc}`/api/symbols/expression-methods`: the {py:obj}`Expr <batcher.plan.expr_ir.core.Expr>` these namespaces hang off.
- {doc}`/user-guide/transform/columns/media-accessor`: the guide these methods are the reference for.
- {doc}`/ml/preparing/multimodal/index`: fetching, decoding, curating, and augmenting media at scale.
- {doc}`/cookbook/ml/pipelines/multimodal/index`: complete captioning, classification, and transcription pipelines.
