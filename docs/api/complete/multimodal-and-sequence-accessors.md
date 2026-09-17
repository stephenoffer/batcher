# Multimodal and sequence accessors reference

This page is the generated reference for the `.image`, `.audio`, and `.video` namespaces on binary media columns and the `.seq` namespace on biological sequence columns. You reach each one from an expression, such as `col("img").image` or `col("dna").seq`, and each method has its own page.

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

## The `.seq` namespace

Methods on a text column read as DNA, RNA, protein, or a FASTQ quality string, reached as `col("s").seq`.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autoclass:: _SeqNamespace
   :no-members:
```

### Strands and translation

Validate a sequence, read its opposite strand, or transcribe and translate it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   _SeqNamespace.is_valid
   _SeqNamespace.complement
   _SeqNamespace.reverse_complement
   _SeqNamespace.transcribe
   _SeqNamespace.back_transcribe
   _SeqNamespace.translate
```

### Composition and physical properties

Count bases and compute a sequence's GC content, weight, and melting temperature.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   _SeqNamespace.base_counts
   _SeqNamespace.gc_content
   _SeqNamespace.gc_skew
   _SeqNamespace.max_homopolymer
   _SeqNamespace.molecular_weight
   _SeqNamespace.melting_temp
   _SeqNamespace.isoelectric_point
   _SeqNamespace.gravy
```

### K-mers and motifs

Break a sequence into k-mers or minimizers, and find degenerate motifs in it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   _SeqNamespace.kmers
   _SeqNamespace.canonical_kmers
   _SeqNamespace.minimizers
   _SeqNamespace.find_motif
   _SeqNamespace.count_motif
```

### Read quality

Decode a FASTQ quality string and summarize it.

```{eval-rst}
.. currentmodule:: batcher.plan.expr_ir.namespaces.sequence

.. autosummary::
   :toctree: generated
   :nosignatures:

   _SeqNamespace.phred_quality
   _SeqNamespace.mean_quality
   _SeqNamespace.expected_errors
```

## See also

- {doc}`/api/relational/expression-accessors`: every accessor method enumerated in one curated page.
- {doc}`expressions`: the `Expr` class these namespaces hang off.
- {doc}`/ml/preparing/multimodal/index`: fetching, decoding, and curating images, audio, and video.
- {doc}`/user-guide/transform/columns/sequence-accessor`: how to use the sequence accessor.
