# Images, audio, and video

These pipelines run a model over media. In all three the model is not the bottleneck, the
decode in front of it is, so each recipe is really about keeping the device busy while the
CPU turns files into tensors.

| Pipeline | What it builds |
|---|---|
| {doc}`Image classification <image-classification>` | A labelled corpus, with the JPEG decode overlapped against the forward pass |
| {doc}`Image captioning <image-captioning>` | A vision-language model over a URL column |
| {doc}`Object detection <object-detection>` | Boxes cut out of the frames they were found in, without leaving the engine |
| {doc}`Audio transcription <audio-transcription>` | Mixed-format audio resampled to what the model wants, then transcribed |

## See also

- {doc}`/ml/preparing/multimodal/index`: the decode guide behind these pipelines.
- {doc}`/benchmarks/results/ai-and-gpu`: what these shapes sustain on real hardware.

```{toctree}
:hidden:

image-classification
image-captioning
object-detection
audio-transcription
```
