# Prepare the data

Models rarely read raw columns. These pages cover the transforms that sit between a
source and a model. Every one of them is an ordinary operator, so a preparation step
streams and distributes exactly like the rest of a plan.

- {doc}`preprocessors/index`: scalers, encoders, imputers, binning, and composition.
- {doc}`/ml/preparing/multimodal/index`: decoding images, audio, and video into tensor columns.
- {doc}`/ml/preparing/tokenization`: tokenizing as a pipeline stage, and packing sequences.

```{toctree}
:hidden:

preprocessors/index
multimodal/index
tokenization
```
