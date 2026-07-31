# Prepare the data

Models rarely read raw columns. These pages cover the transforms that sit between a
source and a model, all of which run as ordinary operators, so they stream and they
distribute like everything else.

- {doc}`preprocessors/index`: scalers, encoders, imputers, binning, and composition.
- {doc}`/ml/preparing/multimodal/index`: decoding images, audio, and video into tensor columns.
- {doc}`/ml/preparing/tokenization`: tokenizing as a pipeline stage, and packing sequences.

```{toctree}
:hidden:

preprocessors/index
multimodal/index
tokenization
```
