# Memory

Start with the contract, then the accounting, then what happens when the accounting says no.

- {doc}`Arrow memory model </architecture/deep-dives/memory/arrow-memory>`: the only columnar contract, and what zero-copy really buys.
- {doc}`Tensor columns </architecture/deep-dives/memory/tensor-columns>`: how an image becomes a column without a Python round trip.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: the process-wide byte account every allocation of consequence reserves against.
- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: staying alive when the data does not fit.

```{toctree}
:hidden:

arrow-memory
tensor-columns
buffer-pool
spilling
```
