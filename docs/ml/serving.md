# Model serving

Run batch inference against an external inference server (Triton, TorchServe, or any
columnar-JSON HTTP endpoint) instead of loading the model in-process. Each adapter is
a load-once class UDF for `ds.ml.map_batches`, so preprocessing stays on CPU workers
while the model call goes to the server, and the stage parallelizes across the cluster.

```python
# docs: skip
import batcher as bt
from batcher.ml.serving import triton_client

udf = triton_client(
    "triton:8000", "resnet50", input_columns=["image"], output_columns=["logits"]
)
scored = bt.read.images("s3://bucket/imgs/", decode=True, size=(224, 224)).ml.map_batches(
    udf, concurrency=(2, 8)
)
```

## The load-once contract

An adapter returns a *class*, not a function. `map_batches` instantiates it once per
worker; the constructor opens the connection (or builds the client), and that client
is reused for every batch the worker sees. The expensive setup — the HTTP session, the
gRPC channel, the tensor metadata handshake — happens once, not per batch. If you write
your own adapter, do the connecting in `__init__` and nothing per-call but the request.

The class implements the `ServingClient` protocol: one `predict` method that takes a
dict of named NumPy arrays and returns a dict of named arrays. Batcher handles the
columnar plumbing on both sides. Input columns are pulled from the Arrow batch and
converted to NumPy in the order given by `input_columns`; output arrays come back keyed
by name and are appended as new columns. The input batch passes through unchanged, so
inference adds columns rather than replacing the row.

## Adapters

| Adapter | Backend |
| --- | --- |
| `triton_client(url, model, *, input_columns, output_columns, protocol="http", model_version="")` | NVIDIA Triton over HTTP or gRPC (`protocol="grpc"`), sending binary tensors. Needs `batcher-engine[triton]`. |
| `torchserve_client(base_url, model, *, input_columns, output_columns, timeout=30.0)` | TorchServe `/predictions/{model}`. |
| `http_client(url, *, input_columns, output_columns, headers=None, timeout=30.0, retries=3)` | Any columnar-JSON REST endpoint (KServe-style). |
| `serving_udf(connect, *, input_columns, output_columns=None)` | Build your own adapter from a zero-arg `connect()` returning a `ServingClient`. |

Use `triton_client` for tensor inputs (decoded images, embeddings) — it sends binary
tensors. `http_client` is for scalar/text features; JSON-encoding a tensor is slow and
bloated, so it warns once if asked to. When `output_columns` is omitted in
`serving_udf`, the server's response keys become the output column names.

## Batching

The batch the server sees is the morsel the pipeline hands the UDF. Set `batch_size`
on `map_batches` to control how many rows go in one `predict` call — large enough to
keep the model's accelerator busy, small enough to fit the request and the server's own
queue. `concurrency` (an int or a `(min, max)` range) sets how many worker copies of
the adapter run in parallel; with a `(min, max)` range the stage autoscales between
those bounds under load. More concurrency means more open connections to the server, so
size it against what the server can absorb.

## Errors and retries

`http_client` retries with exponential backoff on transient failures — connection
errors, timeouts, and the retryable status codes (408, 425, 429, 500, 502, 503, 504).
Other 4xx responses fail immediately, since a malformed request will not improve on a
retry. After `retries` attempts are exhausted the adapter raises `BackendError` with
the endpoint and the last error. Triton and TorchServe adapters surface backend errors
the same way. A failure propagates up through the stage; it is not silently dropped.

## Online serving

`serve_deployment` wraps the same load-once factory as a Ray Serve deployment that
answers per-request calls, coalescing concurrent requests with Serve's batching — so a
model proven in a batch pipeline serves online unchanged. Needs `batcher-engine[serve]`.

| Argument | Meaning |
| --- | --- |
| `build` | Zero-arg callable returning the predictor (`list[input] -> list[output]`); called once per replica. |
| `name` | Deployment name (default `"batcher-model"`). |
| `max_batch_size` | Max requests coalesced into one predictor call (default 16). |
| `batch_wait_timeout_s` | How long Serve waits to fill a batch before flushing (default 0.01s). |
| `**deployment_options` | Forwarded to `@serve.deployment` (e.g. `num_replicas`, `ray_actor_options`). |

```python
# docs: skip
from batcher.ml.serving import serve_deployment
from ray import serve


def build_predictor():
    import torch

    model = torch.load("model.pt").eval().cuda()

    def predict(batch):
        # batch is a list of requests coalesced by Serve; one forward pass for all.
        inputs = torch.stack([torch.as_tensor(x) for x in batch]).cuda()
        with torch.no_grad():
            out = model(inputs)
        return out.cpu().tolist()

    return predict


deployment = serve_deployment(
    build_predictor,
    name="resnet",
    max_batch_size=32,
    batch_wait_timeout_s=0.02,
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
)
serve.run(deployment.bind())
```
