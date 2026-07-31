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
is reused for every batch the worker sees. The expensive setup (the HTTP session, the
gRPC channel, the tensor metadata handshake) happens once, not per batch. If you write
your own adapter, do the connecting in `__init__` and leave nothing per call but the
request itself.

The class implements the `ServingClient` protocol: one `predict` method that takes a
dict of named NumPy arrays and returns a dict of named arrays. Batcher handles the
columnar plumbing on both sides. Input columns are pulled from the Arrow batch and
converted to NumPy in the order given by `input_columns`; output arrays come back keyed
by name and are appended as new columns. The input batch passes through unchanged, so
inference adds columns rather than replacing the row.

Shapes survive the round trip. A tensor input column, meaning every row holds a
same-shape N-d array such as a decoded image, keeps its `(N, *shape)` form across the
boundary. A 1-D output array becomes a scalar column, and a higher-rank output becomes a
tensor column.

## Adapters

One adapter exists per serving backend, and each takes the input and output column names
so the batch maps onto the server's tensor signature:

| Adapter | Backend |
| --- | --- |
| `triton_client(url, model, *, input_columns, output_columns, protocol="http", model_version="")` | NVIDIA Triton over HTTP or gRPC (`protocol="grpc"`), sending binary tensors. Needs `batcher-engine[triton]`. |
| `torchserve_client(base_url, model, *, input_columns, output_columns, timeout=30.0)` | TorchServe `/predictions/{model}`. |
| `http_client(url, *, input_columns, output_columns, headers=None, timeout=30.0, retries=3)` | Any columnar-JSON REST endpoint (KServe-style). |
| `serving_udf(connect, *, input_columns, output_columns=None)` | Build your own adapter from a zero-arg `connect()` returning a `ServingClient`. |

Use `triton_client` for tensor inputs (decoded images, embeddings): it sends binary
tensors, and maps NumPy dtypes, including `bf16` and the `fp8` variants modern
transformers serve in, to Triton's KServe-v2 dtype vocabulary. `http_client` is for
scalar and text features. JSON-encoding a tensor is slow and bloated, so it warns once
if asked to. `torchserve_client` is `http_client` pointed at `/predictions/{model}`, so
a TorchServe handler that accepts and returns `{column: [values...]}` works with no
extra glue.

## Writing your own adapter

`serving_udf` builds an adapter from a `connect()` callable that returns anything
implementing `ServingClient`, meaning a `predict({col: ndarray}) -> {col: ndarray}`
method. `connect()` runs once per worker, so do the client setup there and keep
`predict` to the request itself. When `output_columns` is omitted, the keys of the
returned dict become the output column names.

```python
# docs: skip
from batcher.ml.serving import serving_udf


class MyClient:
    def __init__(self, endpoint):
        self.session = open_session(endpoint)  # the expensive, once-per-worker setup

    def predict(self, inputs):
        logits = self.session.run(inputs["features"])
        return {"logits": logits}


udf = serving_udf(
    lambda: MyClient("grpc://model-server:9000"),
    input_columns=["features"],
    output_columns=["logits"],
)
scored = ds.ml.map_batches(udf, concurrency=(2, 8))
```

The protocol is small enough to exercise without a server. Any object with a `predict`
method satisfies it, so this runs as written:

```python
import batcher as bt
from batcher.ml.serving import serving_udf

class LocalClient:
    def __init__(self):
        self.bias = 0.5          # stands in for the once-per-worker connection

    def predict(self, inputs):
        return {"score": inputs["features"] * 2 + self.bias}

udf = serving_udf(LocalClient, input_columns=["features"], output_columns=["score"])
ds = bt.from_pydict({"features": [1.0, 2.0, 3.0]})
print(ds.ml.map_batches(udf).to_pydict())
# {'features': [1.0, 2.0, 3.0], 'score': [2.5, 4.5, 6.5]}
```

The input columns pass through unchanged and `score` is appended, which is the same
shape a real Triton or TorchServe call produces.

When the server returns raw `logits` (a list per row), turn them into a probability
distribution in the data plane with `.list.softmax()`, and rank the classes with
`.list.arg_sort()` reversed for highest-first. No per-row Python runs:

```python
# docs: skip
from batcher import col
scored = scored.with_columns(
    prob=col("logits").list.softmax(),
    ranked=col("logits").list.arg_sort().list.reverse(),  # class indices, best first
)
```

`arg_sort` gives you positions, and `.list.gather(...)` is what spends them. Together they
turn a score vector and a candidate list into a ranked selection without a per-row loop, which
is the shape of a reranking stage:

```python
import batcher as bt

candidates = bt.from_pydict(
    {"docs": [["low", "high", "mid"]], "scores": [[0.1, 0.9, 0.5]]}
)
best_first = bt.col("scores").list.arg_sort().list.reverse()
print(candidates.select(top2=bt.col("docs").list.gather(best_first.list.head(2))).to_pydict())
# {'top2': [['high', 'mid']]}
```

A cutoff wider than the candidate list is fine — the extra positions come back as nulls rather
than an error, because a fixed `k` against a short candidate set is ordinary.

Two more read the scores rather than reorder them. `.list.log_softmax()` is the log-domain
distribution, and it is not the same as taking the log of `softmax`: a probability small enough
to underflow to zero there becomes `-inf`, while the log form stays finite. That is the whole
reason a scoring pipeline carries log-probabilities.

`.list.entropy()` reduces a row to its uncertainty in nats — zero when the model put all its
mass on one class, `ln n` when it spread evenly over `n`. It is the routing signal for a
cascade: answer the confident rows from the small model and send the rest somewhere more
expensive.

```python
preds = bt.from_pydict({"id": [1, 2], "prob": [[0.99, 0.01], [0.5, 0.5]]})
unsure = preds.filter(bt.col("prob").list.entropy() > bt.lit(0.5))
print(unsure.select("id").to_pydict())
# {'id': [2]}
```

## Batching

The batch the server sees is the morsel the pipeline hands the UDF. Set `batch_size`
on `map_batches` to control how many rows go in one `predict` call: large enough to
keep the model's accelerator busy, small enough to fit the request and the server's own
queue. `concurrency` (an int or a `(min, max)` range) sets how many worker copies of
the adapter run in parallel; with a range, the stage autoscales between those bounds
under load. More concurrency means more open connections, so size it against what the
server can absorb.

## Errors and retries

`http_client` retries with exponential backoff on transient failures: connection
errors, timeouts, and the retryable status codes (408, 425, 429, 500, 502, 503, 504).
Other 4xx responses fail immediately, since a malformed request will not improve on a
retry. Once `retries` attempts are exhausted the adapter raises `BackendError` with the
endpoint and the last error. Triton and TorchServe surface backend errors the same way.
Nothing is dropped silently; a failure propagates up through the stage.

## From batch to online serving

The same load-once factory that backs a batch stage can stand up an online endpoint.
`serve_deployment` wraps it as a Ray Serve deployment that answers per-request calls,
coalescing concurrent requests with Serve's native batching. A model proven in a batch
pipeline then serves online unchanged, with no second execution engine to maintain. The
offline `map_batches` adapter and this online deployment share one `build` factory, so
what you validate in batch is what runs at the endpoint. Needs `batcher-engine[serve]`.

| Argument | Meaning |
| --- | --- |
| `build` | Zero-arg callable returning the predictor (`list[input] -> list[output]`); called once per replica. |
| `name` | Deployment name (default `"batcher-model"`). |
| `max_batch_size` | Max requests coalesced into one predictor call (default 16). |
| `batch_wait_timeout_s` | How long Serve waits to fill a batch before flushing (default 0.01s). |
| `**deployment_options` | Forwarded to `@serve.deployment` (e.g. `num_replicas`, `ray_actor_options`, `autoscaling_config`). |

The `build` factory returns a *batched* predictor: it is handed the list of requests
Serve coalesced (up to `max_batch_size`, or whatever arrived within
`batch_wait_timeout_s`) and runs one forward pass for the whole list, so the GPU sees a
real batch even under per-request traffic. Tune `max_batch_size` and
`batch_wait_timeout_s` together: a bigger batch and a longer wait trade a little
latency for throughput.

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

## See also

- {doc}`Inference </ml/inference/inference>`: in-process batch inference and the `.ml` accessor.
- {doc}`GPU scheduling </ml/inference/gpu>`: `num_gpus` and `concurrency` for GPU stages.
