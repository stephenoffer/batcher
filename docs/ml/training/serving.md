# Model serving

This page covers running batch inference against an external inference server, such as Triton,
TorchServe, or any JSON HTTP endpoint, and putting a batch-validated model behind an online
endpoint with Ray Serve.

Each adapter is a load-once class UDF for
{py:meth}`ds.map_batches <batcher.Dataset.map_batches>`. Preprocessing stays on
CPU workers, the model call goes to the server, and the stage parallelizes across the cluster.
{doc}`Model serving patterns </ml/training/model-serving-patterns>` explains when a served model is
the right choice over loading the weights in the worker.

```python
# docs: skip
import batcher as bt
from batcher.ml.serving import triton_client

udf = triton_client("triton:8000", "resnet50", input_columns=["image"], output_columns=["logits"])
scored = bt.read.images("s3://bucket/imgs/", decode=True, size=(224, 224)).map_batches(
    udf, concurrency=(2, 8)
)
```

## The load-once contract

An adapter returns a *class*, not a function. `map_batches` instantiates it once per worker, the
constructor opens the connection, and that client serves every batch the worker sees. The HTTP
session, the gRPC channel, and the tensor metadata handshake happen once, not per batch.

The client implements the {py:class}`ServingClient <batcher.ml.ServingClient>` protocol: one
`predict` method that takes a dict of named NumPy arrays and returns a dict of named arrays.
Batcher handles the columnar plumbing on both sides. Input columns are pulled from the Arrow batch
and converted to NumPy in the order `input_columns` gives. Output arrays come back keyed by name
and are appended as new columns, so the input columns pass through unchanged.

Inputs must be numeric or tensor columns. A string or nested column has no array form an endpoint
can take, so the adapter raises `BackendError` naming the column. Select or cast it before the
stage.

Shapes survive the round trip. A tensor input column, where every row holds a same-shape array
such as a decoded image, keeps its `(N, *shape)` form across the boundary. A 1-D output array
becomes a scalar column, and a higher-rank output becomes a tensor column.

## Adapters

There's one adapter per serving backend, and each takes the input and output column names so the
batch maps onto the server's tensor signature. All four also take `max_batch_size` and
`pipeline_depth`, described in the next section.

| Adapter | Backend |
| --- | --- |
| `triton_client(url, model, *, input_columns, output_columns, protocol="http", model_version="", retries=2, timeout=None)` | NVIDIA Triton over HTTP, or gRPC with `protocol="grpc"`, sending binary tensors. Needs `batcher-engine[triton]`. |
| `torchserve_client(base_url, model, *, input_columns, output_columns, timeout=30.0, retries=3, tensor_encoding="json")` | TorchServe `/predictions/{model}`. |
| `http_client(url, *, input_columns, output_columns, headers=None, timeout=30.0, retries=3, tensor_encoding="auto")` | Any columnar-JSON REST endpoint, KServe-style. |
| `serving_udf(connect, *, input_columns, output_columns=None, retries=2, retry_backoff=0.5)` | Your own adapter, from a zero-arg `connect()` returning a `ServingClient`. |

`triton_client` sends binary tensors and maps NumPy dtypes to Triton's KServe-v2 vocabulary,
including `bf16` and the `fp8` variants modern transformers serve in. Set `timeout` on it: without
one, a wedged Triton replica blocks the worker forever and the retry never fires.

`http_client` sends a tensor input, anything of rank above 1, in a compact binary envelope by
default. `tensor_encoding="json"` keeps the legacy nested-list shape and warns once, because it
costs a Python conversion per element. `torchserve_client` is `http_client` pointed at
`/predictions/{model}`, and it defaults to `"json"` because that's the shape a stock TorchServe
handler expects. Pass `tensor_encoding="auto"` when your handler decodes the binary envelope.

## How many rows go in one request

An engine batch is thousands of rows, and a serving endpoint is configured for far fewer. A Triton
model config commonly names `max_batch_size: 8`, and Triton answers a request above that window
with an error rather than predictions. An HTTP endpoint answers with a 413, a timeout, or an
out-of-memory error on its own GPU. `max_batch_size` splits each Arrow batch into requests the
server can hold. `pipeline_depth` keeps that many requests in flight, so the remote GPU isn't idle
while this worker encodes and decodes. Results stay in input order either way.

Leave `max_batch_size` unset and `triton_client` reads the window from the model's configuration
on the server, which is where the number is declared and where it can be right. The HTTP adapters
can't ask, so state it yourself. For TorchServe it's the model's registered `batch_size`, and for a
custom endpoint it's whatever that endpoint was built for. A custom `ServingClient` can declare its
window with an optional `batch_window()` method.

```python
# docs: skip
udf = triton_client(
    "triton:8000",
    "resnet50",
    input_columns=["image"],
    output_columns=["logits"],
    pipeline_depth=4,  # four requests in flight; the window comes from the server
)
```

Above the adapter, `batch_size` on `map_batches` sets how many rows reach one call, and
`concurrency`, an int or a `(min, max)` range, sets how many worker copies run in parallel. With a
range, the stage autoscales between those bounds under load. Every copy holds its own connection,
so size `concurrency` against what the server can absorb.

## Writing your own adapter

`serving_udf` builds an adapter from a `connect()` callable that returns anything with a
`predict({col: ndarray}) -> {col: ndarray}` method. `connect()` runs once per worker, so do the
client setup there and keep `predict` to the request itself. If the client also defines `warmup()`,
it's called once at connect time. When `output_columns` is omitted, the keys of the returned dict
become the output column names.

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
scored = ds.map_batches(udf, concurrency=(2, 8))
```

The protocol is small enough to exercise without a server. Any object with a `predict` method
satisfies it, so this runs as written:

```python
import batcher as bt
from batcher.ml.serving import serving_udf


class LocalClient:
    def __init__(self):
        self.bias = 0.5  # stands in for the once-per-worker connection

    def predict(self, inputs):
        return {"score": inputs["features"] * 2 + self.bias}


udf = serving_udf(LocalClient, input_columns=["features"], output_columns=["score"])
ds = bt.from_pydict({"features": [1.0, 2.0, 3.0]})
print(ds.map_batches(udf).to_pydict())
# {'features': [1.0, 2.0, 3.0], 'score': [2.5, 4.5, 6.5]}
```

The input column passes through and `score` is appended, the same shape a real Triton or
TorchServe call produces.

## Working with the scores

When the server returns raw `logits` as a list per row, turn them into a probability distribution
in the data plane with
{py:meth}`.list.softmax() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.softmax>`,
and rank the classes with
{py:meth}`.list.arg_sort() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.arg_sort>`
reversed for highest-first. No per-row Python runs:

```python
# docs: skip
from batcher import col

scored = scored.with_columns(
    prob=col("logits").list.softmax(),
    ranked=col("logits").list.arg_sort().list.reverse(),  # class indices, best first
)
```

{py:meth}`arg_sort <batcher.plan.expr_ir.namespaces.collections._ListNamespace.arg_sort>` gives you
positions and
{py:meth}`.list.gather(...) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.gather>`
spends them. Together they turn a score vector and a candidate list into a ranked selection, which
is the shape of a reranking stage:

```python
import batcher as bt

candidates = bt.from_pydict({"docs": [["low", "high", "mid"]], "scores": [[0.1, 0.9, 0.5]]})
best_first = bt.col("scores").list.arg_sort().list.reverse()
print(candidates.select(top2=bt.col("docs").list.gather(best_first.list.head(2))).to_pydict())
# {'top2': [['high', 'mid']]}
```

A cutoff wider than the candidate list is fine. The extra positions come back as nulls, because a
fixed `k` against a short candidate set is ordinary.

{py:meth}`.list.log_softmax() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.log_softmax>`
is the log-domain distribution, and it isn't the log of `softmax`. A probability small enough to
underflow to zero there becomes `-inf`, while the log form stays finite. That's why a scoring
pipeline carries log-probabilities.

{py:meth}`.list.entropy() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.entropy>`
reduces a row to its uncertainty in nats: zero when the model put all its mass on one class, `ln n`
when it spread evenly over `n`. That's the routing signal for a cascade. Answer the confident rows
from the small model and send the rest somewhere more expensive.

```python
preds = bt.from_pydict({"id": [1, 2], "prob": [[0.99, 0.01], [0.5, 0.5]]})
unsure = preds.filter(bt.col("prob").list.entropy() > bt.lit(0.5))
print(unsure.select("id").to_pydict())
# {'id': [2]}
```

## Errors and retries

`http_client` retries with exponential backoff on transient failures: connection errors, timeouts,
and the retryable status codes 408, 425, 429, 500, 502, 503, and 504. When the server sends a
`Retry-After` header on a 429 or 503, the wait follows it, capped so one response can't stall a
worker. Other 4xx responses fail immediately, since a malformed request won't improve on a retry.
When the attempts run out, the adapter raises {py:exc}`BackendError <batcher.BackendError>` with the
endpoint and the last error.

Triton and custom adapters retry through `serving_udf` with jittered backoff, and a shape or schema
error is never retried. Nothing is dropped silently. A failure propagates up through the stage.

## From batch to online serving

`serve_deployment` wraps a load-once factory as a Ray Serve deployment that answers per-request
calls, coalescing concurrent requests with Serve's native batching. A model proven in a batch
pipeline then serves online with no second execution engine to maintain. It needs
`batcher-engine[serve]`.

The table lists its arguments:

| Argument | Meaning |
| --- | --- |
| `build` | Zero-arg callable returning the predictor (`list[input] -> list[output]`); called once per replica. |
| `name` | Deployment name (default `"batcher-model"`). |
| `max_batch_size` | Max requests coalesced into one predictor call (default 16). |
| `batch_wait_timeout_s` | How long Serve waits to fill a batch before flushing (default 0.01s). |
| `**deployment_options` | Forwarded to `@serve.deployment` (e.g. `num_replicas`, `ray_actor_options`, `autoscaling_config`). |

The `build` factory returns a *batched* predictor. It receives the list of requests Serve
coalesced, up to `max_batch_size` or whatever arrived within `batch_wait_timeout_s`, and runs one
forward pass for the whole list, so the GPU sees a real batch even under per-request traffic. Tune
the two together: a bigger batch and a longer wait trade a little latency for throughput.

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

- {doc}`Model serving patterns </ml/training/model-serving-patterns>`: in-process against served, `run_pipeline`, and `InferencePool`.
- {doc}`Inference </ml/inference/inference>`: in-process batch inference and the `.ml` accessor.
- {doc}`Batch scoring </ml/inference/batch-scoring>`: the offline scoring job end to end.
- {doc}`GPU scheduling </ml/inference/gpu>`: `num_gpus` and `concurrency` for GPU stages.
- {doc}`LLM inference </ml/retrieval/llm/index>`: `http_engine` for OpenAI-compatible endpoints.
- {doc}`ML API </api/models/ml>`: the adapter and `ServingClient` reference.
