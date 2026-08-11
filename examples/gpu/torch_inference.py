"""Batch inference with a torch model, on whatever device is available.

The model moves to the resolved device and the batches follow it. Nothing else in the
pipeline changes, which is the property worth preserving: a script that only runs on a
GPU cannot be tested anywhere else, and an untested GPU path is where the bugs live.

    python examples/gpu/torch_inference.py
    python examples/gpu/torch_inference.py --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa

from _common import torch_device, tpch


def main() -> None:
    try:
        import torch
    except ImportError:
        print("torch is not installed; install batcher-engine[torch] to run this.")
        return

    device = torch_device()
    print("torch device:", device)

    # A stand-in for a real model: linear, deterministic, and small enough to assert on.
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
        model.bias.copy_(torch.tensor([0.5]))
    model = model.to(device).eval()

    lineitem = tpch("lineitem").select("l_quantity", "l_discount").head(50_000)

    def score(batch: pa.RecordBatch) -> pa.RecordBatch:
        """Score one Arrow batch. Batch-first: never one row at a time."""
        features = torch.tensor(
            [batch.column("l_quantity").to_pylist(), batch.column("l_discount").to_pylist()],
            dtype=torch.float32,
            device=device,
        ).T
        with torch.no_grad():
            predictions = model(features).squeeze(-1).cpu().numpy()
        return pa.RecordBatch.from_arrays(
            [*batch.columns, pa.array(predictions)],
            names=[*batch.schema.names, "score"],
        )

    scored = lineitem.map_batches(score)
    result = scored.head(5).to_pydict()
    print(result)

    # The new column exists in the *result*. `Dataset.columns` still reports the
    # input schema, because a Python callback's output schema is not known until it
    # runs — which is the cost of dropping out of the expression language.
    assert "score" in result
    assert scored.count() == lineitem.count()

    # The model is 2*q - d + 0.5, so the output is checkable by hand.
    expected = [
        2.0 * quantity - discount + 0.5
        for quantity, discount in zip(result["l_quantity"], result["l_discount"], strict=True)
    ]
    assert all(
        abs(left - right) < 1e-3 for left, right in zip(result["score"], expected, strict=True)
    )


if __name__ == "__main__":
    main()
