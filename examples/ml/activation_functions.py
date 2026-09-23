"""Apply a model's activation to raw scores without leaving the engine.

A scoring pipeline usually ends with a column of raw logits and a Python loop that maps an
activation over them. That loop is the part that does not scale: it pulls every row into
the driver to compute something the engine can do per batch in Rust.

Every activation a tabular or small MLP head needs is an expression here, so the transform
stays in the plan and runs wherever the plan runs. The shapes are the ones PyTorch uses, so
a threshold tuned against `torch.nn.functional` transfers unchanged.

    python examples/ml/activation_functions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    scores = bt.from_pydict({"logit": [-2.0, -0.5, 0.0, 0.5, 2.0]})

    # The saturating family. `hardsigmoid` and `hardtanh` are the piecewise-linear
    # approximations mobile models ship instead of the exponential forms: same shape,
    # no `exp` per row.
    shaped = scores.select(
        x=col("logit"),
        elu=col("logit").elu(),
        gelu=col("logit").gelu(),
        silu=col("logit").silu(),
        mish=col("logit").mish(),
        leaky=col("logit").leaky_relu(),
        hard_sig=col("logit").hardsigmoid(),
        hard_tanh=col("logit").hardtanh(),
        hard_swish=col("logit").hardswish(),
        softsign=col("logit").softsign(),
        tanhshrink=col("logit").tanhshrink(),
    ).to_pydict()

    # Every activation here is zero-preserving except `hardsigmoid`, which is centred on
    # 0.5 by construction — that is the one to check when a pipeline's "inactive" rows
    # stop looking inactive.
    at_zero = 2
    for name in ("elu", "gelu", "silu", "mish", "leaky", "hard_tanh", "softsign", "tanhshrink"):
        assert shaped[name][at_zero] == 0.0, f"{name} moved zero to {shaped[name][at_zero]}"
    assert shaped["hard_sig"][at_zero] == 0.5

    # `leaky_relu` passes a positive value through and scales a negative one, which is the
    # whole point of it over `relu`: the gradient at -2.0 is not lost.
    assert shaped["leaky"][4] == 2.0
    assert -2.0 < shaped["leaky"][0] < 0.0

    # The saturating pair are bounded, so an outlier logit cannot dominate a downstream sum.
    assert 0.0 <= shaped["hard_sig"][0] <= 1.0
    assert -1.0 <= shaped["hard_tanh"][4] <= 1.0
    assert -1.0 < shaped["softsign"][0] < 0.0

    # `hardswish` is `x * hardsigmoid(x)`, which is what makes it cheap; check the identity
    # rather than a literal, so this keeps testing the relationship if the constants move.
    for i, x in enumerate(shaped["x"]):
        assert abs(shaped["hard_swish"][i] - x * shaped["hard_sig"][i]) < 1e-9

    print("activations over", len(shaped["x"]), "logits")
    print("  gelu(2.0)  =", round(shaped["gelu"][4], 6))
    print("  silu(2.0)  =", round(shaped["silu"][4], 6))
    print("  mish(2.0)  =", round(shaped["mish"][4], 6))


if __name__ == "__main__":
    main()
