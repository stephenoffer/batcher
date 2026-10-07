"""Real-tokenizer token counts, shared by the token-count column and LLM request batching.

The rest of the control plane estimates tokens from characters (`sizing.estimate_tokens`,
`plan.functions.prompt.budget`), which needs no model and is deliberately approximate. This
module is the exact counterpart, for when the approximation is not good enough: the count
comes from the tokenizer the model itself uses.

Two callers need that count and must agree on it, which is the reason they share this
module rather than each loading a tokenizer their own way:

* `ds.ml.token_count` writes the count as a column, so a budget filter
  (``ds.filter(bt.col("n") <= 4096)``) can drop what will not fit before any request is made.
* `TokenBudget` groups an engine's requests so the tokens in flight together stay under a
  budget, and cuts or refuses a prompt that would not fit even alone.

Both resolve the tokenizer with `resolve_tokenizer` and count with `encode_texts`, under the
same `add_special_tokens` policy, so a row the filter kept is a row the batcher counts the
same way. The work is batch-level: a HuggingFace fast tokenizer is called once per Arrow
batch over the whole list of texts, which is where its Rust fast path lives, and the
tokenizer is loaded once per worker by the class UDF or engine that holds it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["TokenBudget", "encode_texts", "resolve_tokenizer", "token_count_udf"]

#: How a group of requests is charged against `TokenBudget.max_batch_tokens`.
_PADDING = ("none", "longest")


def resolve_tokenizer(tokenizer: Any) -> Any:
    """The tokenizer object `tokenizer` names: a model id or path is loaded, anything else kept.

    A string is loaded with ``transformers.AutoTokenizer.from_pretrained``, which reads a
    local directory as readily as a Hub id. Call this where the tokenizer is *used* (a
    worker's UDF ``__init__``, an engine factory), so the vocabulary loads once per worker
    rather than travelling in the plan.

    Args:
        tokenizer: A HuggingFace model id or local path, a tokenizer object (anything with
            ``.encode``, or HuggingFace's ``tokenizers.Tokenizer`` with ``.encode_batch``),
            or a ``str -> list`` callable.

    Returns:
        A tokenizer `encode_texts` can drive.

    Raises:
        PlanError: If `tokenizer` is none of those shapes.
    """
    if isinstance(tokenizer, str):
        from batcher._internal.optional import require

        auto = require(
            "transformers",
            "AutoTokenizer",
            feature="Loading a tokenizer by name",
            provides="transformers",
            extra="transformers",
        )
        return auto.from_pretrained(tokenizer)
    if callable(tokenizer) or hasattr(tokenizer, "encode") or hasattr(tokenizer, "encode_batch"):
        return tokenizer
    raise PlanError(
        f"tokenizer must be a model id or path, a tokenizer object with .encode, or a "
        f"str -> list callable; got {type(tokenizer).__name__}"
    )


def check_tokenizer_spec(tokenizer: Any, *, method: str) -> None:
    """Refuse a tokenizer argument no worker could use, at plan time.

    Raises:
        PlanError: If `tokenizer` is not a string, a tokenizer object, or a callable.
    """
    if isinstance(tokenizer, str) and tokenizer:
        return
    if callable(tokenizer) or hasattr(tokenizer, "encode") or hasattr(tokenizer, "encode_batch"):
        return
    raise PlanError(
        f"{method}(tokenizer=...) must be a HuggingFace model id or local path, a tokenizer "
        f"object with .encode, or a str -> list callable; got {tokenizer!r}"
    )


def _special_kwargs(add_special_tokens: bool | None) -> dict[str, bool]:
    """The keyword a tokenizer call receives: nothing for `None`, so its own default holds."""
    return {} if add_special_tokens is None else {"add_special_tokens": add_special_tokens}


def encode_texts(
    tokenizer: Any, texts: list[str], *, add_special_tokens: bool | None = None
) -> list[list[int]]:
    """Token ids for each text, from one batched call where the tokenizer offers one.

    The three tokenizer shapes are driven the way each is meant to be, which is the same
    discrimination `batcher.ml.preprocessors.Tokenizer` makes:

    * a HuggingFace ``transformers`` tokenizer (callable *and* carrying ``.encode``) is
      called once over the whole list;
    * a ``tokenizers.Tokenizer`` is driven through ``.encode_batch``;
    * anything else with ``.encode``, or a plain ``str -> list`` callable, is applied per
      string, because that is the only call it has.

    Args:
        tokenizer: A tokenizer `resolve_tokenizer` returned.
        texts: The texts to encode. None of them may be null.
        add_special_tokens: Whether the count includes the tokenizer's special tokens (a
            BOS, an EOS, a ``[CLS]``). `None` leaves the tokenizer's own default, which is
            what a tokenizer that takes no such argument gets regardless.

    Returns:
        One list of token ids (or tokens, for a ``str -> list`` callable) per text.

    Examples:
        .. doctest::

            >>> from batcher.ml.llm.tokens import encode_texts
            >>> encode_texts(str.split, ["a b c", "d"])
            [['a', 'b', 'c'], ['d']]
    """
    if not texts:
        return []
    kwargs = _special_kwargs(add_special_tokens)
    if callable(tokenizer) and hasattr(tokenizer, "encode"):
        encoded = tokenizer(list(texts), **kwargs)
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return [list(row) for row in ids]
    if hasattr(tokenizer, "encode_batch"):
        return [list(e.ids) for e in tokenizer.encode_batch(list(texts), **kwargs)]
    encode = getattr(tokenizer, "encode", None)
    if encode is not None:
        return [list(encode(text, **kwargs)) for text in texts]
    return [list(tokenizer(text)) for text in texts]


def token_count_udf(
    column: str,
    tokenizer: Any,
    *,
    output_column: str,
    add_special_tokens: bool | None = None,
) -> type:
    """A **load-once class UDF** appending each text's token count as an ``int64`` column.

    The tokenizer is resolved in ``__init__``, so `map_batches` loads it once per worker and
    reuses it for every batch, and each batch is one tokenizer call. A null text stays null.

    Args:
        column: The text column to count.
        tokenizer: What `resolve_tokenizer` accepts.
        output_column: The name of the appended count column.
        add_special_tokens: The special-token policy `encode_texts` applies.

    Returns:
        A class whose instances map a `pyarrow.RecordBatch` to the batch plus the counts.
    """

    class _TokenCount:
        """Holds one tokenizer for the worker's lifetime; called once per batch."""

        def __init__(self) -> None:
            self._tokenizer = resolve_tokenizer(tokenizer)

        def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
            import pyarrow as pa

            from batcher.ml.tabular.features import append_columns

            texts = batch.column(column).to_pylist()
            present = [i for i, text in enumerate(texts) if text is not None]
            counts: list[int | None] = [None] * len(texts)
            encoded = encode_texts(
                self._tokenizer,
                [str(texts[i]) for i in present],
                add_special_tokens=add_special_tokens,
            )
            for i, ids in zip(present, encoded, strict=True):
                counts[i] = len(ids)
            return append_columns(batch, {output_column: pa.array(counts, type=pa.int64())})

    return _TokenCount


@dataclass(frozen=True)
class TokenBudget:
    """A per-batch token budget an LLM engine honours while it sends a batch of requests.

    An engine that calls a served model sends a batch's requests concurrently. On ragged
    text that is a batch of 64 short prompts one moment and 64 long documents the next, and
    a server's KV cache, a gateway's request-size limit, or a provider's tokens-in-flight
    ceiling sees only the second. A `TokenBudget` makes the engine split each batch into
    consecutive groups whose token cost stays under `max_batch_tokens`, and send one group
    at a time. It is applied *inside* the engine's own request loop, so there is no second
    scheduler deciding what the engine sends.

    Counts come from a real tokenizer, through the same `encode_texts` call
    `ds.ml.token_count` uses, so a corpus filtered on that column is counted the same way
    here. Only the prompt is counted; the reply is bounded separately by ``max_tokens``.

    A prompt larger than the whole budget is handled by `oversized`, with the same
    ``"head"``/``"tail"``/``"error"`` vocabulary as ``vllm_engine(truncation=...)``: cut to
    the budget keeping its first or last tokens, with a warning, or refused with
    `DataQualityError` before anything is sent.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> budget = bt.ml.TokenBudget(str.split, max_batch_tokens=4)
            >>> budget.groups([3, 1, 2, 2])
            [[0, 1], [2, 3]]

    Args:
        tokenizer: A HuggingFace model id or local path (loaded once per worker), a
            tokenizer object, or a ``str -> list`` callable.
        max_batch_tokens: The most prompt tokens one group of requests may carry.
        padding: ``"none"`` (default) charges a group the sum of its prompts' tokens, which
            fits a server that packs sequences; ``"longest"`` charges the longest prompt
            times the group size, which fits a server that pads every sequence in a step.
        oversized: What to do with one prompt over `max_batch_tokens`: ``"head"`` (default)
            keeps its first tokens, ``"tail"`` its last, ``"error"`` raises.
        add_special_tokens: Whether counts include the tokenizer's special tokens. `None`
            keeps the tokenizer's own default.

    Raises:
        PlanError: If `max_batch_tokens` is not a positive integer, or `padding`,
            `oversized` or `tokenizer` is not one of the accepted values.
    """

    tokenizer: Any
    max_batch_tokens: int
    padding: str = "none"
    oversized: str = "head"
    add_special_tokens: bool | None = None

    def __post_init__(self) -> None:
        from batcher.ml.llm.sizing import check_truncation

        check_tokenizer_spec(self.tokenizer, method="TokenBudget")
        if not isinstance(self.max_batch_tokens, int) or self.max_batch_tokens < 1:
            raise PlanError(
                f"TokenBudget(max_batch_tokens=...) must be a positive integer, "
                f"got {self.max_batch_tokens!r}"
            )
        if self.padding not in _PADDING:
            raise PlanError(
                f"TokenBudget(padding=...) must be one of {list(_PADDING)}, got {self.padding!r}"
            )
        check_truncation(self.oversized)

    def groups(self, counts: list[int]) -> list[list[int]]:
        """Split request positions into consecutive groups that each fit the budget.

        Consecutive, so the engine's dispatch order (longest prompt first, from
        `ds.ml.generate`) is kept and the results line up with the requests without a
        permutation. A single request is always a group of its own even when it alone
        exceeds the budget; `oversized` has already decided what happens to it.

        Args:
            counts: Each request's prompt tokens, in dispatch order.

        Returns:
            Lists of positions into `counts`, in order, covering every position once.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> bt.ml.TokenBudget(str.split, 6, padding="longest").groups([3, 2, 2, 1])
                [[0, 1], [2, 3]]
        """
        out: list[list[int]] = []
        current: list[int] = []
        total = longest = 0
        for position, count in enumerate(counts):
            n = max(0, int(count))
            if current:
                if self.padding == "longest":
                    cost = max(longest, n) * (len(current) + 1)
                else:
                    cost = total + n
                if cost > self.max_batch_tokens:
                    out.append(current)
                    current, total, longest = [], 0, 0
            current.append(position)
            total += n
            longest = max(longest, n)
        if current:
            out.append(current)
        return out


class BudgetedBatches:
    """A worker's live `TokenBudget`: the loaded tokenizer, and the per-batch planning.

    Built lazily by the engine on its first batch, so the tokenizer loads on the worker that
    uses it and never on the driver.
    """

    __slots__ = ("_budget", "_tokenizer")

    def __init__(self, budget: TokenBudget) -> None:
        self._budget = budget
        self._tokenizer = resolve_tokenizer(budget.tokenizer)

    def plan(self, requests: list) -> tuple[list, list[list[int]]]:
        """Fit each request to the budget and group them: ``(requests, groups)``.

        Args:
            requests: The engine's requests, each a prompt string or a ``{"prompt": ...}``
                dict, in dispatch order.

        Returns:
            The requests with any oversized prompt cut (or the call refused), and the
            consecutive groups to send one after another.

        Raises:
            DataQualityError: If a prompt exceeds the budget under ``oversized="error"``.
        """
        from batcher.ml.llm.sizing import _truncate_to_window, prompt_text

        budget = self._budget
        texts = [prompt_text(r) for r in requests]
        ids = encode_texts(self._tokenizer, texts, add_special_tokens=budget.add_special_tokens)
        counts = [len(row) for row in ids]
        if any(n > budget.max_batch_tokens for n in counts):
            if budget.oversized != "error" and not hasattr(self._tokenizer, "decode"):
                raise PlanError(
                    f"TokenBudget(oversized={budget.oversized!r}) cuts an over-budget prompt "
                    "back to text, which needs a tokenizer with .decode; this one has none. "
                    "Pass a tokenizer object (or a model id), or use oversized='error'."
                )
            fitted = _truncate_to_window(
                texts,
                self._tokenizer,
                budget.max_batch_tokens,
                policy=budget.oversized,
                setting="TokenBudget(oversized=...)",
                add_special_tokens=budget.add_special_tokens,
                encoded=ids,
            )
            requests = [
                {**r, "prompt": t} if isinstance(r, dict) else t
                for r, t in zip(requests, fitted, strict=True)
            ]
            counts = [min(n, budget.max_batch_tokens) for n in counts]
        return requests, budget.groups(counts)
