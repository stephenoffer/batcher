"""Whether a model expects its prompts wrapped in a chat template.

An instruction-tuned model was trained on conversations rendered through a specific
template (``<|im_start|>user`` … or whatever its family uses). Send it a bare completion
prompt instead and it still answers — in a format it was never tuned on. Nothing raises,
no token budget is exceeded, and the only symptom is output that is quietly worse.

That failure is invisible from the data plane, but it is *visible from the tokenizer*: a
model shipped for instruction following carries a `chat_template`, and a base model does
not. So the mismatch can be detected rather than left to the user to remember.

The check fires only when the caller never said which mode they wanted. Asking for raw
completions from an instruct model is a legitimate thing to do — constrained-choice
classification does exactly that — so an explicit `chat=False` is a decision, not a
mistake, and gets no advice.
"""

from __future__ import annotations

__all__ = ["warn_if_chat_template_unused"]

#: Warned once per process. The engine is built once per worker, but a session that builds
#: several would otherwise repeat one message per engine.
_TEMPLATE_WARNED = False


def has_chat_template(tokenizer: object | None) -> bool:
    """Whether `tokenizer` carries a chat template (i.e. the model is instruction-tuned).

    Args:
        tokenizer: The model's tokenizer, or `None` when one is not reachable.

    Returns:
        True when a non-empty chat template is present.
    """
    if tokenizer is None:
        return False
    try:
        return bool(getattr(tokenizer, "chat_template", None))
    except Exception:  # pragma: no cover - tokenizer implementations vary
        return False


def warn_if_chat_template_unused(tokenizer: object | None, model: str) -> None:
    """Warn once when an instruction-tuned model is being sent un-templated prompts.

    Called only when the caller left `chat` unset, so this never second-guesses a
    deliberate choice. The remedy is one keyword, which is why it is worth saying: the
    alternative is a whole batch-inference run whose output is subtly off with nothing in
    the logs to explain it.

    Args:
        tokenizer: The worker's tokenizer, used to detect the template.
        model: The model id, so the message names what it is talking about.
    """
    global _TEMPLATE_WARNED
    if _TEMPLATE_WARNED or not has_chat_template(tokenizer):
        return
    _TEMPLATE_WARNED = True
    import warnings

    from batcher._internal.errors import PerformanceWarning

    warnings.warn(
        f"{model!r} ships a chat template, so it was instruction-tuned on conversations, "
        f"but prompts are being sent as raw completions. The model will still answer, in a "
        f"format it was never tuned on — degraded output with nothing to signal it. Pass "
        f"chat=True to apply the model's own template, or chat=False to silence this if "
        f"raw completions are what you want (constrained-choice classification, for one).",
        PerformanceWarning,
        stacklevel=3,
    )
