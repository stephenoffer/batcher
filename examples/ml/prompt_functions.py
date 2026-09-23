"""Building prompts and reading chat logs as column expressions — all fourteen helpers.

A batch-inference job spends its first step turning rows into prompts and its last step
reading answers back out of conversations. Both are string work over every row, so they belong
in the engine: each helper here returns an `Expr` for `select` / `with_columns`, and a prompt
over a hundred million rows is built in the data plane rather than in a Python loop.

Three families, in the order a pipeline meets them:

- **assembly**: `render_template`, `wrap_tag`, `tagged_fields`, `join_context`,
  `chatml_prompt`, `instruction_prompt` build the prompt from a row's columns;
- **budget**: `prompt_token_estimate`, `fits_context`, `truncate_to_token_budget`,
  `truncate_middle` keep it inside a context window, estimating tokens from characters;
- **chat**: `conversation_turns`, `last_message`, `ends_with_role`, `render_messages` read a
  list-of-`{role, content}` conversation column.

Null handling differs by family and is asserted below: the truncations keep a null null, while
the assembly helpers render a missing field as empty text.

    python examples/ml/prompt_functions.py
"""

from __future__ import annotations

import batcher as bt


def assemble() -> None:
    rows = bt.from_pydict(
        {
            "question": ["What is the capital of France?", "Who wrote Hamlet?"],
            "passages": [["Paris is the capital.", "", "France is in Europe."], ["Shakespeare."]],
            "system": ["Answer in one word.", "Answer in one word."],
        }
    )
    built = rows.select(
        templated=bt.render_template("Q: {q}\nA:", q=bt.col("question")),
        tagged=bt.wrap_tag(bt.col("question"), "question"),
        # Empty passages are dropped from the joined context rather than leaving a blank gap.
        context=bt.join_context(bt.col("passages"), separator=" | "),
        blocks=bt.tagged_fields(question=bt.col("question"), system=bt.col("system")),
        chat=bt.chatml_prompt(bt.col("question"), system=bt.col("system")),
        alpaca=bt.instruction_prompt(bt.col("system"), bt.col("question")),
    ).to_pydict()
    print(built["chat"][0])

    assert built["templated"][1] == "Q: Who wrote Hamlet?\nA:"
    assert built["tagged"][0] == "<question>What is the capital of France?</question>"
    assert built["context"][0] == "Paris is the capital. | France is in Europe."
    assert (
        built["blocks"][1]
        == "<question>Who wrote Hamlet?</question>\n<system>Answer in one word.</system>"
    )
    assert built["chat"][0].startswith("<|im_start|>system\nAnswer in one word.<|im_end|>")
    assert built["chat"][0].endswith("<|im_start|>assistant\n")
    assert built["alpaca"][0].startswith("### Instruction:\nAnswer in one word.\n\n### Input:\n")
    assert built["alpaca"][0].endswith("### Response:\n")

    # A missing field renders as empty text inside the scaffold: check for nulls before
    # assembling if an empty question should be dropped rather than sent.
    missing = bt.from_pydict({"q": [None]}).select(
        t=bt.wrap_tag(bt.col("q").cast("string"), "question")
    )
    assert missing.to_pydict() == {"t": ["<question></question>"]}


def budget() -> None:
    docs = bt.from_pydict({"doc": ["x" * 40, "short", None], "q": ["Why?", "How?", "What?"]})
    fitted = docs.select(
        tokens=bt.prompt_token_estimate(bt.col("doc"), bt.col("q")),
        fits=bt.fits_context("doc", window=8, reserve_output=2),
        head=bt.truncate_to_token_budget("doc", budget=2),
        middle=bt.truncate_middle("doc", budget=3, marker="~"),
    ).to_pydict()
    print(fitted)

    # 40 characters at 4 per token is 10 tokens: over a window of 8 minus 2 reserved.
    assert fitted["fits"][:2] == [False, True]
    assert fitted["head"][:2] == ["x" * 8, "short"]
    # The middle cut keeps both ends around the marker and fits the 12-character budget.
    assert fitted["middle"][0] == "x" * 6 + "~" + "x" * 5
    assert fitted["middle"][1] == "short"
    # A missing document stays missing through both truncations.
    assert fitted["head"][2] is None and fitted["middle"][2] is None
    assert fitted["tokens"][0] > fitted["tokens"][1]


def read_conversations() -> None:
    logs = bt.from_pydict(
        {
            "msgs": [
                [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "2+2?"},
                    {"role": "assistant", "content": "4"},
                ],
                [
                    {"role": "user", "content": "hello?"},
                ],
            ]
        }
    )
    read = logs.select(
        turns=bt.conversation_turns("msgs"),
        replies=bt.conversation_turns("msgs", role="assistant"),
        answer=bt.last_message("msgs", role="assistant"),
        last=bt.last_message("msgs"),
        answered=bt.ends_with_role("msgs", "assistant"),
        text=bt.render_messages("msgs"),
    ).to_pydict()
    print(read["text"][0])

    assert read["turns"] == [3, 1]
    assert read["replies"] == [1, 0]
    assert read["answer"] == ["4", None]
    assert read["last"] == ["4", "hello?"]
    # The second log was never answered: a collection failure, not a short conversation.
    assert read["answered"] == [True, False]
    assert read["text"][0] == "system: Be brief.\nuser: 2+2?\nassistant: 4"


def main() -> None:
    assemble()
    budget()
    read_conversations()


if __name__ == "__main__":
    main()
