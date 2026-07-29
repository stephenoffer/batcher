"""Did the model obey the output format you asked for?

Format compliance is the cheapest eval there is and the one that catches the most
regressions. If you asked for JSON and ``valid_json_rate`` drops to 0.7, that is a
production incident regardless of how good the prose is.

    python examples/metrics/text_formatting.py
"""

from __future__ import annotations

import batcher as bt


def main() -> None:
    outputs = bt.from_pydict(
        {
            "text": [
                '{"answer": 42}',
                '{"answer": 42',  # truncated, so not valid JSON
                "- first\n- second\n- third",
                "1. first\n2. second",
                "# Heading\n\nSome prose.",
                "The answer is 42",
                "```python\nprint(1)\n```",
                "<answer>42</answer>",
            ],
        }
    )

    compliance = outputs.select(
        json_present=bt.json_present_rate("text"),
        json_valid=bt.valid_json_rate("text"),
        bullets=bt.bullet_list_rate("text"),
        numbered=bt.numbered_list_rate("text"),
        headings=bt.heading_rate("text"),
        code_blocks=bt.code_block_present_rate("text"),
        tables=bt.table_rate("text"),
        md_links=bt.markdown_link_rate("text"),
        numeric_answer=bt.numeric_answer_rate("text"),
        boxed=bt.boxed_answer_rate("text"),
        choice=bt.choice_answer_rate("text"),
        tagged=bt.tagged_answer_rate("text", tag="answer"),
    ).to_pydict()

    print(compliance)

    for name, value in compliance.items():
        assert 0.0 <= value[0] <= 1.0, name

    # Only the well-formed object counts as JSON: the truncated one is neither
    # "present" nor valid, so both rates land on 1 of 8.
    assert compliance["json_present"][0] == 0.125
    assert compliance["json_valid"][0] == 0.125
    assert compliance["bullets"][0] == 0.125
    assert compliance["numbered"][0] == 0.125
    assert compliance["headings"][0] == 0.125
    assert compliance["code_blocks"][0] == 0.125
    assert compliance["tagged"][0] == 0.125

    # The gate this exists for: refuse to ship if JSON compliance slipped.
    assert compliance["json_valid"][0] < 0.9  # this sample run would fail the gate


if __name__ == "__main__":
    main()
