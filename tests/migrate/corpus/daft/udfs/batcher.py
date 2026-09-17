import daft
import batcher as bt


# batcher-migrate: Daft `daft.func` has no Batcher equivalent yet: @daft.func Expression-level UDFs with return_dtype=, unnest=, gpus=, max_concurrency=, use_process=, on_error=, max_retries=
@daft.func
def shout(text: str) -> str:
    return text.upper() + "!"


df = bt.from_pydict({"s": ["hi", "yo"]})
loud = df.with_columns(loud=shout(bt.col("s")))
# batcher-migrate: Daft `Expression.length` maps to Expr.str.len_chars / Expr.list.len; rewrite by hand
tokens = df.with_columns(n=bt.col("s").length())
print(loud.to_pydict(), tokens.to_pydict())
