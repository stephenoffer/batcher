import daft


# batcher-migrate: Daft `daft.func` has no Batcher equivalent yet: @daft.func Expression-level UDFs with return_dtype=, unnest=, gpus=, max_concurrency=, use_process=, on_error=, max_retries=
@daft.func
def shout(text: str) -> str:
    return text.upper() + "!"


df = daft.from_pydict({"s": ["hi", "yo"]})
# batcher-migrate: Daft `Dataset.with_columns` has no exact Daft spelling; left as written
loud = df.with_columns(loud=shout(daft.col("s")))
# batcher-migrate: Daft `Expression.length` maps to Expr.str.len_chars / Expr.list.len; rewrite by hand
# batcher-migrate: Daft `Expr.length` has no exact Daft spelling; left as written
# batcher-migrate: Daft `Dataset.with_columns` has no exact Daft spelling; left as written
tokens = df.with_columns(n=daft.col("s").length())
print(loud.to_pydict(), tokens.to_pydict())
