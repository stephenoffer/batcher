import daft


@daft.func
def shout(text: str) -> str:
    return text.upper() + "!"


df = daft.from_pydict({"s": ["hi", "yo"]})
loud = df.with_column("loud", shout(daft.col("s")))
tokens = df.with_column("n", daft.col("s").length())
print(loud.to_pydict(), tokens.to_pydict())
