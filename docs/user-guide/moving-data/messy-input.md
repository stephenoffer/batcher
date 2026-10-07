# Read messy input

This page covers reading a corpus that contains files, rows or values that will not read, and how Batcher keeps each kind of failure from quietly deleting data. It builds on {doc}`reading-data`.

Real corpora contain members that will not read. Batcher separates three failures that look
alike and have different fixes, so reaching for the wrong flag cannot quietly delete data.

An **unreadable file** is one whose bytes the format cannot parse at all: a truncated
upload, a zero-byte object, a JPEG whose trailer never arrived. `on_error="skip"` drops the
file and reads the rest. Use it when the input is a corpus you do not control.

The source object behind the read keeps the audit trail. `corrupt_files()` names every
path it dropped, so a short result is explainable rather than mysterious:

```python
import os
import tempfile

import batcher as bt
from batcher.io import ParquetSource

corpus = tempfile.mkdtemp()
bt.from_pydict({"id": [1, 2]}).write.parquet(os.path.join(corpus, "good.parquet"))
with open(os.path.join(corpus, "zbad.parquet"), "wb") as f:
    _ = f.write(b"not a parquet file")

print(bt.read.parquet(corpus, on_error="skip").count())
# 2

source = ParquetSource(corpus, on_error="skip")
_ = source.read()
print([os.path.basename(path) for path in source.corrupt_files()])
# ['zbad.parquet']
```

A **malformed row** is one record inside a file that is otherwise fine: a CSV row carrying
a field the header does not have, or an NDJSON line that is not JSON at all. The file is
readable, so `on_error` is the wrong answer for it. Dropping the file would discard every
good row to be rid of one bad line. Pass `on_bad_lines` instead, which drops the record.

```python
path = os.path.join(tempfile.mkdtemp(), "events.csv")
with open(path, "w") as f:
    f.write("id,amount\n1,10\n2,20,stray\n3,30\n")

print(bt.read.csv(path, on_bad_lines="skip").to_pydict())
# {'id': [1, 3], 'amount': [10, 30]}
```

`read.json` takes the same flag, for the same reason and with the same three values.

```python
jsonl = os.path.join(tempfile.mkdtemp(), "events.jsonl")
with open(jsonl, "w") as f:
    f.write('{"id": 1}\n<html>gateway timeout</html>\n{"id": 3}\n')

print(bt.read.json(jsonl, on_bad_lines="skip").to_pydict())
# {'id': [1, 3]}
```

`on_bad_lines` takes `"error"` (the default, which refuses the read), `"warn"` (drop the
row and log it with the offending text), or `"skip"` (drop it silently). Dropped rows are
counted on the metrics export as `malformed_rows_total`, separately from the
`skipped_total` that counts whole files, because a total mixing rows with files answers
neither question.

The third failure is **a wrong encoding**: bytes that are readable, but not in the encoding
you asked for. A text corpus assembled from scrapes, exports and legacy systems is a
mixture, and a single stray byte is not a reason to lose a file. `read.text` replaces what
it cannot decode with U+FFFD by default, in both `mode="line"` and `mode="file"`.
`errors="strict"` turns it into a per-file failure that `on_error="skip"` will then drop:

```python
d = tempfile.mkdtemp()
with open(os.path.join(d, "legacy.txt"), "wb") as f:
    _ = f.write("caf\xe9\n".encode("cp1252"))

print(bt.read.text(d).to_pydict()["text"])
print(bt.read.text(d, encoding="cp1252").to_pydict()["text"])
```

Replacement is a fallback, not an answer. If you know what the bytes are, naming the
`encoding` is the fix.

Coming from another engine, the spellings map as follows.

| Their option | Batcher |
|---|---|
| Spark `mode="FAILFAST"` | `on_bad_lines="error"` (the default) |
| Spark `mode="DROPMALFORMED"` | `on_bad_lines="skip"` |
| Spark `mode="PERMISSIVE"` | no equivalent; Batcher has no corrupt-record column |
| pandas `on_bad_lines=` | the same name and the same three values |
| Polars `ignore_errors=True` | `on_bad_lines="skip"`, plus `schema=` if what you want is an unconvertible value to survive as text |

A value that will not convert to its column's type is a third thing again, and neither flag
touches it. The schema comes from the file's first block, so a column that is integral for
a million rows and then holds `"N/A"` is inference having been shown too little. Declare the
type with `schema=` rather than tolerating the row. `on_bad_lines` deliberately refuses to
delete such a record: dropping it would remove the very rows that were about to tell you
the inferred type is wrong.

## See also

- {doc}`reading-data`
- {doc}`specialized-formats`
- {doc}`/user-guide/trust/data-quality`
