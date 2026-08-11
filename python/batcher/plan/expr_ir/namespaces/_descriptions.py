"""The curated per-accessor docstrings, keyed by accessor name.

Data, not code: `_bind` prefers an entry here over its generic per-family template, so each
generated accessor reads like a hand-written method. They live in their own module because
they are the bulk of the accessor machinery by line count and none of it by logic — keeping
them beside the binder pushed that file past its size limit while saying nothing about how
binding works.

Where a name is reused across families with the same meaning (``reverse``, ``min``/``max`` on
lists), one entry covers it; the per-family fallback handles anything absent. Every example
here is executed by the doctest builder, so an entry that lies fails the docs build — but the
*prose* is verified against the engine by hand, so do not edit one without re-checking the
behaviour it claims.
"""

from __future__ import annotations

__all__ = ["_DESCRIPTIONS"]


# Curated per-accessor docstrings, keyed by accessor name. The factory prefers an
# entry here over the generic per-family ``doc(name)`` template, so each bound
# accessor reads like a hand-written method. Where a name is reused across families
# with the same meaning (``reverse``, ``min``/``max`` on lists), one entry covers it;
# the per-family fallback handles anything absent. Verified against the engine —
# do not edit without re-checking behavior.
_DESCRIPTIONS: dict[str, str] = {
    # --- .str string→string transforms (null → null) ------------------------
    "upper": (
        "The string with every letter uppercased.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"s": ["Hello"]})\n'
        '        >>> ds.select(r=bt.col("s").str.upper()).to_pydict()\n'
        "        {'r': ['HELLO']}"
    ),
    "lower": (
        "The string with every letter lowercased.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"s": ["Hello"]})\n'
        '        >>> ds.select(r=bt.col("s").str.lower()).to_pydict()\n'
        "        {'r': ['hello']}"
    ),
    # --- .dt date/time field extraction (all → Int64 unless noted) ----------
    "year": (
        "The year component, e.g. 2021.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.year()).to_pydict()\n'
        "        {'r': [2024]}"
    ),
    "month": (
        "The month-of-year component, 1-12.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.month()).to_pydict()\n'
        "        {'r': [2]}"
    ),
    "day": (
        "The day-of-month component, 1-31.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.day()).to_pydict()\n'
        "        {'r': [15]}"
    ),
    "hour": (
        "The hour-of-day component, 0-23.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.hour()).to_pydict()\n'
        "        {'r': [13]}"
    ),
    "minute": (
        "The minute-of-hour component, 0-59.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.minute()).to_pydict()\n'
        "        {'r': [45]}"
    ),
    "second": (
        "The second-of-minute component, 0-59.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.second()).to_pydict()\n'
        "        {'r': [30]}"
    ),
    "quarter": (
        "The calendar quarter, 1-4.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.quarter()).to_pydict()\n'
        "        {'r': [1]}"
    ),
    "week": (
        "The ISO 8601 week number, 1-53.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.week()).to_pydict()\n'
        "        {'r': [7]}"
    ),
    "dayofweek": (
        "The day of week, Sunday = 0 through Saturday = 6.\n\n"
        "For ISO numbering use ``isodow``; they differ only on Sunday, as the example is.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 18, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.dayofweek()).to_pydict()  # a Sunday\n'
        "        {'r': [0]}"
    ),
    "dayofyear": (
        "The day of year, 1 through 366.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.dayofyear()).to_pydict()\n'
        "        {'r': [46]}"
    ),
    "epoch": (
        "Seconds since the Unix epoch, 1970-01-01 00:00:00 UTC (→ Int64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2021, 3, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(bt.col("d").dt.epoch().alias("r")).to_pydict()\n'
        "        {'r': [1615815930]}"
    ),
    "dayname": (
        'The full English weekday name, e.g. "Monday" (→ Utf8).\n\n'
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.dayname()).to_pydict()\n'
        "        {'r': ['Thursday']}"
    ),
    "monthname": (
        'The full English month name, e.g. "January" (→ Utf8).\n\n'
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.monthname()).to_pydict()\n'
        "        {'r': ['February']}"
    ),
    "isodow": (
        "The ISO day of week, Monday = 1 through Sunday = 7.\n\n"
        "For the DuckDB numbering (Sunday = 0 through Saturday = 6) use ``dayofweek``.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 18, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.isodow()).to_pydict()  # a Sunday\n'
        "        {'r': [7]}"
    ),
    "century": (
        "The century, e.g. 2021 → 21.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.century()).to_pydict()\n'
        "        {'r': [21]}"
    ),
    "decade": (
        "The decade, e.g. 2021 → 202.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.decade()).to_pydict()\n'
        "        {'r': [202]}"
    ),
    "millennium": (
        "The millennium, e.g. 2021 → 3.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.millennium()).to_pydict()\n'
        "        {'r': [3]}"
    ),
    "last_day": (
        "The last day of the instant's month (→ Date).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        "        >>> import datetime as dt\n"
        '        >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})\n'
        '        >>> ds.select(r=bt.col("d").dt.last_day()).to_pydict()\n'
        "        {'r': [datetime.date(2024, 2, 29)]}"
    ),
    # --- .list per-row reductions over each list value ----------------------
    # The reductions return null on an empty or null list; len/n_unique return 0
    # for an empty list and null for a null list.
    "len": (
        "The number of elements in each list (→ Int64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[3, 1, 2]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.len()).to_pydict()\n'
        "        {'r': [3]}"
    ),
    "sum": (
        "The sum of the elements of each list.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1, 2, 3]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.sum()).to_pydict()\n'
        "        {'r': [6]}"
    ),
    "min": (
        "The smallest element of each list.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[3, 1, 2]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.min()).to_pydict()\n'
        "        {'r': [1]}"
    ),
    "max": (
        "The largest element of each list.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[3, 1, 2]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.max()).to_pydict()\n'
        "        {'r': [3]}"
    ),
    "mean": (
        "The arithmetic mean of the elements of each list (→ Float64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1, 2, 3]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.mean()).to_pydict()\n'
        "        {'r': [2.0]}"
    ),
    "n_unique": (
        "The count of distinct elements in each list (→ Int64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1, 2, 2, 3]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.n_unique()).to_pydict()\n'
        "        {'r': [3]}"
    ),
    "sort": (
        "Each list sorted ascending (→ list).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[3, 1, 2]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.sort()).to_pydict()\n'
        "        {'r': [[1, 2, 3]]}"
    ),
    "sort_desc": (
        "Each list sorted descending, nulls last (\u2192 list).\n\n"
        "Not the reverse of :meth:`sort`: ascending puts nulls last, so reversing would\n"
        "move them to the front. DuckDB's ``list_reverse_sort`` leaves them at the back.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[3, 1, None, 2]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.sort_desc()).to_pydict()\n'
        "        {'r': [[3, 2, 1, None]]}"
    ),
    "product": (
        "The product of the elements of each list.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1, 2, 3, 4]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.product()).to_pydict()\n'
        "        {'r': [24.0]}"
    ),
    "std": (
        "The sample standard deviation of the elements of each list (→ Float64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1, 2, 3]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.std()).to_pydict()\n'
        "        {'r': [1.0]}"
    ),
    "var": (
        "The sample variance of the elements of each list (→ Float64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1, 2, 3]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.var()).to_pydict()\n'
        "        {'r': [1.0]}"
    ),
    "unique": (
        "The distinct elements of each list, first-seen order preserved (→ list).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1, 2, 2, 3]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.unique()).to_pydict()\n'
        "        {'r': [[1, 2, 3]]}"
    ),
    "median": (
        "The median of the elements of each list (→ Float64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1, 2, 3, 4]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.median()).to_pydict()\n'
        "        {'r': [2.5]}"
    ),
    "arg_min": (
        "The 0-based index of the smallest element of each list (→ Int64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[3, 1, 2]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.arg_min()).to_pydict()\n'
        "        {'r': [1]}"
    ),
    "arg_max": (
        "The 0-based index of the largest element of each list (→ Int64).\n\n"
        "Ties take the first occurrence.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"a": [[3.0, 1.0, 4.0, 1.0]]})\n'
        '        >>> ds.select(bt.col("a").list.arg_max().alias("r")).to_pydict()\n'
        "        {'r': [2]}"
    ),
    "arg_sort": (
        "The 0-based indices that sort each list ascending (→ list of Int64).\n\n"
        "Stable (ties keep original order); `reverse` it for a descending / top-k ranking "
        "of a per-row score or logit vector.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"scores": [[0.3, 0.9, 0.1]]})\n'
        '        >>> ds.select(r=bt.col("scores").list.arg_sort()).to_pydict()\n'
        "        {'r': [[2, 0, 1]]}"
    ),
    "l2_norm": (
        "The Euclidean norm, sqrt(sum of squares), of each list (→ Float64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[3.0, 4.0]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.l2_norm()).to_pydict()\n'
        "        {'r': [5.0]}"
    ),
    "l1_norm": (
        "The Manhattan (L1) norm, sum of absolute values, of each list (→ Float64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[3.0, -4.0]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.l1_norm()).to_pydict()\n'
        "        {'r': [7.0]}"
    ),
    "max_abs": (
        "The maximum absolute value of each list — the MaxAbs scaling divisor (→ Float64).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1.0, -5.0, 3.0]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.max_abs()).to_pydict()\n'
        "        {'r': [5.0]}"
    ),
    "normalize": (
        "Each list L2-normalized to unit length (→ list); embedding prep.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[3.0, 4.0]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.normalize()).to_pydict()\n'
        "        {'r': [[0.6, 0.8]]}"
    ),
    "cum_sum": (
        "The cumulative sum of each list — element ``i`` sums ``0..=i`` (→ same-length list).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1.0, 2.0, 3.0]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.cum_sum()).to_pydict()\n'
        "        {'r': [[1.0, 3.0, 6.0]]}"
    ),
    "diff": (
        "The first difference of each list (→ list of the same length).\n\n"
        "Element ``i`` is ``xᵢ - xᵢ₋₁`` and element 0 is null (no predecessor); the "
        "delta-feature building block for audio (MFCC deltas) and time-series.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"xs": [[1.0, 2.0, 4.0, 7.0]]})\n'
        '        >>> ds.select(r=bt.col("xs").list.diff()).to_pydict()\n'
        "        {'r': [[None, 1.0, 2.0, 3.0]]}"
    ),
    "softmax": (
        "Softmax over each list — logits to a probability distribution summing to 1 (→ list).\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"logits": [[0.0, 0.0]]})\n'
        '        >>> ds.select(p=bt.col("logits").list.softmax()).to_pydict()\n'
        "        {'p': [[0.5, 0.5]]}"
    ),
    "log_softmax": (
        "Log-softmax over each list — the log-domain distribution, in nats (\u2192 list).\n\n"
        "Not the same as taking the log of :meth:`softmax`: a probability small enough to\n"
        "underflow to 0 there becomes ``-inf``, while here it stays a large negative finite\n"
        "number. That is the reason a scoring or training pipeline carries log-probabilities\n"
        "at all, so the conversion has to happen in the log domain to be worth anything.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"logits": [[0.0, 0.0]]})\n'
        "        >>> [round(v, 4) for v in ds.select(\n"
        '        ...     p=bt.col("logits").list.log_softmax()\n'
        '        ... ).to_pydict()["p"][0]]\n'
        "        [-0.6931, -0.6931]"
    ),
    "entropy": (
        "Shannon entropy of each list read as a distribution, in nats (\u2192 Float64).\n\n"
        "The row is normalized by its own sum first, so it works on a probability vector, a\n"
        "count vector, or unnormalized weights alike. It is 0 when all the mass sits on one\n"
        "outcome and ``ln n`` when it is spread evenly over ``n`` \u2014 the per-row uncertainty\n"
        "of a classifier\u2019s output, a retrieval score distribution, or an attention row, and\n"
        "the signal to route a low-confidence row to a larger model or to a human. A row\n"
        "totalling zero has no distribution to measure and yields null.\n\n"
        "Examples:\n"
        "    .. doctest::\n\n"
        "        >>> import batcher as bt\n"
        '        >>> ds = bt.from_pydict({"p": [[1.0, 0.0], [0.5, 0.5]]})\n'
        "        >>> [round(v, 4) for v in ds.select(\n"
        '        ...     h=bt.col("p").list.entropy()\n'
        '        ... ).to_pydict()["h"]]\n'
        "        [0.0, 0.6931]"
    ),
    # `reverse` is shared by .str (reverse characters) and .list (reverse order);
    # the per-family fallback disambiguates it, so it is intentionally omitted here.
}
