"""Every object-store scheme actually gets in-place write semantics, not just set membership.

Publishing a written file by `rename` is atomic on a real filesystem and is a **full
server-side object copy** on an object store, which is why `_ArrowFileSystem` carries an
`atomic_rename` flag and why `_OBJECT_STORE_SCHEMES` exists to clear it. A scheme missing
from that set is not an error anywhere -- the write still succeeds, it just silently costs a
copy of every byte, and gains a window where a reader can observe the temporary object.

`test_object_store_portability.py` guards the set against a scheme being *removed*, which is
worth having. What it does not do is connect the set to the behaviour its own name promises:
its assertion is `canonical in _OBJECT_STORE_SCHEMES` for a hand-written list of nine
schemes, so it would pass unchanged if `_wrap_user_filesystem` stopped reading the set at
all, and it says nothing about the ten schemes not on that list (`s3`, `gs`, `abfs`, `az`,
`azure`, `gcs`, `s3a`, `wasb`, `wasbs`, `abfss`) -- including the three most used.

So this file asserts the consequence instead of the membership, and derives its cases from
`_OBJECT_STORE_SCHEMES` and `_SCHEME_ALIASES` so that adding a scheme to either extends the
coverage automatically rather than needing a second edit here.

No network and no credentials: `_wrap_user_filesystem` takes a caller-supplied filesystem
and reads the *scheme of the path* to make this decision, so a `LocalFileSystem` handle and
an `s3://` URI exercise the real code path offline. That is the same seam
`test_object_store_portability.py` already uses for its one hand-in case.
"""

from __future__ import annotations

import pyarrow.fs as pafs
import pytest

import batcher.io.filesystem as fsmod

pytestmark = pytest.mark.unit

_ALL_SCHEMES = sorted(fsmod._OBJECT_STORE_SCHEMES | set(fsmod._SCHEME_ALIASES))


def _wrapped(path: str):
    return fsmod._wrap_user_filesystem(path, pafs.LocalFileSystem())


@pytest.mark.parametrize("scheme", _ALL_SCHEMES)
def test_an_object_store_scheme_is_never_rename_published(scheme):
    """The property the set exists for, asserted through the code that reads the set."""
    fs = _wrapped(f"{scheme}://bucket/dir/part-0.parquet")
    assert fs._atomic_rename is False, (
        f"`{scheme}://` would publish by rename, which an object store serves as a full "
        "server-side copy of the object"
    )


@pytest.mark.parametrize("scheme", _ALL_SCHEMES)
def test_an_object_store_scheme_is_cacheable(scheme):
    """The other half of the same decision, and the reason it is one branch and not two.

    `cacheable` is the exact complement of `atomic_rename` at this seam. Pinning only one of
    them would let the pair drift apart into a state where a remote store is neither renamed
    atomically nor cached.
    """
    assert _wrapped(f"{scheme}://bucket/x.parquet")._cacheable is True


@pytest.mark.parametrize("scheme", _ALL_SCHEMES)
def test_the_scheme_prefix_survives_the_wrap(scheme):
    """A store path keeps its `<scheme>://` prefix, so callers can go on passing full URIs."""
    assert _wrapped(f"{scheme}://bucket/x.parquet")._prefix == f"{scheme}://"


@pytest.mark.parametrize("path", ["/tmp/x.parquet", "file:///tmp/x.parquet", "relative/x.parquet"])
def test_a_real_filesystem_still_publishes_atomically(path):
    """The negative control. Without it, a function that returned `False` unconditionally
    would satisfy every assertion above -- and losing atomic publish on a POSIX filesystem
    is the more damaging of the two mistakes, because it is a correctness property rather
    than a cost one.
    """
    fs = _wrapped(path)
    assert fs._atomic_rename is True
    assert fs._cacheable is False


def test_an_unknown_scheme_is_treated_as_a_real_filesystem():
    """The conservative direction: an unrecognized scheme keeps atomic publish.

    This is what makes a *missing* entry in `_OBJECT_STORE_SCHEMES` cost performance rather
    than correctness, and it is worth pinning so the default cannot be flipped to "assume
    object store" on the reasoning that it is the more common case.
    """
    fs = _wrapped("quantumfs://bucket/x.parquet")
    assert fs._atomic_rename is True
    assert fs._cacheable is False


def test_every_alias_agrees_with_the_scheme_it_aliases():
    """An alias exists to make two spellings behave identically; assert that they do.

    `s3n://` and `s3://` differing here would be invisible -- both are object stores, both
    write correctly, and only the copy cost would differ between two spellings of one store.
    """
    for alias, canonical in fsmod._SCHEME_ALIASES.items():
        a, c = _wrapped(f"{alias}://b/x"), _wrapped(f"{canonical}://b/x")
        assert (a._atomic_rename, a._cacheable) == (c._atomic_rename, c._cacheable), (
            f"`{alias}://` and `{canonical}://` name one store but were treated differently"
        )
