"""`FeatureSpec` — pinning the exact feature contract between training and serving.

The most expensive bug in tabular ML is not in the model; it is in the columns around it. A
model scores by position, so a serving request that presents its features in a different
order, with an extra column, or with a column silently retyped produces confident garbage
and raises nothing. The gap between "the training frame" and "the serving frame" is where
that happens, and nothing in a fitted model or a fitted preprocessor closes it.

A `FeatureSpec` is that contract, captured from the training data and checked against
everything downstream. It records the feature columns, their order, and their dtypes, and it
does three things with them: **validate** that a frame matches (raising on the mismatch a
model would otherwise absorb), **align** a frame to the pinned order (so a reordered serving
frame is fixed rather than rejected), and **travel** as JSON beside the model it describes.

It is deliberately not a transformer. It learns nothing statistical and changes no value; it
only asserts that the shape a model was trained on is the shape it is being asked to score.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.stats._shared import require_columns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["FeatureSpec"]

#: Bumped when the JSON shape changes incompatibly, so an old reader errors rather than
#: silently misreading a field.
_SCHEMA_VERSION = 1


class FeatureSpec:
    """The pinned feature contract of a trained model — columns, order, and dtypes.

    Build one from the training frame with `from_dataset`, save it beside the model, and load
    and apply it at serving time. `validate` raises on any drift a model would otherwise
    score against silently; `align` reorders a frame to the pinned order and selects exactly
    the feature columns, which is usually what you want at serving time.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import FeatureSpec
            >>> train = bt.from_pydict({"age": [30, 40], "income": [50.0, 60.0], "label": [0, 1]})
            >>> spec = FeatureSpec.from_dataset(train, features=["age", "income"])
            >>> spec.features
            ['age', 'income']

    Dtype names are the engine's own (``"int64"``, ``"double"``, ``"string"``, …) as
    `Dataset.dtypes` renders them, so a spec built with `from_dataset` and one built by hand
    agree. Prefer `from_dataset`, which reads them off the training frame.

    Args:
        features: The feature columns, in the exact order a model consumes them.
        dtypes: The pinned dtype of each feature, keyed by name.
    """

    __slots__ = ("dtypes", "features")

    def __init__(self, features: Sequence[str], dtypes: dict[str, str]) -> None:
        feats = list(features)
        if not feats:
            raise PlanError("a FeatureSpec needs at least one feature column")
        missing = [f for f in feats if f not in dtypes]
        if missing:
            raise PlanError(f"no dtype pinned for feature(s) {missing}")
        self.features = feats
        self.dtypes = {name: dtypes[name] for name in feats}

    @classmethod
    def from_dataset(
        cls, ds: Dataset, *, features: Sequence[str] | None = None, exclude: Sequence[str] = ()
    ) -> FeatureSpec:
        """Capture the feature contract from a training `Dataset`.

        Args:
            ds: The training dataset.
            features: The feature columns to pin, in model order. Defaults to every column
                except those in `exclude`.
            exclude: Columns to leave out when `features` is not given — the label, an id,
                anything that is not a feature.

        Returns:
            A `FeatureSpec` recording those columns' order and dtypes.

        Raises:
            PlanError: If `features` is empty or names a column the dataset lacks.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import FeatureSpec
                >>> ds = bt.from_pydict({"a": [1.0], "b": [2.0], "y": [0]})
                >>> FeatureSpec.from_dataset(ds, exclude=["y"]).features
                ['a', 'b']
        """
        schema = dict(zip(ds.columns, [str(dt) for dt in ds.dtypes], strict=True))
        if features is None:
            chosen = [c for c in ds.columns if c not in set(exclude)]
        else:
            chosen = list(features)
            require_columns(ds, *chosen, hint="Pass a real column.")
        return cls(chosen, {name: schema[name] for name in chosen})

    def validate(self, ds: Dataset, *, check_dtypes: bool = True) -> None:
        """Raise unless `ds` presents exactly the pinned features, in order and by dtype.

        The check a model cannot make for itself. A missing feature, an extra one, a
        reordering, or a retyped column each produces a specific, actionable error here
        rather than a silently wrong prediction downstream.

        Args:
            ds: The frame to check — a serving batch, a fresh extract, a re-run.
            check_dtypes: Also require each feature's dtype to match the pinned one. Turn it
                off when a widening cast (Int32 to Int64) is acceptable and expected.

        Raises:
            PlanError: On a missing, extra, misordered, or (when `check_dtypes`) retyped
                feature, naming exactly what differs.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import FeatureSpec
                >>> spec = FeatureSpec(["a", "b"], {"a": "int64", "b": "double"})
                >>> spec.validate(bt.from_pydict({"a": [1], "b": [2.0]}))
        """
        present = list(ds.columns)
        feature_set = set(self.features)
        actual_features = [c for c in present if c in feature_set]
        # Against a set, not the `present` list: both sides scale with the model's feature
        # count, so a wide tabular spec (thousands of features is normal) was validating in
        # time quadratic in its own width — on a check that exists to be cheap.
        present_set = set(present)
        missing = [f for f in self.features if f not in present_set]
        if missing:
            raise PlanError(
                f"the frame is missing pinned feature(s) {missing}. A tabular model scores by "
                "position, so a missing feature shifts every later column onto the wrong slot."
            )
        if actual_features != self.features:
            raise PlanError(
                f"the frame's feature order {actual_features} does not match the pinned order "
                f"{self.features}. Call align() to fix it, or reorder the columns."
            )
        if check_dtypes:
            schema = dict(zip(ds.columns, [str(dt) for dt in ds.dtypes], strict=True))
            mismatched = {
                name: (self.dtypes[name], schema[name])
                for name in self.features
                if schema[name] != self.dtypes[name]
            }
            if mismatched:
                detail = ", ".join(
                    f"{name}: pinned {pinned}, got {actual}"
                    for name, (pinned, actual) in mismatched.items()
                )
                raise PlanError(
                    f"feature dtype mismatch ({detail}). A retyped column changes what a model "
                    "sees; cast it back, or pass check_dtypes=False if the change is intended."
                )

    def align(self, ds: Dataset, *, cast: bool = False) -> Dataset:
        """Select exactly the pinned features, in the pinned order — the serving-frame fix.

        A serving frame routinely arrives with extra columns and in whatever order the
        upstream system produced. This projects it to the feature set the model expects, in
        the order the model expects, so a reordered or column-rich frame is *repaired* rather
        than rejected. With `cast`, each feature is coerced to its pinned dtype as well.

        Args:
            ds: The frame to align.
            cast: Also cast each feature to its pinned dtype.

        Returns:
            A new lazy `Dataset` with exactly the pinned features, in order.

        Raises:
            PlanError: If a pinned feature is absent, since alignment cannot invent it.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import FeatureSpec
                >>> spec = FeatureSpec(["a", "b"], {"a": "int64", "b": "double"})
                >>> messy = bt.from_pydict({"b": [2.0], "extra": ["x"], "a": [1]})
                >>> spec.align(messy).columns
                ['a', 'b']
        """
        missing = [f for f in self.features if f not in ds.columns]
        if missing:
            raise PlanError(
                f"cannot align: the frame lacks pinned feature(s) {missing}. Alignment reorders "
                "and selects; it cannot create a missing column."
            )
        aligned = ds.select(*self.features)
        if cast:
            from batcher.plan.expr_ir import col

            aligned = aligned.with_columns(
                **{name: col(name).cast(self.dtypes[name]) for name in self.features}
            )
        return aligned

    def to_dict(self) -> dict[str, Any]:
        """The spec as a plain, JSON-safe dictionary.

        Examples:
            .. doctest::

                >>> from batcher.ml import FeatureSpec
                >>> FeatureSpec(["a"], {"a": "int64"}).to_dict()["features"]
                ['a']

        Returns:
            A dict with ``version``, ``features``, and ``dtypes``.
        """
        return {
            "version": _SCHEMA_VERSION,
            "features": list(self.features),
            "dtypes": dict(self.dtypes),
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> FeatureSpec:
        """Rebuild a spec from a `to_dict` document.

        Args:
            document: The dictionary `to_dict` produced.

        Returns:
            The reconstructed `FeatureSpec`.

        Raises:
            PlanError: On an unsupported schema version.

        Examples:
            .. doctest::

                >>> from batcher.ml import FeatureSpec
                >>> doc = {"version": 1, "features": ["a"], "dtypes": {"a": "int64"}}
                >>> FeatureSpec.from_dict(doc).features
                ['a']
        """
        version = document.get("version")
        if version != _SCHEMA_VERSION:
            raise PlanError(
                f"unsupported FeatureSpec schema version {version!r}; this build reads "
                f"version {_SCHEMA_VERSION}."
            )
        return cls(document["features"], document["dtypes"])

    def save(self, path: str) -> None:
        """Write the spec to `path` as JSON (local or cloud URI).

        Examples:
            .. doctest::

                >>> import os, tempfile
                >>> from batcher.ml import FeatureSpec
                >>> spec = FeatureSpec(["a"], {"a": "int64"})
                >>> target = os.path.join(tempfile.mkdtemp(), "spec.json")
                >>> spec.save(target)
                >>> FeatureSpec.load(target).features
                ['a']

        Args:
            path: Where to write it.
        """
        from batcher.io.filesystem import resolve_filesystem

        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True).encode()
        filesystem = resolve_filesystem(path)
        with filesystem.atomic_writer(path) as handle:
            handle.write(payload)

    @classmethod
    def load(cls, path: str) -> FeatureSpec:
        """Read a spec written by `save`.

        Args:
            path: The local path or cloud URI to read.

        Returns:
            The reconstructed `FeatureSpec`.

        Raises:
            PlanError: On an unreadable document or an unsupported schema version.

        Examples:
            .. doctest::

                >>> import os, tempfile
                >>> from batcher.ml import FeatureSpec
                >>> target = os.path.join(tempfile.mkdtemp(), "spec.json")
                >>> FeatureSpec(["a", "b"], {"a": "int64", "b": "double"}).save(target)
                >>> FeatureSpec.load(target).features
                ['a', 'b']
        """
        from batcher.io.filesystem import resolve_filesystem

        filesystem = resolve_filesystem(path)
        with filesystem.open(path, "rb") as handle:
            raw = handle.read()
        try:
            document = json.loads(raw)
        except ValueError as exc:
            raise PlanError(f"{path!r} is not a saved FeatureSpec: {exc}") from exc
        return cls.from_dict(document)

    def __repr__(self) -> str:
        """``FeatureSpec(3 features: age, income, ...)`` — a short summary."""
        shown = ", ".join(self.features[:3])
        tail = ", ..." if len(self.features) > 3 else ""
        return f"FeatureSpec({len(self.features)} features: {shown}{tail})"
