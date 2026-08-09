"""`Pipeline` — the preprocessing and the model as one fitted object.

`Chain` composes preprocessors; the model was always left outside it. That split is where
train/serve skew comes from, because it makes the *caller* responsible for remembering
which transforms a model was trained behind, in what order, and for applying exactly those
at serving time. Nothing checks that, and the failure is silent: a model scored behind one
fewer transform returns numbers, not an error.

`Pipeline` closes that by owning both. `fit` fits each step on the output of the last and
then fits the model on the fully transformed frame; `predict` replays exactly the same
sequence. Because it is one object it also saves as one file, so what ships to serving is
the whole recipe rather than a model plus a memo.

This is scikit-learn's `Pipeline`, with the difference that every step here is lazy: the
transforms are `Expr` projections, so a `predict` over a fitted pipeline is one plan the
engine optimizes end to end rather than a sequence of materialized frames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.ml.preprocessors.base import Preprocessor

__all__ = ["Pipeline"]


class Pipeline:
    """Fit preprocessors and a model together, and score with exactly the same sequence.

    Steps are applied left to right, each fitted on the previous one's output. The model is
    fitted last, on the fully transformed frame, and `predict` replays the identical
    sequence — which is the property that makes a served prediction match a validated one.

    The model is anything with `fit`/`predict` over a `Dataset`: a Batcher estimator, or a
    callable pair wrapped to look like one.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LinearRegression, Pipeline, StandardScaler
            >>> train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
            >>> pipe = Pipeline(StandardScaler(["x"]), model=LinearRegression(["x"], "y"))
            >>> scored = pipe.fit(train).predict(train)
            >>> [round(v, 6) for v in scored.to_pydict()["prediction"]]
            [2.0, 4.0, 6.0, 8.0]

    Args:
        steps: The preprocessors to apply, in order.
        model: The estimator to fit on the transformed frame.
    """

    __slots__ = ("model", "steps")

    def __init__(self, *steps: Preprocessor, model: Any) -> None:
        from batcher.ml.preprocessors.base import Preprocessor as Base

        bad = [s for s in steps if not isinstance(s, Base)]
        if bad:
            raise PlanError(
                f"Pipeline steps must be preprocessors, got {type(bad[0]).__name__}. The "
                "model goes in the model= argument, not in the step list."
            )
        if not callable(getattr(model, "fit", None)) or not callable(
            getattr(model, "predict", None)
        ):
            raise PlanError(
                f"Pipeline model must have fit() and predict(), got "
                f"{type(model).__name__}. Any Batcher estimator qualifies."
            )
        self.steps = list(steps)
        self.model = model

    def fit(self, ds: Dataset) -> Pipeline:
        """Fit each step on the previous one's output, then the model on the result.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression, Pipeline, StandardScaler
                >>> train = bt.from_pydict({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
                >>> pipe = Pipeline(StandardScaler(["x"]), model=LinearRegression(["x"], "y"))
                >>> pipe.fit(train).steps[0].is_fitted
                True

        Args:
            ds: The training data.

        Returns:
            ``self``, fitted.
        """
        transformed = ds
        for step in self.steps:
            transformed = step.fit_transform(transformed)
        self.model.fit(transformed)
        return self

    def transform(self, ds: Dataset) -> Dataset:
        """Apply the fitted steps without scoring, for inspecting the model's actual input.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression, Pipeline, StandardScaler
                >>> train = bt.from_pydict({"x": [1.0, 3.0], "y": [2.0, 6.0]})
                >>> pipe = Pipeline(StandardScaler(["x"]), model=LinearRegression(["x"], "y"))
                >>> pipe.fit(train).transform(train).to_pydict()["x"]
                [-1.0, 1.0]

        Args:
            ds: The dataset to transform.

        Returns:
            A new lazy `Dataset` with every step applied.
        """
        transformed = ds
        for step in self.steps:
            transformed = step.transform(transformed)
        return transformed

    def predict(self, ds: Dataset) -> Dataset:
        """Transform, then score — the same sequence `fit` used.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression, Pipeline, StandardScaler
                >>> train = bt.from_pydict({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
                >>> pipe = Pipeline(StandardScaler(["x"]), model=LinearRegression(["x"], "y"))
                >>> "prediction" in pipe.fit(train).predict(train).columns
                True

        Args:
            ds: The dataset to score.

        Returns:
            A new lazy `Dataset` with the model's prediction column appended.
        """
        return self.model.predict(self.transform(ds))

    def save(self, path: str) -> None:
        """Write the whole pipeline — steps and model — to `path` as one JSON document.

        Saving the two halves separately is what lets them drift apart, so they are written
        together on purpose: what ships to serving is the recipe, not a model and a memo
        about which transforms preceded it.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> from batcher.ml import LinearRegression, Pipeline, StandardScaler
                >>> train = bt.from_pydict({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
                >>> pipe = Pipeline(StandardScaler(["x"]), model=LinearRegression(["x"], "y"))
                >>> target = os.path.join(tempfile.mkdtemp(), "pipe.json")
                >>> pipe.fit(train).save(target)
                >>> len(Pipeline.load(target).steps)
                1

        Args:
            path: Where to write it; a local path or a cloud URI.

        Raises:
            PlanError: If a step or the model holds state JSON cannot represent.
        """
        from batcher.ml.persistence.document import SCHEMA_VERSION, write_document
        from batcher.ml.persistence.models import model_to_dict
        from batcher.ml.preprocessors.persistence import to_dict

        write_document(
            {
                "version": SCHEMA_VERSION,
                "class": "Pipeline",
                "steps": [to_dict(step) for step in self.steps],
                "model": model_to_dict(self.model),
            },
            path,
        )

    @staticmethod
    def load(path: str) -> Pipeline:
        """Read a pipeline written by `save`, steps and model both fitted.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> from batcher.ml import LinearRegression, Pipeline, StandardScaler
                >>> train = bt.from_pydict({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
                >>> pipe = Pipeline(StandardScaler(["x"]), model=LinearRegression(["x"], "y"))
                >>> target = os.path.join(tempfile.mkdtemp(), "pipe.json")
                >>> pipe.fit(train).save(target)
                >>> "prediction" in Pipeline.load(target).predict(train).columns
                True

        Args:
            path: The local path or cloud URI to read.

        Returns:
            The reconstructed pipeline, ready to `predict`.

        Raises:
            PlanError: On an unreadable document or an unknown schema version.
        """
        from batcher.ml.persistence.document import check_version, read_document
        from batcher.ml.persistence.models import model_from_dict
        from batcher.ml.preprocessors.persistence import from_dict

        document = read_document(path)
        check_version(document)
        steps = [from_dict(step) for step in document.get("steps", [])]
        return Pipeline(*steps, model=model_from_dict(document["model"]))
