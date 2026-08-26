"""An end-to-end tabular ML workflow: split, fit, score, evaluate, monitor.

The classical-ML lifecycle as one Batcher script, with no step leaving the engine and no
step materializing the data on the driver:

1. **Split** with a group-aware k-fold, so the same customer never appears in both the
   training and the validation half — the most common silent leak in applied ML.
2. **Preprocess** with fitted objects, so the held-out split inherits the *training*
   statistics rather than learning its own.
3. **Train** a scikit-learn model. This is the one step that is not a Batcher operator:
   fitting needs the whole matrix, so the training fold is collected here deliberately.
4. **Score** the validation fold with ``ds.ml.predict``, which loads the model once per
   worker and hands it whole Arrow batches.
5. **Evaluate** with ``ds.ml.evaluate``, whose aggregate metrics are a single pass — and
   then evaluate *per segment*, which costs the same and is where the interesting failures
   show up.
6. **Monitor** for input drift against the training distribution, which is the only signal
   available before the labels arrive.

Needs scikit-learn (``pip install 'batcher-engine[sklearn]'``).

    python examples/tabular_ml.py
"""

from __future__ import annotations

import random

import batcher as bt
from batcher.ml.metrics import calibration_curve, lift_table
from batcher.ml.preprocessors import Chain, MissingIndicator, SimpleImputer, StandardScaler
from batcher.ml.stats import drift_report, information_value

REGIONS = ("emea", "amer", "apac")


def build_dataset(rows: int, *, seed: int, drifted: bool = False) -> bt.Dataset:
    """A synthetic churn table: one row per subscription, several rows per customer."""
    rng = random.Random(seed)
    shift = 20.0 if drifted else 0.0
    customers = [f"cust-{i // 3:05d}" for i in range(rows)]
    tenure = [rng.gauss(24.0, 10.0) + shift for _ in range(rows)]
    monthly = [rng.gauss(70.0, 25.0) for _ in range(rows)]
    support = [float(rng.randint(0, 6)) for _ in range(rows)]
    # Churn falls with tenure and rises with support contacts, plus noise.
    churn = [
        int(rng.random() < 1.0 / (1.0 + pow(2.718, 0.06 * t - 0.45 * s - 0.6)))
        for t, s in zip(tenure, support, strict=True)
    ]
    # A tenth of the tenure readings are missing, as they are in every real table.
    tenure = [None if rng.random() < 0.1 else t for t in tenure]
    return bt.from_pydict(
        {
            "customer": customers,
            "region": [rng.choice(REGIONS) for _ in range(rows)],
            "tenure_months": tenure,
            "monthly_charge": monthly,
            "support_contacts": support,
            "churned": churn,
        }
    )


def main() -> None:
    """Run the whole workflow and print what each step found."""
    features = ["tenure_months", "monthly_charge", "support_contacts"]
    ds = build_dataset(4_000, seed=7)

    # 1. Group-aware split. Rows repeat a customer, so a plain k-fold would put the same
    #    customer in both halves and the model would memorize them instead of learning.
    train, validate = ds.ml.kfold(5, group="customer")[0]
    print(f"train {train.count()} rows / validate {validate.count()} rows")
    # The leak this split exists to prevent is invisible in every downstream number: a
    # model trained with the same customer on both sides scores *better*, so nothing here
    # would look wrong. Check the property directly.
    train_customers = set(train.select("customer").to_pydict()["customer"])
    validate_customers = set(validate.select("customer").to_pydict()["customer"])
    assert not (train_customers & validate_customers), (
        f"{len(train_customers & validate_customers)} customers appear in both folds — the "
        f"group-aware split is not grouping"
    )
    assert train.count() + validate.count() == ds.count(), "a fold must partition, not sample"

    # 2. Fit the preprocessing on train only; transform both with the same fitted state.
    #    MissingIndicator runs *before* the imputer, or the missingness signal is destroyed.
    chain = Chain(
        MissingIndicator("tenure_months"),
        SimpleImputer(["tenure_months"]),
        StandardScaler(features),
    )
    train_ready = chain.fit_transform(train)
    validate_ready = chain.transform(validate)
    # `transform` must reuse the *training* statistics. If it refit on the validation fold
    # the scaled columns would still look perfectly reasonable — centred, unit variance —
    # while having quietly leaked the held-out distribution into the features.
    scaled_train = train_ready.select(*features).to_pydict()
    for name in features:
        mean = sum(scaled_train[name]) / len(scaled_train[name])
        assert abs(mean) < 0.05, f"{name} should be centred on the training fold, got {mean:.3f}"
    # The missingness indicator has to survive imputation, or the signal it carries is gone
    # by the time the model sees the row.
    assert "tenure_months_missing" in train_ready.schema.names
    assert train_ready.filter(bt.col("tenure_months").is_null()).count() == 0, "imputed"

    # 3. Fit the model. The only step that needs the whole matrix in memory.
    from sklearn.linear_model import LogisticRegression

    frame = train_ready.select(*features, "churned").to_pydict()
    matrix = list(zip(*(frame[name] for name in features), strict=True))
    model = LogisticRegression(max_iter=500).fit(matrix, frame["churned"])

    # 4. Score. The model loads once per worker and sees whole Arrow batches.
    scored = validate_ready.ml.predict(
        model, features=features, method="predict_proba", output_columns=["p_stay", "p_churn"]
    )

    # 5. Evaluate — the whole metric set in one pass over the scored rows.
    report = scored.ml.evaluate("churned", y_score="p_churn")
    for name, value in report.items():
        print(f"  {name:20s} {value:.4f}")
    # The generator in `build_dataset` makes churn a real function of tenure and support
    # contacts, so a working pipeline must beat chance by a clear margin. Asserting only
    # that the metrics *exist* would pass on a model that learned nothing, which is exactly
    # what a broken split or a mis-ordered feature matrix produces.
    assert report["roc_auc"] > 0.65, f"the signal is learnable; got AUC {report['roc_auc']:.3f}"
    assert 0.0 <= report["accuracy"] <= 1.0 and 0.0 <= report["brier_score"] <= 1.0
    # `predict_proba` output columns must be a distribution, not two independent scores.
    probabilities = scored.select("p_stay", "p_churn").to_pydict()
    for stay, churn in zip(probabilities["p_stay"], probabilities["p_churn"], strict=True):
        assert abs(stay + churn - 1.0) < 1e-9

    # ...and the same metrics per region, which costs the same and is where a model that
    # looks fine overall turns out not to be.
    per_region = scored.ml.evaluate(
        "churned", y_score="p_churn", by="region", metrics=["roc_auc", "accuracy"]
    )
    print("\nper region:")
    segments = per_region.sort("region").to_pydict()
    print(segments)
    # Grouped evaluation must cover every segment and stay in range; a `by=` that silently
    # dropped a group would print a shorter, entirely plausible table.
    assert sorted(segments["region"]) == sorted(REGIONS)
    assert all(0.0 <= v <= 1.0 for v in segments["roc_auc"] + segments["accuracy"])

    # The decile lift table, which is what a retention team actually reads.
    lift = lift_table(scored, "churned", "p_churn").to_pydict()
    print("\ntop-decile lift:", round(lift["lift"][0], 2))
    # Lift is the point of a ranking model: the top decile must contain more churners than
    # a random decile would. A lift of 1.0 means the scores carry no ordering information,
    # which is a silent failure — every metric above can still look acceptable.
    assert lift["lift"][0] > 1.0, f"top decile is no better than chance: {lift['lift'][0]:.2f}"

    # Calibration: does a predicted 0.7 actually churn 70% of the time?
    calibration = calibration_curve(scored, "churned", "p_churn", bins=5).to_pydict()
    print("calibration error by bin:", [round(v, 3) for v in calibration["calibration_error"]])
    assert len(calibration["calibration_error"]) == 5, "one row per requested bin"
    assert all(0.0 <= e <= 1.0 for e in calibration["calibration_error"])

    # 6. Feature screening and drift, neither of which needs a model.
    print("\ninformation value:")
    iv = {name: information_value(train, name, "churned", buckets=5) for name in features}
    for name in features:
        print(f"  {name:20s} {iv[name]:.4f}")
    # `build_dataset` weights support contacts (0.45) more heavily than tenure (0.06) and
    # gives monthly charge no influence at all, so a screen that works must rank them that
    # way. Checking only that the numbers are positive would pass on a shuffled column.
    assert iv["support_contacts"] > iv["tenure_months"] > iv["monthly_charge"], iv

    later = build_dataset(4_000, seed=11, drifted=True)
    drift = drift_report(train, later, features, buckets=5).to_pydict()
    print("\ndrift vs training:")
    for name, psi, shift in zip(drift["column"], drift["psi"], drift["mean_shift"], strict=True):
        verdict = "SIGNIFICANT" if psi > 0.25 else "moderate" if psi > 0.1 else "stable"
        print(f"  {name:20s} psi={psi:6.3f} ({verdict})  mean shift {shift:+.2f}")
    # `drifted=True` shifts tenure by exactly +20 and leaves the other two alone, so this is
    # a drift detector with a known answer: it must fire on tenure and stay quiet on the
    # rest. A detector that fired on everything, or nothing, would print an equally
    # confident table.
    psi = dict(zip(drift["column"], drift["psi"], strict=True))
    shifts = dict(zip(drift["column"], drift["mean_shift"], strict=True))
    assert psi["tenure_months"] > 0.25, "the injected shift must read as SIGNIFICANT"
    assert psi["monthly_charge"] < 0.1 and psi["support_contacts"] < 0.1, "no false alarms"
    assert abs(shifts["tenure_months"] - 20.0) < 1.0, "and the size of the shift is recovered"


if __name__ == "__main__":
    main()
