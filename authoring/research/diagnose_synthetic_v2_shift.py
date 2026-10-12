from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Load generator functions without running its main() or overwriting files.
module = runpy.run_path("authoring/generate_synthetic_fraud_v2.py")
generate_features = module["generate_features"]
calibrate_intercept = module["calibrate_intercept"]
sigmoid = module["sigmoid"]
FEATURES = module["FEATURES"]

SEEDS = [1101, 1202, 1303, 1404, 1505]
N_TRAIN = 25_000
N_EVAL = 17_000
TARGET_PREVALENCE = 0.025

OUT = Path(
    "authoring/research/"
    "synthetic_v2_regime_diagnostic_results.csv"
)


def make_models():
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
        ),
        "Interaction logistic": make_pipeline(
            PolynomialFeatures(
                degree=2,
                interaction_only=True,
                include_bias=False,
            ),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                C=0.1,
                random_state=42,
            ),
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def metrics(y, scores):
    k = max(1, int(len(y) * 0.05))
    top = np.argsort(scores)[-k:]

    return {
        "fraud_prevalence": float(y.mean()),
        "AP": average_precision_score(y, scores),
        "AUC": roc_auc_score(y, scores),
        "top5_recall": y[top].sum() / max(1, y.sum()),
    }


def simple_rule(x):
    return (
        x["transaction_velocity"]
        / max(x["transaction_velocity"].std(), 1e-12)
        + x["distance_km"]
        / max(x["distance_km"].std(), 1e-12)
    ).to_numpy()


rows = []

for seed in SEEDS:
    # Independent random streams make the experiment reproducible.
    # The training data are shared by both evaluation conditions.
    seed_sequence = np.random.SeedSequence(seed)
    train_ss, control_ss, shift_ss, random_ss = seed_sequence.spawn(4)

    train_rng = np.random.default_rng(train_ss)
    control_rng = np.random.default_rng(control_ss)
    shift_rng = np.random.default_rng(shift_ss)
    random_rng = np.random.default_rng(random_ss)

    # Historical training regime and its fixed calibration intercept.
    train_x, train_score = generate_features(
        N_TRAIN, train_rng, "training"
    )
    intercept = calibrate_intercept(
        train_score, TARGET_PREVALENCE
    )
    train_y = train_rng.binomial(
        1, sigmoid(train_score + intercept)
    )

    # Control: same historical risk formula, fresh transactions.
    control_x, control_score = generate_features(
        N_EVAL, control_rng, "validation"
    )
    control_y = control_rng.binomial(
        1, sigmoid(control_score + intercept)
    )

    # Shift: later risk formula, same feature-generation mechanism
    # and the same training-derived intercept.
    shift_x, shift_score = generate_features(
        N_EVAL, shift_rng, "evaluation"
    )
    shift_y = shift_rng.binomial(
        1, sigmoid(shift_score + intercept)
    )

    X_train = train_x[FEATURES]
    X_control = control_x[FEATURES]
    X_shift = shift_x[FEATURES]

    datasets = {
        "same_regime_control": (X_control, control_y),
        "shifted_regime": (X_shift, shift_y),
    }

    # Label-free baselines.
    random_scores = {
        period: random_rng.random(len(y))
        for period, (_, y) in datasets.items()
    }
    rule_scores = {
        period: simple_rule(x)
        for period, (x, _) in datasets.items()
    }

    candidates = {
        "Random ranking": random_scores,
        "Velocity + distance rule": rule_scores,
    }

    # Fit each model once. Evaluate the same fitted model in both regimes.
    for name, model in make_models().items():
        model.fit(X_train, train_y)
        candidates[name] = {
            period: model.predict_proba(x)[:, 1]
            for period, (x, _) in datasets.items()
        }

    for method, period_scores in candidates.items():
        for period, (x, y) in datasets.items():
            row = {
                "seed": seed,
                "method": method,
                "period": period,
                **metrics(y, period_scores[period]),
            }
            rows.append(row)

    print(
        f"Seed {seed}: "
        f"train={train_y.mean():.3%}, "
        f"control={control_y.mean():.3%}, "
        f"shifted={shift_y.mean():.3%}"
    )

results = pd.DataFrame(rows)

# New output path; existing V2 evaluation results are not touched.
results.to_csv(OUT, index=False)

metric_cols = ["fraud_prevalence", "AP", "AUC", "top5_recall"]
summary = results.groupby(["period", "method"])[metric_cols].mean()

print("\n=== MEAN RESULTS ACROSS FIVE SEEDS ===")
print(summary.round(4).to_string())

print("\n=== AP VARIABILITY ACROSS SEEDS ===")
print(
    results.groupby(["period", "method"])["AP"]
    .agg(["mean", "std", "min", "max"])
    .round(4)
    .to_string()
)

# Positive AP drop means performance is lower under the shifted regime.
ap = results.pivot(index="seed", columns=["period", "method"], values="AP")
methods = results["method"].unique()

print("\n=== PER-SEED AP DROP: CONTROL MINUS SHIFTED ===")
for method in methods:
    control = ap[("same_regime_control", method)]
    shifted = ap[("shifted_regime", method)]
    drops = control - shifted

    print(
        f"{method}: "
        f"mean drop={drops.mean():.4f}, "
        f"seed SD={drops.std(ddof=1):.4f}, "
        f"control mean={control.mean():.4f}, "
        f"shifted mean={shifted.mean():.4f}"
    )

print(f"\nDetailed results saved to: {OUT}")
print("Original generator and evaluation results were not modified.")
