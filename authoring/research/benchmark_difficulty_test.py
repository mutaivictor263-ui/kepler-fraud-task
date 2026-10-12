import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import average_precision_score, roc_auc_score

generate_period = runpy.run_path(
    "authoring/generate_synthetic_fraud.py"
)["generate_period"]

SEEDS = [101, 202, 303, 404, 505]
FEATURES = [
    "amount_log", "merchant_risk", "device_novelty",
    "failed_attempts", "transaction_velocity", "distance_km",
    "account_age_days", "hour_of_day",
]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def evaluation_score_base(df):
    return (
        0.15 * df["amount_log"]
        + 0.45 * df["merchant_risk"]
        + 1.0 * df["device_novelty"]
        + 0.35 * df["failed_attempts"]
        + 0.42 * df["transaction_velocity"]
        + 0.010 * df["distance_km"]
        - 0.0002 * df["account_age_days"]
        + 0.65 * (
            (df["transaction_velocity"] >= 4)
            & (df["distance_km"] >= 80)
        ).astype(float)
    ).to_numpy()


def build_models():
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
        ),
        "Interaction logistic": make_pipeline(
            PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False
            ),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                C=0.1,
                random_state=42,
            ),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "Histogram gradient boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def measure(name, seed, y, scores, n_review):
    top = np.argsort(scores)[-n_review:]
    positives = int(np.sum(y))
    return {
        "seed": seed,
        "method": name,
        "evaluation_prevalence": float(np.mean(y)),
        "average_precision": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "top5_precision": float(np.mean(y[top])),
        "top5_recall": float(np.sum(y[top]) / max(1, positives)),
    }


records = []

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    train = generate_period(25_000, rng, "training")
    validation = generate_period(8_000, rng, "validation")
    evaluation = generate_period(17_000, rng, "evaluation")

    X_train = train[FEATURES]
    y_train = train["label"].to_numpy()
    X_val = validation[FEATURES]
    y_val = validation["label"].to_numpy()
    X_eval = evaluation[FEATURES]

    # Calibrate the evaluation intercept so expected fraud prevalence
    # matches the observed training prevalence.
    base = evaluation_score_base(evaluation)
    low, high = -12.0, 2.0
    for _ in range(100):
        mid = (low + high) / 2
        if sigmoid(base + mid).mean() < y_train.mean():
            low = mid
        else:
            high = mid

    probabilities = sigmoid(base + (low + high) / 2)
    label_rng = np.random.default_rng(seed + 900_000)
    y_eval = label_rng.binomial(1, probabilities)

    n_review = max(1, int(len(evaluation) * 0.05))
    print(
        f"\nSeed {seed}: train={y_train.mean():.3%}, "
        f"evaluation={y_eval.mean():.3%}, "
        f"expected evaluation={probabilities.mean():.3%}"
    )

    random_scores = np.random.default_rng(seed + 700_000).random(
        len(evaluation)
    )
    row = measure(
        "Random ranking", seed, y_eval, random_scores, n_review
    )
    records.append(row)

    # Diagnostic only: this uses the hidden evaluation-generating
    # relationship and is not a candidate agent solution.
    records.append(
        measure("Oracle diagnostic", seed, y_eval, base, n_review)
    )

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]
        eval_scores = model.predict_proba(X_eval)[:, 1]

        val_ap = average_precision_score(y_val, val_scores)
        row = measure(name, seed, y_eval, eval_scores, n_review)
        row["validation_ap"] = float(val_ap)
        records.append(row)

        print(
            f"  {name}: validation AP={val_ap:.4f}, "
            f"evaluation AP={row['average_precision']:.4f}, "
            f"AUC={row['roc_auc']:.4f}, "
            f"top-5% recall={row['top5_recall']:.3%}"
        )

    print(
        f"  Random ranking AP={records[-(len(build_models()) + 1)]['average_precision']:.4f}"
        " (see saved results for all baselines)"
    )

results = pd.DataFrame(records)
out = Path("authoring/research/benchmark_difficulty_test_results.csv")
results.to_csv(out, index=False)

summary = results.groupby("method").agg(
    runs=("seed", "count"),
    evaluation_ap_mean=("average_precision", "mean"),
    evaluation_ap_std=("average_precision", "std"),
    evaluation_auc_mean=("roc_auc", "mean"),
    top5_precision_mean=("top5_precision", "mean"),
    top5_recall_mean=("top5_recall", "mean"),
)

print("\nDIFFICULTY SUMMARY — FIVE SEEDS")
print(summary.round(4).to_string())
print(f"\nDetailed results saved to: {out}")
print("Research diagnostic only; no preview or task files were overwritten.")
