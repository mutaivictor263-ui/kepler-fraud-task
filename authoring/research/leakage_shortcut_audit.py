import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
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


distribution_rows = []
association_rows = []
order_rows = []

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    train = generate_period(25_000, rng, "training")
    validation = generate_period(8_000, rng, "validation")
    evaluation = generate_period(17_000, rng, "evaluation")

    # Match expected evaluation prevalence to training prevalence.
    base = evaluation_score_base(evaluation)
    low, high = -12.0, 2.0
    for _ in range(100):
        mid = (low + high) / 2
        if sigmoid(base + mid).mean() < train["label"].mean():
            low = mid
        else:
            high = mid

    probabilities = sigmoid(base + (low + high) / 2)
    y_eval = np.random.default_rng(seed + 900_000).binomial(
        1, probabilities
    )

    # 1. Feature-distribution shift.
    for feature in FEATURES:
        a = train[feature].to_numpy()
        b = evaluation[feature].to_numpy()
        pooled_sd = np.sqrt((np.var(a) + np.var(b)) / 2)
        smd = (np.mean(b) - np.mean(a)) / pooled_sd if pooled_sd else 0.0

        distribution_rows.append({
            "seed": seed,
            "feature": feature,
            "train_mean": np.mean(a),
            "evaluation_mean": np.mean(b),
            "standardized_mean_difference": smd,
            "ks_statistic": ks_2samp(a, b).statistic,
        })

    # 2. Single-feature shortcut test on evaluation labels.
    for feature in FEATURES:
        scores = evaluation[feature].to_numpy()
        ap_forward = average_precision_score(y_eval, scores)
        ap_reverse = average_precision_score(y_eval, -scores)
        association_rows.append({
            "seed": seed,
            "feature": feature,
            "best_direction_ap": max(ap_forward, ap_reverse),
            "auc": roc_auc_score(y_eval, scores),
        })

    # 3. Check if row position alone predicts the labels.
    row_position = np.arange(len(evaluation))
    order_rows.append({
        "seed": seed,
        "evaluation_prevalence": y_eval.mean(),
        "row_position_auc": roc_auc_score(y_eval, row_position),
        "row_position_ap": average_precision_score(y_eval, row_position),
    })

distribution = pd.DataFrame(distribution_rows)
association = pd.DataFrame(association_rows)
order = pd.DataFrame(order_rows)

out_dir = Path("authoring/research")
distribution.to_csv(out_dir / "audit_feature_distributions.csv", index=False)
association.to_csv(out_dir / "audit_single_feature_shortcuts.csv", index=False)
order.to_csv(out_dir / "audit_row_order.csv", index=False)

print("\n=== FEATURE DISTRIBUTION SHIFT ===")
print(
    distribution.groupby("feature").agg(
        mean_abs_smd=("standardized_mean_difference",
                      lambda x: np.mean(np.abs(x))),
        mean_ks=("ks_statistic", "mean"),
    ).sort_values("mean_ks", ascending=False).round(4).to_string()
)

print("\n=== SINGLE-FEATURE SHORTCUTS ===")
print(
    association.groupby("feature").agg(
        mean_best_direction_ap=("best_direction_ap", "mean"),
        max_best_direction_ap=("best_direction_ap", "max"),
        mean_auc=("auc", "mean"),
    ).sort_values("mean_best_direction_ap", ascending=False)
    .round(4).to_string()
)

print("\n=== ROW-ORDER CHECK ===")
print(order.round(4).to_string(index=False))
print(
    "\nRandom-ranking AP baseline is approximately the evaluation "
    "fraud prevalence, around 0.025 in this experiment."
)
print("\nAudit CSVs saved under authoring/research/.")
print("Research-only; preview datasets were not overwritten.")
