import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

generate_period = runpy.run_path(
    "authoring/generate_synthetic_fraud.py"
)["generate_period"]

FEATURES = [
    "amount_log", "merchant_risk", "device_novelty",
    "failed_attempts", "transaction_velocity", "distance_km",
    "account_age_days", "hour_of_day",
]

rows = []

for seed in [101, 202, 303, 404, 505]:
    rng = np.random.default_rng(seed)
    train = generate_period(25_000, rng, "training")
    validation = generate_period(8_000, rng, "validation")
    evaluation = generate_period(17_000, rng, "evaluation")

    # These are deliberately simple, transparent candidate rules.
    velocity = evaluation["transaction_velocity"].to_numpy()
    distance = evaluation["distance_km"].to_numpy()

    rules = {
        "Velocity only": velocity,
        "Distance only": distance,
        "Velocity + distance": (
            velocity / (velocity.std() or 1)
            + distance / (distance.std() or 1)
        ),
        "Velocity-distance interaction": (
            (velocity >= 4) & (distance >= 80)
        ).astype(float),
    }

    # Match expected evaluation prevalence to training prevalence.
    base = (
        0.15 * evaluation["amount_log"]
        + 0.45 * evaluation["merchant_risk"]
        + 1.0 * evaluation["device_novelty"]
        + 0.35 * evaluation["failed_attempts"]
        + 0.42 * evaluation["transaction_velocity"]
        + 0.010 * evaluation["distance_km"]
        - 0.0002 * evaluation["account_age_days"]
        + 0.65 * (
            (evaluation["transaction_velocity"] >= 4)
            & (evaluation["distance_km"] >= 80)
        ).astype(float)
    ).to_numpy()

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    low, high = -12.0, 2.0
    for _ in range(100):
        mid = (low + high) / 2
        if sigmoid(base + mid).mean() < train["label"].mean():
            low = mid
        else:
            high = mid

    probs = sigmoid(base + (low + high) / 2)
    y = np.random.default_rng(seed + 900_000).binomial(1, probs)
    k = int(len(y) * 0.05)

    for name, scores in rules.items():
        top = np.argsort(scores)[-k:]
        rows.append({
            "seed": seed,
            "rule": name,
            "AP": average_precision_score(y, scores),
            "AUC": roc_auc_score(y, scores),
            "top5_recall": y[top].sum() / max(1, y.sum()),
        })

results = pd.DataFrame(rows)
out = Path("authoring/research/simple_rule_shortcut_results.csv")
results.to_csv(out, index=False)

print("\nSIMPLE-RULE SHORTCUT TEST")
print(
    results.groupby("rule").agg(
        AP_mean=("AP", "mean"),
        AP_std=("AP", "std"),
        AUC_mean=("AUC", "mean"),
        top5_recall_mean=("top5_recall", "mean"),
    ).round(4).to_string()
)
print(f"\nSaved to {out}")
print("Research only; existing files were not overwritten.")
