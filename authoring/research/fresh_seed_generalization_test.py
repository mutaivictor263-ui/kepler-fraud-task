import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

generate_period = runpy.run_path(
    "authoring/generate_synthetic_fraud.py"
)["generate_period"]

SEEDS = [606, 707, 808, 909, 1001]
FEATURES = [
    "amount_log", "merchant_risk", "device_novelty",
    "failed_attempts", "transaction_velocity", "distance_km",
    "account_age_days", "hour_of_day",
]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def evaluation_base(df):
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


def make_models():
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
    }


rows = []

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    train = generate_period(25_000, rng, "training")
    generate_period(8_000, rng, "validation")  # Preserve generator sequence.
    evaluation = generate_period(17_000, rng, "evaluation")

    X_train = train[FEATURES]
    y_train = train["label"].to_numpy()
    X_eval = evaluation[FEATURES]

    base = evaluation_base(evaluation)
    low, high = -12.0, 2.0
    for _ in range(100):
        mid = (low + high) / 2
        if sigmoid(base + mid).mean() < y_train.mean():
            low = mid
        else:
            high = mid

    probabilities = sigmoid(base + (low + high) / 2)
    y_eval = np.random.default_rng(seed + 900_000).binomial(
        1, probabilities
    )

    # Simple rule: no labels or evaluation outcomes are used to set weights.
    velocity = X_eval["transaction_velocity"].to_numpy()
    distance = X_eval["distance_km"].to_numpy()
    rule_scores = (
        velocity / max(velocity.std(), 1e-12)
        + distance / max(distance.std(), 1e-12)
    )

    candidates = {"Velocity + distance rule": rule_scores}

    for name, model in make_models().items():
        model.fit(X_train, y_train)
        candidates[name] = model.predict_proba(X_eval)[:, 1]

    for name, scores in candidates.items():
        k = max(1, int(len(y_eval) * 0.05))
        top = np.argsort(scores)[-k:]
        rows.append({
            "seed": seed,
            "method": name,
            "evaluation_prevalence": y_eval.mean(),
            "AP": average_precision_score(y_eval, scores),
            "AUC": roc_auc_score(y_eval, scores),
            "top5_recall": y_eval[top].sum() / max(1, y_eval.sum()),
        })

    print(
        f"Seed {seed}: train prevalence={y_train.mean():.3%}; "
        f"evaluation prevalence={y_eval.mean():.3%}"
    )

results = pd.DataFrame(rows)
out = Path("authoring/research/fresh_seed_generalization_results.csv")
results.to_csv(out, index=False)

print("\nFRESH-SEED GENERALIZATION SUMMARY")
print(
    results.groupby("method").agg(
        AP_mean=("AP", "mean"),
        AP_std=("AP", "std"),
        AUC_mean=("AUC", "mean"),
        top5_recall_mean=("top5_recall", "mean"),
    ).round(4).to_string()
)
print(f"\nDetailed results saved to: {out}")
print("Research only; existing files were not overwritten.")
