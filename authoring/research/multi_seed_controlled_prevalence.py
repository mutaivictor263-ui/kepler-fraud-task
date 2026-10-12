import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

generator = runpy.run_path("authoring/generate_synthetic_fraud.py")
generate_period = generator["generate_period"]

SEEDS = [101, 202, 303, 404, 505]
FEATURES = [
    "amount_log",
    "merchant_risk",
    "device_novelty",
    "failed_attempts",
    "transaction_velocity",
    "distance_km",
    "account_age_days",
    "hour_of_day",
]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def evaluation_score_base(df):
    return (
        0.1
        + 0.15 * df["amount_log"]
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

records = []

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    train = generate_period(25_000, rng, "training")
    validation = generate_period(8_000, rng, "validation")
    evaluation = generate_period(17_000, rng, "evaluation")

    X_train = train[FEATURES]
    y_train = train["label"]
    X_val = validation[FEATURES]
    y_val = validation["label"]
    X_eval = evaluation[FEATURES]

    # Match expected evaluation prevalence to observed training prevalence.
    target_rate = float(y_train.mean())
    base = evaluation_score_base(evaluation)

    low, high = -12.0, 2.0
    for _ in range(100):
        mid = (low + high) / 2
        if sigmoid(base + mid).mean() < target_rate:
            low = mid
        else:
            high = mid

    eval_prob = sigmoid(base + (low + high) / 2)

    # Use a separate deterministic RNG for controlled evaluation labels.
    label_rng = np.random.default_rng(seed + 900_000)
    y_eval = label_rng.binomial(1, eval_prob)

    print(
        f"\nSeed {seed}: "
        f"train prevalence={y_train.mean():.3%}, "
        f"controlled evaluation prevalence={y_eval.mean():.3%}, "
        f"expected evaluation prevalence={eval_prob.mean():.3%}"
    )

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]
        eval_scores = model.predict_proba(X_eval)[:, 1]

        k = max(1, int(len(evaluation) * 0.05))
        top = np.argsort(eval_scores)[-k:]

        record = {
            "seed": seed,
            "model": name,
            "train_prevalence": y_train.mean(),
            "evaluation_prevalence": y_eval.mean(),
            "expected_evaluation_prevalence": eval_prob.mean(),
            "validation_ap": average_precision_score(y_val, val_scores),
            "evaluation_ap": average_precision_score(y_eval, eval_scores),
            "validation_auc": roc_auc_score(y_val, val_scores),
            "evaluation_auc": roc_auc_score(y_eval, eval_scores),
            "top5_precision": y_eval[top].mean(),
            "top5_recall": y_eval[top].sum() / max(1, y_eval.sum()),
        }
        records.append(record)

        print(
            f"  {name}: "
            f"validation AP={record['validation_ap']:.4f}, "
            f"evaluation AP={record['evaluation_ap']:.4f}, "
            f"evaluation AUC={record['evaluation_auc']:.4f}, "
            f"top-5% recall={record['top5_recall']:.3%}"
        )

results = pd.DataFrame(records)
out = Path("authoring/research/multi_seed_controlled_prevalence_results.csv")
results.to_csv(out, index=False)

summary = results.groupby("model").agg(
    validation_ap_mean=("validation_ap", "mean"),
    evaluation_ap_mean=("evaluation_ap", "mean"),
    evaluation_ap_std=("evaluation_ap", "std"),
    evaluation_auc_mean=("evaluation_auc", "mean"),
    evaluation_auc_std=("evaluation_auc", "std"),
    top5_recall_mean=("top5_recall", "mean"),
    evaluation_prevalence_mean=("evaluation_prevalence", "mean"),
    expected_prevalence_mean=("expected_evaluation_prevalence", "mean"),
)

print("\nCONTROLLED-PREVALENCE SUMMARY")
print(summary.round(4).to_string())
print(f"\nDetailed results saved to: {out}")
print("Research only; existing preview files were not overwritten.")
