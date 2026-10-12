from pathlib import Path
import runpy

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

    X_train, y_train = train[FEATURES], train["label"]
    X_val, y_val = validation[FEATURES], validation["label"]
    X_eval, y_eval = evaluation[FEATURES], evaluation["label"]

    print(
        f"\nSeed {seed}: "
        f"train fraud={y_train.mean():.3%}, "
        f"validation fraud={y_val.mean():.3%}, "
        f"evaluation fraud={y_eval.mean():.3%}"
    )

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]
        eval_scores = model.predict_proba(X_eval)[:, 1]

        record = {
            "seed": seed,
            "model": name,
            "train_prevalence": y_train.mean(),
            "validation_prevalence": y_val.mean(),
            "evaluation_prevalence": y_eval.mean(),
            "validation_ap": average_precision_score(y_val, val_scores),
            "evaluation_ap": average_precision_score(y_eval, eval_scores),
            "validation_auc": roc_auc_score(y_val, val_scores),
            "evaluation_auc": roc_auc_score(y_eval, eval_scores),
        }

        k = max(1, int(len(evaluation) * 0.05))
        top = np.argsort(eval_scores)[-k:]
        record["top5_precision"] = y_eval.iloc[top].mean()
        record["top5_recall"] = y_eval.iloc[top].sum() / max(1, y_eval.sum())
        records.append(record)

        print(
            f"  {name}: "
            f"validation AP={record['validation_ap']:.4f}, "
            f"evaluation AP={record['evaluation_ap']:.4f}, "
            f"evaluation AUC={record['evaluation_auc']:.4f}, "
            f"top-5% recall={record['top5_recall']:.3%}"
        )

results = pd.DataFrame(records)
out = Path("authoring/research/multi_seed_results.csv")
results.to_csv(out, index=False)

summary = results.groupby("model").agg(
    evaluation_ap_mean=("evaluation_ap", "mean"),
    evaluation_ap_std=("evaluation_ap", "std"),
    evaluation_auc_mean=("evaluation_auc", "mean"),
    evaluation_auc_std=("evaluation_auc", "std"),
    top5_recall_mean=("top5_recall", "mean"),
    top5_recall_std=("top5_recall", "std"),
    validation_ap_mean=("validation_ap", "mean"),
    evaluation_prevalence_mean=("evaluation_prevalence", "mean"),
)

print("\nSUMMARY ACROSS FIVE SEEDS")
print(summary.round(4).to_string())
print(f"\nDetailed results saved to: {out}")
print("Research only; no original preview files were overwritten.")
