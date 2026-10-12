import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

module = runpy.run_path("authoring/generate_synthetic_fraud_v2.py")
generate_features = module["generate_features"]
calibrate_intercept = module["calibrate_intercept"]
sigmoid = module["sigmoid"]

SEEDS = [1101, 1202, 1303, 1404, 1505]
FEATURES = module["FEATURES"]


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
        "AP": average_precision_score(y, scores),
        "AUC": roc_auc_score(y, scores),
        "top5_recall": y[top].sum() / max(1, y.sum()),
    }


rows = []

for seed in SEEDS:
    rng = np.random.default_rng(seed)

    train_x, train_score = generate_features(25_000, rng, "training")
    intercept = calibrate_intercept(train_score, 0.025)
    train_y = rng.binomial(1, sigmoid(train_score + intercept))

    val_x, val_score = generate_features(8_000, rng, "validation")
    val_y = rng.binomial(1, sigmoid(val_score + intercept))

    eval_x, eval_score = generate_features(17_000, rng, "evaluation")
    eval_y = rng.binomial(1, sigmoid(eval_score + intercept))

    X_train = train_x[FEATURES]
    X_val = val_x[FEATURES]
    X_eval = eval_x[FEATURES]

    # Simple rule uses no labels and no fitted weights.
    rule_val = (
        X_val["transaction_velocity"] /
        max(X_val["transaction_velocity"].std(), 1e-12)
        + X_val["distance_km"] /
        max(X_val["distance_km"].std(), 1e-12)
    )
    rule_eval = (
        X_eval["transaction_velocity"] /
        max(X_eval["transaction_velocity"].std(), 1e-12)
        + X_eval["distance_km"] /
        max(X_eval["distance_km"].std(), 1e-12)
    )

    rng_rank = np.random.default_rng(seed + 700_000)
    random_val = rng_rank.random(len(val_y))
    random_eval = rng_rank.random(len(eval_y))

    candidates = {
        "Random ranking": (random_val, random_eval),
        "Velocity + distance rule": (rule_val, rule_eval),
    }

    for name, model in make_models().items():
        model.fit(X_train, train_y)
        candidates[name] = (
            model.predict_proba(X_val)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_val),
            model.predict_proba(X_eval)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_eval),
        )

    for name, (val_scores, eval_scores) in candidates.items():
        for period, y, scores in [
            ("validation", val_y, val_scores),
            ("evaluation", eval_y, eval_scores),
        ]:
            rows.append({
                "seed": seed,
                "method": name,
                "period": period,
                "fraud_prevalence": y.mean(),
                **metrics(y, np.asarray(scores)),
            })

    print(
        f"Seed {seed}: train={train_y.mean():.3%}, "
        f"validation={val_y.mean():.3%}, "
        f"evaluation={eval_y.mean():.3%}"
    )

results = pd.DataFrame(rows)
out = Path("authoring/research/synthetic_v2_evaluation_results.csv")
results.to_csv(out, index=False)

print("\nMEAN RESULTS ACROSS FIVE SEEDS")
summary = results.groupby(["period", "method"])[
    ["fraud_prevalence", "AP", "AUC", "top5_recall"]
].mean().round(4)
print(summary.to_string())

print("\nSTANDARD DEVIATION OF AVERAGE PRECISION")
print(
    results.groupby(["period", "method"])["AP"]
    .std().round(4).to_string()
)

print(f"\nDetailed results saved to: {out}")
