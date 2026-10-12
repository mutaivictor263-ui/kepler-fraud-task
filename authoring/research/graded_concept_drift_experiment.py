from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


MODULE = runpy.run_path("authoring/generate_synthetic_fraud_v2.py")
generate_features = MODULE["generate_features"]
calibrate_intercept = MODULE["calibrate_intercept"]
sigmoid = MODULE["sigmoid"]
FEATURES = MODULE["FEATURES"]

# Twenty reproducible seeds; five severity levels per seed.
SEEDS = list(range(1101, 1121))
SHIFT_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]

N_TRAIN = 25_000
N_EVAL = 17_000
TARGET_PREVALENCE = 0.025

OUT_DIR = Path("authoring/research")
DETAILS_OUT = OUT_DIR / "graded_concept_drift_results.csv"
SUMMARY_OUT = OUT_DIR / "graded_concept_drift_summary.csv"
PAIRED_OUT = OUT_DIR / "graded_concept_drift_paired_differences.csv"


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
    top = np.argpartition(scores, len(scores) - k)[-k:]

    return {
        "fraud_prevalence": float(y.mean()),
        "AP": float(average_precision_score(y, scores)),
        "AUC": float(roc_auc_score(y, scores)),
        "top5_recall": float(y[top].sum() / max(1, y.sum())),
    }


def simple_rule(x, velocity_scale, distance_scale):
    return (
        x["transaction_velocity"].to_numpy() / velocity_scale
        + x["distance_km"].to_numpy() / distance_scale
    )


def main():
    rows = []

    for seed in SEEDS:
        train_ss, eval_ss, label_ss, random_ss = (
            np.random.SeedSequence(seed).spawn(4)
        )

        train_rng = np.random.default_rng(train_ss)

        # Train entirely on the historical risk formula.
        train_x, train_score = generate_features(
            N_TRAIN, train_rng, "training"
        )
        intercept = calibrate_intercept(
            train_score, TARGET_PREVALENCE
        )
        train_y = train_rng.binomial(
            1, sigmoid(train_score + intercept)
        )
        X_train = train_x[FEATURES]

        # Reinitialize the same feature stream for each formula.
        # This creates identical evaluation features for both formulas.
        control_rng = np.random.default_rng(eval_ss)
        shift_rng = np.random.default_rng(eval_ss)

        control_x, control_score = generate_features(
            N_EVAL, control_rng, "validation"
        )
        shift_x, shift_score = generate_features(
            N_EVAL, shift_rng, "evaluation"
        )

        if not control_x.equals(shift_x):
            raise AssertionError(
                f"Feature rows differ between formulas for seed {seed}"
            )

        X_eval = control_x[FEATURES]

        # Shared uniforms reduce random label noise between shift levels.
        label_rng = np.random.default_rng(label_ss)
        label_uniforms = label_rng.random(N_EVAL)

        # All fitted models see the same training data and evaluation rows.
        candidates = {
            name: model.fit(X_train, train_y).predict_proba(X_eval)[:, 1]
            for name, model in make_models().items()
        }

        random_rng = np.random.default_rng(random_ss)
        candidates["Random ranking"] = random_rng.random(N_EVAL)

        velocity_scale = max(
            float(train_x["transaction_velocity"].std()), 1e-12
        )
        distance_scale = max(
            float(train_x["distance_km"].std()), 1e-12
        )
        candidates["Velocity + distance rule"] = simple_rule(
            control_x, velocity_scale, distance_scale
        )

        for alpha in SHIFT_LEVELS:
            # Linear interpolation between historical and later risk scores.
            risk_score = (
                (1.0 - alpha) * control_score
                + alpha * shift_score
            )
            probabilities = sigmoid(risk_score + intercept)
            y = (label_uniforms < probabilities).astype(np.int8)

            if np.unique(y).size != 2:
                raise ValueError(
                    f"Only one label class at seed={seed}, alpha={alpha}"
                )

            for method, scores in candidates.items():
                rows.append({
                    "seed": seed,
                    "shift_level": alpha,
                    "shift_percent": int(alpha * 100),
                    "method": method,
                    **metrics(y, scores),
                })

        print(
            f"Seed {seed}: training prevalence={train_y.mean():.3%}; "
            "completed 5 shift levels"
        )

    details = pd.DataFrame(rows)
    details.to_csv(DETAILS_OUT, index=False)

    metric_names = ["fraud_prevalence", "AP", "AUC", "top5_recall"]
    summary = (
        details.groupby(["shift_level", "shift_percent", "method"])[
            metric_names
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(SUMMARY_OUT, index=False)

    # Paired seed-level change relative to the historical formula (alpha=0).
    paired_metrics = ["AP", "AUC", "top5_recall", "fraud_prevalence"]
    baseline = details[details["shift_level"] == 0.0].set_index(
        ["seed", "method"]
    )

    paired_rows = []
    for _, row in details[details["shift_level"] > 0.0].iterrows():
        base = baseline.loc[(row["seed"], row["method"])]
        result = {
            "seed": int(row["seed"]),
            "shift_level": float(row["shift_level"]),
            "shift_percent": int(row["shift_percent"]),
            "method": row["method"],
        }
        for metric in paired_metrics:
            result[f"{metric}_change_from_control"] = (
                float(row[metric]) - float(base[metric])
            )
        paired_rows.append(result)

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(PAIRED_OUT, index=False)

    print("\n=== Mean AP by shift level and method ===")
    ap_table = details.pivot_table(
        index="shift_percent",
        columns="method",
        values="AP",
        aggfunc="mean",
    )
    print(ap_table.round(4).to_string())

    print("\n=== Mean fraud prevalence by shift level ===")
    prevalence = (
        details.groupby("shift_percent")["fraud_prevalence"]
        .agg(["mean", "std"])
    )
    print(prevalence.round(4).to_string())

    print("\n=== Output files ===")
    print(DETAILS_OUT)
    print(SUMMARY_OUT)
    print(PAIRED_OUT)
    print("Existing V2 baseline files were not modified.")


if __name__ == "__main__":
    main()
