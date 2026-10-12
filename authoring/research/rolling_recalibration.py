
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "environment/data/creditcard.csv"
TARGET_PRECISION = 0.30

df = pd.read_csv(DATA_PATH)
df = df.sort_values("Time").reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"]
n = len(df)

# Each tuple: training end, calibration start/end, evaluation end.
# All boundaries are chronological proportions of the dataset.
ROUNDS = [
    ("Round 1", 0.50, 0.50, 0.60, 0.70),
    ("Round 2", 0.60, 0.60, 0.70, 0.80),
]

def build_models():
    return [
        (
            "Logistic Regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]


def select_threshold(y_cal, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_cal, probabilities
    )

    eligible = np.flatnonzero(
        precision[:-1] >= TARGET_PRECISION
    )

    if len(eligible):
        best_idx = eligible[
            np.argmax(recall[:-1][eligible])
        ]
        return float(thresholds[best_idx]), "30% precision target"

    # Fallback is explicitly marked; it does not meet the target.
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), "Fallback: highest calibration F1"


results = []

print("ROLLING RETRAINING AND RECALIBRATION")
print("Final 20% remains untouched.")
print(f"Target calibration precision: {TARGET_PRECISION:.0%}")

for round_name, train_frac, cal_start_frac, cal_end_frac, eval_end_frac in ROUNDS:
    train_end = int(n * train_frac)
    cal_start = int(n * cal_start_frac)
    cal_end = int(n * cal_end_frac)
    eval_end = int(n * eval_end_frac)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_cal, y_cal = X.iloc[cal_start:cal_end], y.iloc[cal_start:cal_end]
    X_eval, y_eval = X.iloc[cal_end:eval_end], y.iloc[cal_end:eval_end]

    print("\n" + "=" * 75)
    print(round_name)
    print(f"Training transactions: {len(X_train):,}")
    print(f"Calibration transactions: {len(X_cal):,}")
    print(f"Evaluation transactions: {len(X_eval):,}")
    print(f"Training fraud cases: {int(y_train.sum())}")
    print(f"Calibration fraud cases: {int(y_cal.sum())}")
    print(f"Evaluation fraud cases: {int(y_eval.sum())}")

    for model_name, model in build_models():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_seconds = time.time() - start_time

        # Choose threshold from calibration period only.
        cal_prob = model.predict_proba(X_cal)[:, 1]
        threshold, selection_rule = select_threshold(y_cal, cal_prob)

        # Evaluate once on the next chronological period.
        eval_prob = model.predict_proba(X_eval)[:, 1]
        eval_pred = (eval_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_eval, eval_pred, labels=[0, 1]
        ).ravel()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        ap = average_precision_score(y_eval, eval_prob)
        auc = roc_auc_score(y_eval, eval_prob)

        results.append({
            "Round": round_name,
            "Model": model_name,
            "Training fraud cases": int(y_train.sum()),
            "Calibration fraud cases": int(y_cal.sum()),
            "Evaluation fraud cases": int(y_eval.sum()),
            "Threshold": threshold,
            "Selection rule": selection_rule,
            "Precision": precision,
            "Recall": recall,
            "True positives": int(tp),
            "False positives": int(fp),
            "False negatives": int(fn),
            "Average precision": ap,
            "ROC-AUC": auc,
            "Training seconds": training_seconds,
        })

        print(f"\n{model_name}")
        print(f"Threshold: {threshold:.6f} ({selection_rule})")
        print(f"Evaluation precision: {precision:.1%}")
        print(f"Evaluation recall: {recall:.1%}")
        print(f"Fraud caught: {tp}/{int(y_eval.sum())}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"Average precision: {ap:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Training time: {training_seconds:.1f} seconds")

results_df = pd.DataFrame(results)
results_df.to_csv("rolling_recalibration_results.csv", index=False)

print("\n" + "=" * 100)
print("ROLLING RECALIBRATION RESULTS")
columns = [
    "Round", "Model", "Threshold", "Selection rule",
    "Evaluation fraud cases", "Precision", "Recall",
    "False positives", "False negatives",
    "Average precision", "ROC-AUC",
]

print(results_df[columns].to_string(
    index=False,
    formatters={
        "Threshold": "{:.6f}".format,
        "Precision": "{:.1%}".format,
        "Recall": "{:.1%}".format,
        "Average precision": "{:.4f}".format,
        "ROC-AUC": "{:.4f}".format,
    },
))

# Compare with the previous frozen-threshold experiment if available.
try:
    fixed = pd.read_csv("fixed_threshold_results.csv")

    fixed = fixed.rename(columns={
        "Precision": "Fixed precision",
        "Recall": "Fixed recall",
        "False positives": "Fixed false positives",
        "False negatives": "Fixed false negatives",
        "Average precision": "Fixed average precision",
        "ROC-AUC": "Fixed ROC-AUC",
    })

    rolling = results_df.rename(columns={
        "Precision": "Rolling precision",
        "Recall": "Rolling recall",
        "False positives": "Rolling false positives",
        "False negatives": "Rolling false negatives",
        "Average precision": "Rolling average precision",
        "ROC-AUC": "Rolling ROC-AUC",
    })

    comparison = rolling.merge(
        fixed,
        left_on=["Model", "Round"],
        right_on=["Model", "Period"],
        how="left",
    )

    comparison_columns = [
        "Model", "Round",
        "Fixed precision", "Rolling precision",
        "Fixed recall", "Rolling recall",
        "Fixed false positives", "Rolling false positives",
        "Fixed false negatives", "Rolling false negatives",
    ]

    print("\nFIXED VS ROLLING COMPARISON")
    print(comparison[comparison_columns].to_string(
        index=False,
        formatters={
            "Fixed precision": "{:.1%}".format,
            "Rolling precision": "{:.1%}".format,
            "Fixed recall": "{:.1%}".format,
            "Rolling recall": "{:.1%}".format,
        },
    ))

    comparison.to_csv("fixed_vs_rolling_comparison.csv", index=False)
    print("\nSaved comparison to fixed_vs_rolling_comparison.csv")

except FileNotFoundError:
    print("\nPrevious fixed_threshold_results.csv not found; skipped comparison.")

print("\nSaved rolling results to rolling_recalibration_results.csv")
print("No model or threshold was selected using the final 20%.")
