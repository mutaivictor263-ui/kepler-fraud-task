
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

TARGET_PRECISION = 0.30

df = pd.read_csv("environment/data/creditcard.csv")
df = df.sort_values("Time").reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"]
n = len(df)

# Initial training: first 50%.
train_end = int(n * 0.50)

# Threshold calibration: next 10%.
cal_end = int(n * 0.60)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_cal, y_cal = X.iloc[train_end:cal_end], y.iloc[train_end:cal_end]

# Two later evaluation periods; no threshold changes.
periods = [
    ("Future period 1 (60-70%)", int(n * 0.60), int(n * 0.70)),
    ("Future period 2 (70-80%)", int(n * 0.70), int(n * 0.80)),
]

models = [
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

results = []

for name, model in models:
    print(f"\n{'=' * 70}\n{name}")

    start = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - start

    # Select threshold on the calibration period only.
    cal_prob = model.predict_proba(X_cal)[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y_cal, cal_prob
    )

    eligible = np.flatnonzero(
        precision[:-1] >= TARGET_PRECISION
    )

    if len(eligible):
        best_idx = eligible[
            np.argmax(recall[:-1][eligible])
        ]
        threshold = float(thresholds[best_idx])
        rule = "30% precision target"
    else:
        f1 = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_idx = int(np.argmax(f1))
        threshold = float(thresholds[best_idx])
        rule = "Fallback: highest calibration F1"

    print(f"Calibration threshold: {threshold:.6f}")
    print(f"Threshold rule: {rule}")
    print(f"Training time: {fit_seconds:.1f} seconds")

    # Apply the same model and threshold to each later period.
    for period_name, start_idx, end_idx in periods:
        X_period = X.iloc[start_idx:end_idx]
        y_period = y.iloc[start_idx:end_idx]

        probabilities = model.predict_proba(X_period)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_period, predictions, labels=[0, 1]
        ).ravel()

        actual_precision = tp / (tp + fp) if tp + fp else 0.0
        actual_recall = tp / (tp + fn) if tp + fn else 0.0

        ap = average_precision_score(y_period, probabilities)
        auc = roc_auc_score(y_period, probabilities)

        results.append({
            "Model": name,
            "Period": period_name,
            "Fixed threshold": threshold,
            "Fraud cases": int(y_period.sum()),
            "Precision": actual_precision,
            "Recall": actual_recall,
            "True positives": int(tp),
            "False positives": int(fp),
            "False negatives": int(fn),
            "Average precision": ap,
            "ROC-AUC": auc,
        })

        print(f"\n{period_name}")
        print(f"Fraud cases: {int(y_period.sum())}")
        print(f"Precision: {actual_precision:.1%}")
        print(f"Recall: {actual_recall:.1%}")
        print(f"Fraud caught: {tp}/{int(y_period.sum())}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"Average precision: {ap:.4f}")
        print(f"ROC-AUC: {auc:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("fixed_threshold_results.csv", index=False)

print("\nFIXED-THRESHOLD RESULTS")
print(results_df.to_string(
    index=False,
    formatters={
        "Fixed threshold": "{:.6f}".format,
        "Precision": "{:.1%}".format,
        "Recall": "{:.1%}".format,
        "Average precision": "{:.4f}".format,
        "ROC-AUC": "{:.4f}".format,
    },
))

print("\nSaved to fixed_threshold_results.csv")
print("The final 20% of transactions was not used.")
