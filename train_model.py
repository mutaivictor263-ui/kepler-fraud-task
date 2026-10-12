
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "environment/data/creditcard.csv"
TARGET_PRECISION = 0.30

# Load data and preserve chronological order.
df = pd.read_csv(DATA_PATH)
df = df.sort_values("Time").reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"]

# Chronological split: 70% training, 10% validation, 20% reserved test.
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

print("Data split:")
print(f"Training transactions: {len(X_train):,}")
print(f"Validation transactions: {len(X_val):,}")
print(f"Training fraud cases: {int(y_train.sum())}")
print(f"Validation fraud cases: {int(y_val.sum())}")
print("Test set: reserved and not evaluated in this experiment.")


def evaluate_model(name, model):
    """Train a model and select its threshold using validation data only."""

    print(f"\n{'=' * 60}")
    print(f"Training {name}...")
    start = time.time()

    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]

    elapsed = time.time() - start

    precision, recall, thresholds = precision_recall_curve(
        y_val, val_prob
    )

    # Select the highest-recall threshold with validation precision >= 30%.
    eligible = np.flatnonzero(
        precision[:-1] >= TARGET_PRECISION
    )

    if len(eligible):
        best_idx = eligible[
            np.argmax(recall[:-1][eligible])
        ]
        selection_rule = "highest recall with precision >= 30%"
    else:
        # Fallback if no threshold meets the target.
        f1 = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_idx = int(np.argmax(f1))
        selection_rule = "fallback: highest validation F1"

    threshold = float(thresholds[best_idx])
    val_pred = (val_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, val_pred, labels=[0, 1]
    ).ravel()

    actual_precision = tp / (tp + fp) if tp + fp else 0.0
    actual_recall = tp / (tp + fn) if tp + fn else 0.0

    ap = average_precision_score(y_val, val_prob)
    roc_auc = roc_auc_score(y_val, val_prob)

    results = {
        "Model": name,
        "Threshold": threshold,
        "Precision": actual_precision,
        "Recall": actual_recall,
        "False positives": int(fp),
        "False negatives": int(fn),
        "Fraud caught": int(tp),
        "Transactions flagged": int(tp + fp),
        "Average precision": ap,
        "ROC-AUC": roc_auc,
        "Training seconds": elapsed,
        "Selection rule": selection_rule,
    }

    print(f"Selection rule: {selection_rule}")
    print(f"Selected threshold: {threshold:.6f}")
    print(f"Validation precision: {actual_precision:.1%}")
    print(f"Validation recall: {actual_recall:.1%}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Fraud cases caught: {tp} of {int(y_val.sum())}")
    print(f"Average precision: {ap:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Training time: {elapsed:.1f} seconds")
    print("Validation confusion matrix:")
    print(confusion_matrix(y_val, val_pred, labels=[0, 1]))
    print("\nValidation classification report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    return results


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
    results.append(evaluate_model(name, model))

comparison = pd.DataFrame(results)

print("\n" + "=" * 80)
print("MODEL COMPARISON — VALIDATION SET ONLY")

columns = [
    "Model",
    "Threshold",
    "Precision",
    "Recall",
    "False positives",
    "False negatives",
    "Fraud caught",
    "Transactions flagged",
    "Average precision",
    "ROC-AUC",
    "Training seconds",
]

print(
    comparison[columns].to_string(
        index=False,
        formatters={
            "Threshold": "{:.6f}".format,
            "Precision": "{:.1%}".format,
            "Recall": "{:.1%}".format,
            "Average precision": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format,
            "Training seconds": "{:.1f}".format,
        },
    )
)

print("\nComparison saved to validation_model_comparison.csv")
comparison[columns].to_csv(

    "validation_model_comparison.csv",
    index=False,
)
