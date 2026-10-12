
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = Path("environment/data/creditcard.csv")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PRECISION = 0.30
RANDOM_STATE = 42

# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = {"Time", "Class"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain these columns: {required_columns}"
    )

if df["Class"].isna().any():
    raise ValueError("The target column contains missing values.")

# Preserve chronological order.
df = df.sort_values("Time").reset_index(drop=True)

X = df.drop(columns=["Class", "Time"])
y = df["Class"].astype(int)

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Class must contain only 0 and 1.")

# --------------------------------------------------
# 3. Chronological split: 70% / 10% / 20%
# --------------------------------------------------

n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.80)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = (
    X.iloc[train_end:validation_end],
    y.iloc[train_end:validation_end],
)
X_test, y_test = X.iloc[validation_end:], y.iloc[validation_end:]

print("=" * 65)
print("FINAL FROZEN FRAUD DETECTION EVALUATION")
print("=" * 65)

print(f"Total transactions: {n:,}")
print(
    f"Training:   {len(X_train):,} transactions | "
    f"{int(y_train.sum())} fraud cases"
)
print(
    f"Validation: {len(X_val):,} transactions | "
    f"{int(y_val.sum())} fraud cases"
)
print(
    f"Evaluation: {len(X_test):,} transactions | "
    f"{int(y_test.sum())} fraud cases"
)

# --------------------------------------------------
# 4. Train Logistic Regression on training data only
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
])

start_time = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - start_time

print(f"\nTraining time: {training_seconds:.2f} seconds")

# --------------------------------------------------
# 5. Select the operating threshold on validation only
# --------------------------------------------------

val_probabilities = model.predict_proba(X_val)[:, 1]

val_precision, val_recall, thresholds = precision_recall_curve(
    y_val, val_probabilities
)

# The final precision/recall entries have no matching threshold.
candidate_precision = val_precision[:-1]
candidate_recall = val_recall[:-1]

eligible = np.flatnonzero(
    candidate_precision >= TARGET_PRECISION
)

if len(eligible) > 0:
    # Maximize recall while meeting the validation precision target.
    # Break recall ties by choosing higher precision.
    best_index = max(
        eligible,
        key=lambda i: (
            candidate_recall[i],
            candidate_precision[i],
        ),
    )
    threshold = float(thresholds[best_index])
    threshold_method = (
        f"Highest validation recall with precision >= "
        f"{TARGET_PRECISION:.0%}"
    )
else:
    # Fallback when the validation data cannot meet the target.
    candidate_f1 = (
        2 * candidate_precision * candidate_recall
        / (candidate_precision + candidate_recall + 1e-12)
    )
    best_index = int(np.argmax(candidate_f1))
    threshold = float(thresholds[best_index])
    threshold_method = "Fallback: highest validation F1"

validation_predictions = (
    val_probabilities >= threshold
).astype(int)

print("\nThreshold selection (validation data)")
print(f"Method: {threshold_method}")
print(f"Selected threshold: {threshold:.6f}")
print(
    f"Validation precision: "
    f"{precision_score(y_val, validation_predictions, zero_division=0):.4f}"
)
print(
    f"Validation recall:    "
    f"{recall_score(y_val, validation_predictions, zero_division=0):.4f}"
)
print(
    f"Validation F1:        "
    f"{f1_score(y_val, validation_predictions, zero_division=0):.4f}"
)

# --------------------------------------------------
# 6. Freeze the model and threshold; evaluate once
# --------------------------------------------------

# Do not retrain or change the threshold after inspecting
# these evaluation results.
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

precision = precision_score(
    y_test, test_predictions, zero_division=0
)
recall = recall_score(
    y_test, test_predictions, zero_division=0
)
f1 = f1_score(
    y_test, test_predictions, zero_division=0
)
average_precision = average_precision_score(
    y_test, test_probabilities
)
roc_auc = roc_auc_score(y_test, test_probabilities)

flagged = int(test_predictions.sum())
fraud_count = int(y_test.sum())
prevalence = float(y_test.mean())

print("\n" + "=" * 65)
print("FROZEN EVALUATION RESULTS")
print("=" * 65)

print(f"Frozen threshold:       {threshold:.6f}")
print(f"Transactions evaluated: {len(y_test):,}")
print(f"Actual fraud cases:     {fraud_count}")
print(f"Fraud prevalence:       {prevalence:.4%}")
print(f"Transactions flagged:   {flagged:,}")
print(f"True positives:         {tp}")
print(f"False positives:        {fp}")
print(f"False negatives:        {fn}")
print(f"True negatives:         {tn}")
print(f"Precision:              {precision:.4%}")
print(f"Recall:                 {recall:.4%}")
print(f"F1 score:               {f1:.4f}")
print(f"Average precision:      {average_precision:.4f}")
print(f"ROC-AUC:                {roc_auc:.4f}")

print("\nConfusion matrix (rows = actual, columns = predicted)")
print("                 Predicted legitimate | Predicted fraud")
print(f"Actual legitimate       {tn:>8,}       | {fp:>8,}")
print(f"Actual fraud            {fn:>8,}       | {tp:>8,}")

# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

results = pd.DataFrame([{
    "model": "Logistic Regression",
    "threshold": threshold,
    "threshold_method": threshold_method,
    "target_validation_precision": TARGET_PRECISION,
    "train_transactions": len(X_train),
    "validation_transactions": len(X_val),
    "evaluation_transactions": len(X_test),
    "evaluation_fraud_cases": fraud_count,
    "fraud_prevalence": prevalence,
    "transactions_flagged": flagged,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "true_negatives": tn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "average_precision": average_precision,
    "roc_auc": roc_auc,
    "training_seconds": training_seconds,
    "evaluation_note": (
        "Final 20% previously inspected in earlier experiments; "
        "not an untouched independent test set."
    ),
}])

output_path = OUTPUT_DIR / "final_frozen_evaluation.csv"
results.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
print("\nEvaluation complete.")
