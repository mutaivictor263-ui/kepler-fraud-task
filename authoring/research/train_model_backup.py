
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load and sort the data chronologically
df = pd.read_csv("environment/data/creditcard.csv")
df = df.sort_values("Time").reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"]

# Chronological 70% / 10% / 20% split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

print("Fraud cases:")
print("Training:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Testing:", int(y_test.sum()))

# Train the baseline model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
)

model.fit(X_train, y_train)

# Select threshold using validation data only
val_prob = model.predict_proba(X_val)[:, 1]


precision, recall, thresholds = precision_recall_curve(y_val, val_prob)

# Compare operating points using validation data only.
target_precisions = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

rows = []

for target_precision in target_precisions:
    valid = np.flatnonzero(precision[:-1] >= target_precision)

    if len(valid) == 0:
        continue

    # Among thresholds meeting the target precision,
    # choose the one with the highest validation recall.
    best_idx = valid[np.argmax(recall[:-1][valid])]
    candidate_threshold = float(thresholds[best_idx])

    val_pred = (val_prob >= candidate_threshold).astype(int)

    tn = int(((y_val == 0) & (val_pred == 0)).sum())
    fp = int(((y_val == 0) & (val_pred == 1)).sum())
    fn = int(((y_val == 1) & (val_pred == 0)).sum())
    tp = int(((y_val == 1) & (val_pred == 1)).sum())

    actual_precision = tp / (tp + fp) if tp + fp else 0.0
    actual_recall = tp / (tp + fn) if tp + fn else 0.0

    rows.append({
        "Minimum precision": target_precision,
        "Threshold": candidate_threshold,
        "Actual precision": actual_precision,
        "Recall": actual_recall,
        "False positives": fp,
        "False negatives": fn,
        "Flagged transactions": tp + fp,
    })

tradeoff = pd.DataFrame(rows)

print("\nVALIDATION THRESHOLD COMPARISON")
if not tradeoff.empty:
    print(
        tradeoff.to_string(
            index=False,
            formatters={
                "Minimum precision": "{:.0%}".format,
                "Threshold": "{:.6f}".format,
                "Actual precision": "{:.1%}".format,
                "Recall": "{:.1%}".format,
            },
        )
    )
else:
    print("No threshold met any of the target precision levels.")

# Keep the preselected rule: highest recall at validation
# precision of at least 20%; otherwise use highest validation F1.
valid = np.flatnonzero(precision[:-1] >= 0.20)

if len(valid):
    best_idx = valid[np.argmax(recall[:-1][valid])]
    threshold = float(thresholds[best_idx])
    selection_rule = "highest recall with validation precision >= 20%"
else:
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_idx = int(np.argmax(f1))
    threshold = float(thresholds[best_idx])
    selection_rule = "highest validation F1 (20% precision not reached)"

print("\nSelected threshold:", round(threshold, 6))
print("Selection rule:", selection_rule)

