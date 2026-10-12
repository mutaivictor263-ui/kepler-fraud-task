
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

# Expanding-window chronological validation:
# Fold 1: train 0-50%, validate 50-60%
# Fold 2: train 0-60%, validate 60-70%
# Fold 3: train 0-70%, validate 70-80%
# The final 20% is deliberately excluded.
FOLDS = [
    ("Fold 1", 0.50, 0.60),
    ("Fold 2", 0.60, 0.70),
    ("Fold 3", 0.70, 0.80),
]

df = pd.read_csv(DATA_PATH)
df = df.sort_values("Time").reset_index(drop=True)

X = df.drop(columns=["Class"])
y = df["Class"]

n = len(df)

print("CHRONOLOGICAL WALK-FORWARD STABILITY EVALUATION")
print(f"Total transactions: {n:,}")
print(f"Total fraud cases: {int(y.sum())}")
print("Final 20%: reserved and not used in this experiment.")
print(f"Target validation precision: {TARGET_PRECISION:.0%}")


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


def choose_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    eligible = np.flatnonzero(
        precision[:-1] >= TARGET_PRECISION
    )

    if len(eligible):
        best_idx = eligible[
            np.argmax(recall[:-1][eligible])
        ]
        rule = "30% precision target"
    else:
        f1 = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_idx = int(np.argmax(f1))
        rule = "Fallback: highest F1"

    return float(thresholds[best_idx]), rule


results = []

for fold_name, train_fraction, val_end_fraction in FOLDS:
    train_end = int(n * train_fraction)
    val_end = int(n * val_end_fraction)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    print("\n" + "=" * 75)
    print(f"{fold_name}: {train_fraction:.0%} training / next 10% validation")
    print(f"Training transactions: {len(X_train):,}")
    print(f"Validation transactions: {len(X_val):,}")
    print(f"Training fraud cases: {int(y_train.sum())}")
    print(f"Validation fraud cases: {int(y_val.sum())}")

    if y_train.nunique() < 2 or y_val.nunique() < 2:
        print("Skipping fold: training or validation has only one class.")
        continue

    for model_name, model in build_models():
        print(f"\nTraining {model_name}...")
        start = time.time()

        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_val)[:, 1]
        elapsed = time.time() - start

        threshold, rule = choose_threshold(y_val, probabilities)
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        ap = average_precision_score(y_val, probabilities)
        roc_auc = roc_auc_score(y_val, probabilities)

        results.append({
            "Fold": fold_name,
            "Model": model_name,
            "Training transactions": len(X_train),
            "Validation fraud cases": int(y_val.sum()),
            "Threshold": threshold,
            "Selection rule": rule,
            "Precision": precision,
            "Recall": recall,
            "True positives": int(tp),
            "False positives": int(fp),
            "False negatives": int(fn),
            "Average precision": ap,
            "ROC-AUC": roc_auc,
            "Training seconds": elapsed,
        })

        print(f"Threshold: {threshold:.6f} ({rule})")
        print(f"Precision: {precision:.1%}")
        print(f"Recall: {recall:.1%}")
        print(f"Fraud caught: {tp}/{int(y_val.sum())}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"Average precision: {ap:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Elapsed training time: {elapsed:.1f} seconds")


if not results:
    raise RuntimeError("No folds completed successfully.")

results_df = pd.DataFrame(results)
results_df.to_csv("stability_fold_results.csv", index=False)

print("\n" + "=" * 100)
print("RESULTS BY FOLD")
print(
    results_df[
        [
            "Fold", "Model", "Validation fraud cases",
            "Threshold", "Precision", "Recall",
            "False positives", "False negatives",
            "Average precision", "ROC-AUC",
        ]
    ].to_string(
        index=False,
        formatters={
            "Threshold": "{:.6f}".format,
            "Precision": "{:.1%}".format,
            "Recall": "{:.1%}".format,
            "Average precision": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format,
        },
    )
)

print("\nMEAN METRICS ACROSS FOLDS")
summary = results_df.groupby("Model").agg(
    Mean_precision=("Precision", "mean"),
    Precision_std=("Precision", "std"),
    Mean_recall=("Recall", "mean"),
    Recall_std=("Recall", "std"),
    Mean_average_precision=("Average precision", "mean"),
    Mean_ROC_AUC=("ROC-AUC", "mean"),
    Mean_training_seconds=("Training seconds", "mean"),
)

print(summary.round(4).to_string())

print("\nSaved detailed results to stability_fold_results.csv")
