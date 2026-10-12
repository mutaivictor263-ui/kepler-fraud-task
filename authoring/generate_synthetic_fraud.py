from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20261012
OUTPUT_DIR = Path("authoring/research/synthetic_preview")

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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_period(n, rng, period):
    """Generate transaction features and labels for one time period."""
    x = pd.DataFrame({
        "amount_log": rng.normal(3.0, 1.1, n).clip(0, 9),
        "merchant_risk": rng.beta(2, 5, n),
        "device_novelty": rng.beta(1.2, 4.0, n),
        "failed_attempts": rng.poisson(0.25, n).clip(0, 6),
        "transaction_velocity": rng.poisson(1.5, n).clip(0, 12),
        "distance_km": rng.exponential(35, n).clip(0, 1500),
        "account_age_days": rng.exponential(700, n).clip(1, 5000),
        "hour_of_day": rng.integers(0, 24, n),
    })

    # A learnable baseline fraud mechanism.
    score = (
        -6.0
        + 0.25 * x["amount_log"]
        + 1.5 * x["merchant_risk"]
        + 2.3 * x["device_novelty"]
        + 0.7 * x["failed_attempts"]
        + 0.20 * x["transaction_velocity"]
        + 0.006 * x["distance_km"]
        - 0.0005 * x["account_age_days"]
    )

    if period == "evaluation":
        # Temporal drift: some familiar signals weaken, while other
        # combinations become more important.
        score = (
            -5.9
            + 0.15 * x["amount_log"]
            + 0.45 * x["merchant_risk"]
            + 1.0 * x["device_novelty"]
            + 0.35 * x["failed_attempts"]
            + 0.42 * x["transaction_velocity"]
            + 0.010 * x["distance_km"]
            - 0.0002 * x["account_age_days"]
            + 0.65
            * (
                (x["transaction_velocity"] >= 4)
                & (x["distance_km"] >= 80)
            ).astype(float)
        )

    probabilities = sigmoid(score)
    labels = rng.binomial(1, probabilities)

    x.insert(0, "transaction_id", [
        f"{period}_{i:06d}" for i in range(n)
    ])
    x["label"] = labels
    return x


def main():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    periods = {
        "train.csv": ("training", 25_000),
        "validation.csv": ("validation", 8_000),
        "evaluation_with_labels.csv": ("evaluation", 17_000),
    }

    for filename, (period, n) in periods.items():
        data = generate_period(n, rng, period)
        path = OUTPUT_DIR / filename
        data.to_csv(path, index=False)

        print(
            f"{filename}: rows={len(data):,}, "
            f"fraud={int(data['label'].sum()):,}, "
            f"fraud_rate={data['label'].mean():.3%}, "
            f"size={path.stat().st_size / 1024:.1f} KB"
        )

    print("\nPreview files created in:", OUTPUT_DIR)
    print("These are research files, not final benchmark files.")


if __name__ == "__main__":
    main()
