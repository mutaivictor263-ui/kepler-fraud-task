from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20261012
OUTPUT_DIR = Path("authoring/research/synthetic_v2_preview")

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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def generate_features(n, rng, period):
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

    # Bounded transformations reduce domination by extreme values.
    amount = (x["amount_log"] - 3.0) / 1.5
    merchant = (x["merchant_risk"] - 0.286) / 0.16
    device = (x["device_novelty"] - 0.231) / 0.18
    failures = np.log1p(x["failed_attempts"])
    velocity = np.log1p(x["transaction_velocity"])
    distance = np.log1p(x["distance_km"]) / 2.0
    age = np.log1p(x["account_age_days"]) / 2.0
    night = ((x["hour_of_day"] <= 5) |
             (x["hour_of_day"] >= 23)).astype(float)

    if period != "evaluation":
        # Historical regime: several signals and interactions matter.
        score = (
            0.65 * amount
            + 0.90 * merchant
            + 0.95 * device
            + 0.55 * failures
            + 0.45 * velocity
            + 0.30 * distance
            - 0.20 * age
            + 0.75 * device * failures
            + 0.60 * merchant * velocity
            + 0.35 * distance * night
            - 0.25 * amount * age
        )
    else:
        # Later regime: importance shifts to different combinations.
        # The evaluation labels remain generated only for research.
        score = (
            0.20 * amount
            + 0.35 * merchant
            + 0.35 * device
            + 0.30 * failures
            + 0.25 * velocity
            + 0.15 * distance
            - 0.10 * age
            + 1.05 * device * velocity
            + 0.95 * failures * night
            + 0.80 * merchant * distance
            - 0.75 * amount * device
            - 0.55 * age * failures
        )

    return x, score.to_numpy()


def calibrate_intercept(scores, target_rate):
    low, high = -20.0, 20.0

    for _ in range(100):
        mid = (low + high) / 2
        if sigmoid(scores + mid).mean() < target_rate:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def generate_labeled(n, rng, period, target_rate=None):
    x, score = generate_features(n, rng, period)

    if target_rate is not None:
        score = score + calibrate_intercept(score, target_rate)

    probabilities = sigmoid(score)
    labels = rng.binomial(1, probabilities)

    # Identifiers are for research bookkeeping, never model features.
    x.insert(
        0,
        "transaction_id",
        [f"{period}_{i:06d}" for i in range(n)],
    )
    x["label"] = labels

    return x


def main():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = generate_labeled(25_000, rng, "training")
    target_rate = train["label"].mean()

    validation = generate_labeled(
        8_000, rng, "validation", target_rate=target_rate
    )
    evaluation = generate_labeled(
        17_000, rng, "evaluation", target_rate=target_rate
    )

    datasets = {
        "train.csv": train,
        "validation.csv": validation,
        "evaluation_with_labels.csv": evaluation,
    }

    for filename, data in datasets.items():
        path = OUTPUT_DIR / filename
        data.to_csv(path, index=False)

        print(
            f"{filename}: rows={len(data):,}, "
            f"fraud={int(data['label'].sum()):,}, "
            f"fraud_rate={data['label'].mean():.3%}, "
            f"size={path.stat().st_size / 1024:.1f} KB"
        )

    print("\nExperimental V2 files:", OUTPUT_DIR)
    print("Research only; not final benchmark data.")
    print("Features:", ", ".join(FEATURES))


if __name__ == "__main__":
    main()
