from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import average_precision_score, roc_auc_score


MODULE = runpy.run_path("authoring/generate_synthetic_fraud_v2.py")
generate_features = MODULE["generate_features"]
calibrate_intercept = MODULE["calibrate_intercept"]
sigmoid = MODULE["sigmoid"]
FEATURES = MODULE["FEATURES"]

SEEDS = [1101, 1202, 1303, 1404, 1505]
N_TRAIN = 25_000
N_EVAL = 17_000
TARGET_PREVALENCE = 0.025

OUT_DIR = Path("authoring/research")
DETAILS_OUT = OUT_DIR / "fixed_scaling_rule_sensitivity.csv"
SUMMARY_OUT = OUT_DIR / "fixed_scaling_rule_sensitivity_summary.csv"


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


def mean_t_interval(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())

    if n < 2:
        return mean, np.nan, np.nan

    se = values.std(ddof=1) / np.sqrt(n)
    critical = t.ppf(0.975, df=n - 1)
    return mean, float(mean - critical * se), float(mean + critical * se)


def main():
    rows = []

    for seed in SEEDS:
        train_ss, control_ss, shift_ss, _ = (
            np.random.SeedSequence(seed).spawn(4)
        )
        train_rng = np.random.default_rng(train_ss)
        control_rng = np.random.default_rng(control_ss)
        shift_rng = np.random.default_rng(shift_ss)

        train_x, train_score = generate_features(
            N_TRAIN, train_rng, "training"
        )
        intercept = calibrate_intercept(
            train_score, TARGET_PREVALENCE
        )
        train_y = train_rng.binomial(
            1, sigmoid(train_score + intercept)
        )

        control_x, control_score = generate_features(
            N_EVAL, control_rng, "validation"
        )
        control_y = control_rng.binomial(
            1, sigmoid(control_score + intercept)
        )

        shift_x, shift_score = generate_features(
            N_EVAL, shift_rng, "evaluation"
        )
        shift_y = shift_rng.binomial(
            1, sigmoid(shift_score + intercept)
        )

        train_features = train_x[FEATURES]
        control_features = control_x[FEATURES]
        shift_features = shift_x[FEATURES]

        # Training-only scales are reused in both evaluation regimes.
        velocity_scale = max(
            float(train_features["transaction_velocity"].std()), 1e-12
        )
        distance_scale = max(
            float(train_features["distance_km"].std()), 1e-12
        )

        datasets = {
            "same_regime_control": (control_features, control_y),
            "shifted_regime": (shift_features, shift_y),
        }

        for period, (x, y) in datasets.items():
            # Original rule: each evaluation dataset supplies its own scales.
            original_scores = simple_rule(
                x,
                max(float(x["transaction_velocity"].std()), 1e-12),
                max(float(x["distance_km"].std()), 1e-12),
            )

            # Sensitivity rule: both scales come from training data.
            fixed_scores = simple_rule(
                x, velocity_scale, distance_scale
            )

            for version, scores in [
                ("original_per_dataset_scaling", original_scores),
                ("fixed_training_scaling", fixed_scores),
            ]:
                rows.append({
                    "seed": seed,
                    "period": period,
                    "scaling": version,
                    "velocity_scale": (
                        velocity_scale if version == "fixed_training_scaling"
                        else float(x["transaction_velocity"].std())
                    ),
                    "distance_scale": (
                        distance_scale if version == "fixed_training_scaling"
                        else float(x["distance_km"].std())
                    ),
                    **metrics(y, scores),
                })

        print(
            f"Seed {seed}: train={train_y.mean():.3%}, "
            f"control={control_y.mean():.3%}, "
            f"shifted={shift_y.mean():.3%}"
        )

    details = pd.DataFrame(rows)
    details.to_csv(DETAILS_OUT, index=False)

    summary_rows = []
    for scaling in [
        "original_per_dataset_scaling",
        "fixed_training_scaling",
    ]:
        for metric in ["AP", "AUC", "top5_recall"]:
            control = (
                details[
                    (details["scaling"] == scaling)
                    & (details["period"] == "same_regime_control")
                ].set_index("seed")[metric].sort_index()
            )
            shifted = (
                details[
                    (details["scaling"] == scaling)
                    & (details["period"] == "shifted_regime")
                ].set_index("seed")[metric].sort_index()
            )

            # Positive drop means lower performance in the shifted regime.
            drops = control.to_numpy() - shifted.to_numpy()
            mean_drop, ci_low, ci_high = mean_t_interval(drops)

            summary_rows.append({
                "scaling": scaling,
                "metric": metric,
                "seed_count": len(drops),
                "mean_control": float(control.mean()),
                "mean_shifted": float(shifted.mean()),
                "mean_drop_control_minus_shifted": mean_drop,
                "seed_t_ci_lower": ci_low,
                "seed_t_ci_upper": ci_high,
                "seed_drop_sd": float(drops.std(ddof=1)),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\n=== Sensitivity summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nDetailed results: {DETAILS_OUT}")
    print(f"Summary results:  {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
