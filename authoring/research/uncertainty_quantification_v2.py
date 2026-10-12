from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Load definitions only. Do not run the generator's main() function.
module = runpy.run_path("authoring/generate_synthetic_fraud_v2.py")
generate_features = module["generate_features"]
calibrate_intercept = module["calibrate_intercept"]
sigmoid = module["sigmoid"]
FEATURES = module["FEATURES"]

SEEDS = [1101, 1202, 1303, 1404, 1505]
N_TRAIN = 25_000
N_EVAL = 17_000
TARGET_PREVALENCE = 0.025
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261012

OUT_DIR = Path("authoring/research")
METRICS_OUT = OUT_DIR / "synthetic_v2_bootstrap_ci_results.csv"
DIFFERENCES_OUT = OUT_DIR / "synthetic_v2_paired_difference_ci.csv"

PERIODS = ["same_regime_control", "shifted_regime"]


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


def simple_rule(x):
    return (
        x["transaction_velocity"]
        / max(x["transaction_velocity"].std(), 1e-12)
        + x["distance_km"]
        / max(x["distance_km"].std(), 1e-12)
    ).to_numpy()


def calculate_metrics(y, scores):
    k = max(1, int(len(y) * 0.05))
    top = np.argpartition(scores, len(scores) - k)[-k:]

    return {
        "AP": float(average_precision_score(y, scores)),
        "AUC": float(roc_auc_score(y, scores)),
        "top5_recall": float(y[top].sum() / max(1, y.sum())),
    }


def percentile_ci(values):
    return np.quantile(values, [0.025, 0.975])


def mean_t_interval(values):
    """Two-sided 95% t interval across independent experimental seeds."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())

    if n < 2:
        return mean, np.nan, np.nan

    standard_error = values.std(ddof=1) / np.sqrt(n)
    critical_value = t.ppf(0.975, df=n - 1)

    return (
        mean,
        float(mean - critical_value * standard_error),
        float(mean + critical_value * standard_error),
    )


def main():
    rng_boot = np.random.default_rng(BOOTSTRAP_SEED)
    methods = [
        "Random ranking",
        "Velocity + distance rule",
        "Logistic regression",
        "Interaction logistic",
        "Histogram gradient boosting",
    ]

    # Store per-seed bootstrap distributions for each metric.
    bootstrap_values = {}
    point_values = {}
    detailed_rows = []

    for seed in SEEDS:
        seed_sequence = np.random.SeedSequence(seed)
        train_ss, control_ss, shift_ss, random_ss = seed_sequence.spawn(4)

        train_rng = np.random.default_rng(train_ss)
        control_rng = np.random.default_rng(control_ss)
        shift_rng = np.random.default_rng(shift_ss)
        random_rng = np.random.default_rng(random_ss)

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

        datasets = {
            "same_regime_control": (
                control_x[FEATURES],
                control_y,
                control_x,
            ),
            "shifted_regime": (
                shift_x[FEATURES],
                shift_y,
                shift_x,
            ),
        }

        scores_by_period = {
            "same_regime_control": {
                "Random ranking": random_rng.random(N_EVAL),
                "Velocity + distance rule": simple_rule(control_x),
            },
            "shifted_regime": {
                "Random ranking": random_rng.random(N_EVAL),
                "Velocity + distance rule": simple_rule(shift_x),
            },
        }

        # Fit each model once per seed and evaluate that same fit in both regimes.
        for name, model in make_models().items():
            model.fit(train_x[FEATURES], train_y)

            for period, (x, _, _) in datasets.items():
                scores_by_period[period][name] = (
                    model.predict_proba(x)[:, 1]
                )

        for period, (_, y, _) in datasets.items():
            # Use one bootstrap sample for all methods in a period.
            # This preserves comparisons between models on the same resampled rows.
            boot_for_period = {
                method: {metric: [] for metric in ("AP", "AUC", "top5_recall")}
                for method in methods
            }

            point_for_period = {}

            for method in methods:
                scores = scores_by_period[period][method]
                point = calculate_metrics(y, scores)
                point_for_period[method] = point
                point_values[(seed, period, method)] = point

            for _ in range(N_BOOTSTRAP):
                indices = rng_boot.integers(0, len(y), size=len(y))
                y_b = y[indices]

                # ROC-AUC is undefined if a resample contains one class only.
                # With this dataset size that is rare, but handle it explicitly.
                for method in methods:
                    scores_b = scores_by_period[period][method][indices]

                    if np.unique(y_b).size < 2:
                        boot_for_period[method]["AP"].append(
                            average_precision_score(y_b, scores_b)
                        )
                        boot_for_period[method]["top5_recall"].append(np.nan)
                        boot_for_period[method]["AUC"].append(np.nan)
                    else:
                        m = calculate_metrics(y_b, scores_b)
                        for metric, value in m.items():
                            boot_for_period[method][metric].append(value)

            for method in methods:
                for metric in ("AP", "AUC", "top5_recall"):
                    vals = np.asarray(boot_for_period[method][metric], dtype=float)
                    vals = vals[np.isfinite(vals)]
                    bootstrap_values[(seed, period, method, metric)] = vals

                    lower, upper = percentile_ci(vals)
                    detailed_rows.append({
                        "seed": seed,
                        "period": period,
                        "method": method,
                        "metric": metric,
                        "estimate": point_for_period[method][metric],
                        "bootstrap_ci_lower": float(lower),
                        "bootstrap_ci_upper": float(upper),
                        "bootstrap_replicates_valid": len(vals),
                        "bootstrap_replicates_requested": N_BOOTSTRAP,
                    })

        print(
            f"Seed {seed}: train={train_y.mean():.3%}, "
            f"control={control_y.mean():.3%}, "
            f"shifted={shift_y.mean():.3%}"
        )

    detailed = pd.DataFrame(detailed_rows)
    detailed.to_csv(METRICS_OUT, index=False)

    # Compare AP between regimes using independent transaction resampling
    # within each regime. Aggregate each bootstrap replicate over the five seeds.
    difference_rows = []

    for method in methods:
        control_estimates = np.array([
            point_values[(seed, "same_regime_control", method)]["AP"]
            for seed in SEEDS
        ])
        shift_estimates = np.array([
            point_values[(seed, "shifted_regime", method)]["AP"]
            for seed in SEEDS
        ])
        seed_drops = control_estimates - shift_estimates

        _, seed_ci_lower, seed_ci_upper = mean_t_interval(seed_drops)

        control_boot = np.vstack([
            bootstrap_values[
                (seed, "same_regime_control", method, "AP")
            ][:N_BOOTSTRAP]
            for seed in SEEDS
        ])
        shift_boot = np.vstack([
            bootstrap_values[
                (seed, "shifted_regime", method, "AP")
            ][:N_BOOTSTRAP]
            for seed in SEEDS
        ])

        # Both matrices have one row per seed and one column per bootstrap draw.
        # Draws in the two regimes are independent; the fitted model remains
        # matched by seed when averaging across experimental runs.
        n_reps = min(control_boot.shape[1], shift_boot.shape[1])
        bootstrap_drops = (
            control_boot[:, :n_reps].mean(axis=0)
            - shift_boot[:, :n_reps].mean(axis=0)
        )
        boot_ci_lower, boot_ci_upper = percentile_ci(bootstrap_drops)

        difference_rows.append({
            "method": method,
            "seed_count": len(SEEDS),
            "bootstrap_replicates": n_reps,
            "mean_control_ap": float(control_estimates.mean()),
            "mean_shifted_ap": float(shift_estimates.mean()),
            "mean_ap_drop_control_minus_shifted": float(seed_drops.mean()),
            "transaction_bootstrap_ci_lower": float(boot_ci_lower),
            "transaction_bootstrap_ci_upper": float(boot_ci_upper),
            "seed_t_ci_lower": seed_ci_lower,
            "seed_t_ci_upper": seed_ci_upper,
            "seed_ap_drop_sd": float(seed_drops.std(ddof=1)),
        })

    differences = pd.DataFrame(difference_rows)
    differences.to_csv(DIFFERENCES_OUT, index=False)

    print("\n=== PER-SEED METRIC BOOTSTRAP INTERVALS ===")
    print(
        detailed[
            (detailed["metric"] == "AP")
        ][[
            "seed", "period", "method", "estimate",
            "bootstrap_ci_lower", "bootstrap_ci_upper",
        ]].round(4).to_string(index=False)
    )

    print("\n=== MEAN AP DIFFERENCE: CONTROL MINUS SHIFTED ===")
    print(differences.round(4).to_string(index=False))

    print(f"\nSaved per-seed metric intervals: {METRICS_OUT}")
    print(f"Saved AP difference intervals: {DIFFERENCES_OUT}")
    print("Existing generator and evaluation result files were not modified.")


if __name__ == "__main__":
    main()
