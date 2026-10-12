# Synthetic Fraud Detection V2: Uncertainty Analysis

## 1. Objective

This analysis examines uncertainty in the V2 synthetic fraud-detection experiment. It supplements the existing evaluation and regime-shift diagnostic results; it does not replace them.

The primary question is whether average precision (AP) changes when methods trained under the historical regime are evaluated under a shifted fraud-risk regime, and how robust that conclusion is to transaction sampling and variation across experimental seeds.

## 2. Experimental design

The analysis uses five experimental seeds: 1101, 1202, 1303, 1404, and 1505. Each experiment uses 25,000 training observations and evaluates performance on 17,000 observations, with a target fraud prevalence of 2.5%.

Five methods are evaluated:

1. Random ranking
2. Velocity + distance rule
3. Logistic regression
4. Interaction logistic
5. Histogram gradient boosting

The two evaluation periods are the same-regime control and the shifted regime. The existing experiment fits models according to the original evaluation procedure and compares their performance across these regimes.

Uncertainty quantification uses 2,000 bootstrap replicates. The metric-level output contains AP, area under the receiver operating characteristic curve (AUC), and top-5% recall intervals for each seed, regime, and method.

## 3. Uncertainty estimates

Two complementary views of uncertainty are reported.

### 3.1 Transaction-bootstrap intervals

The transaction-bootstrap intervals quantify variation from resampling evaluation observations, conditional on the five fitted experimental runs. Control and shifted evaluation transactions are resampled independently because they represent different simulated transactions. Comparisons between methods within a regime use common resampled observations.

These intervals characterize evaluation-sample uncertainty; they do not capture all uncertainty from generating new datasets, fitting new models, or changing the data-generating process.

### 3.2 Seed-level intervals

The seed-level t intervals summarize variation in the AP difference across the five experimental seeds. With only five seeds, these estimates are exploratory and sensitive to between-seed variation.

The transaction-bootstrap and seed-level intervals therefore answer different questions and should be interpreted together rather than treated as interchangeable evidence.

## 4. Average precision results

The AP drop is defined as:

**AP drop = mean control AP − mean shifted-regime AP**

A positive value indicates lower AP in the shifted regime. A negative value indicates higher AP in the shifted regime.

| Method | Mean control AP | Mean shifted AP | AP drop | Transaction-bootstrap interval | Seed-level t interval |
|---|---:|---:|---:|---:|---:|
| Random ranking | 0.0260 | 0.0263 | -0.0003 | [-0.0026, 0.0025] | [-0.0042, 0.0035] |
| Velocity + distance rule | 0.0369 | 0.0446 | -0.0077 | [-0.0122, -0.0034] | [-0.0128, -0.0026] |
| Logistic regression | 0.3591 | 0.3628 | -0.0036 | [-0.0355, 0.0277] | [-0.0622, 0.0549] |
| Interaction logistic | 0.3704 | 0.2983 | 0.0722 | [0.0404, 0.1011] | [0.0113, 0.1330] |
| Histogram gradient boosting | 0.3490 | 0.2999 | 0.0490 | [0.0171, 0.0777] | [-0.0056, 0.1037] |

Values are rounded; the CSV files retain the underlying numerical precision.

## 5. Interpretation

### Interaction logistic

Interaction logistic regression has a mean AP decrease of approximately 0.0722 between the control and shifted regimes. Both the transaction-bootstrap interval and the seed-level t interval exclude zero. This is the clearest evidence in this experiment that its ranking performance deteriorates under the specified synthetic regime shift.

### Histogram gradient boosting

Histogram gradient boosting has a mean AP decrease of approximately 0.0490. Its transaction-bootstrap interval excludes zero, but its seed-level t interval includes zero. The apparent decline is therefore less robust to variation across experimental seeds than the transaction-bootstrap interval alone might suggest.

### Logistic regression

Logistic regression has a small mean AP increase of approximately 0.0036 in the shifted regime. Both intervals include zero, so the experiment does not establish a clear AP change for this method.

### Velocity + distance rule

The rule's mean AP increases from approximately 0.0369 to 0.0446. Both intervals for the control-minus-shifted difference lie below zero, consistent with improved AP under this particular shift. However, its absolute AP remains low compared with the fitted statistical and machine-learning methods. The result should not be interpreted as evidence that the rule is generally robust to distribution shift.

### Random ranking

Random ranking remains close to the expected low-precision baseline at the specified fraud prevalence. Both intervals include zero, and the results do not indicate a systematic AP change.

## 6. Limitations

This is a synthetic experiment. The findings depend on the specified feature-generation process, fraud prevalence, historical and shifted risk formulas, model configurations, and evaluation design. Real transaction data may exhibit different patterns.

Only five seeds were evaluated. The seed-level intervals consequently have limited precision and should be treated as exploratory rather than definitive. Bootstrap intervals conditional on fitted runs do not replace repeated independent experiments or evaluation on real-world data.

AP is particularly informative for this imbalanced classification task, but it is not a complete measure of operational performance. AUC and top-5% recall are retained in the metric-level output for complementary assessment. None of these synthetic results alone establishes production effectiveness, investigator workload reduction, or real-world fraud detection performance.

## 7. Reproducibility and files

The uncertainty analysis is implemented in:

- `authoring/research/uncertainty_quantification_v2.py`

Generated outputs:

- `authoring/research/synthetic_v2_bootstrap_ci_results.csv` — metric estimates and bootstrap intervals by seed, regime, and method.
- `authoring/research/synthetic_v2_paired_difference_ci.csv` — mean AP by regime, AP differences, transaction-bootstrap intervals, seed-level t intervals, and seed-level AP-drop standard deviations.

The original evaluation and regime-diagnostic CSVs are retained separately. This report supplements those outputs and does not replace their results.

## 8. Conclusion

Under the tested synthetic regime shift, interaction logistic regression shows the clearest decline in AP across both uncertainty summaries. Histogram gradient boosting shows a decline in the transaction-bootstrap analysis, but the seed-level interval includes zero. Logistic regression and random ranking show no clear AP change, while the velocity + distance rule improves modestly in this specific shifted regime despite low absolute AP.

These findings describe the tested simulation only. Further independent simulations and validation on representative real-world data would be needed before drawing operational conclusions.
