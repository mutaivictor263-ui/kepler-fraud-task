# Fixed-Scaling Sensitivity Analysis for Synthetic V2

## Objective

Assess whether the velocity-plus-distance rule's observed performance difference between the same-regime control and shifted regime depends on recalculating feature standard deviations separately for each evaluation dataset.

## Method

The analysis used five seeds (1101, 1202, 1303, 1404, 1505), 25,000 training observations per seed, and 17,000 observations per evaluation regime. The training-derived calibration intercept was reused for the control and shifted regimes.

Two versions of the same label-free rule were compared:

1. Original per-dataset scaling: divide transaction velocity and distance by their respective standard deviations calculated within each evaluation dataset.
2. Fixed training scaling: divide both features by standard deviations estimated from the training data and reuse those scales for both evaluation regimes.

Average precision (AP), area under the ROC curve (AUC), and recall among the top 5% of ranked observations were calculated for each seed and regime.

## Results

| Scaling method | Control AP | Shifted AP | AP difference (control − shifted) |
|---|---:|---:|---:|
| Original per-dataset scaling | 0.0369 | 0.0446 | -0.0077 |
| Fixed training scaling | 0.0368 | 0.0446 | -0.0078 |

The original and fixed-scaling results were also similar for AUC and top-5% recall. Under fixed training scaling, the mean control-minus-shifted differences were -0.0327 for AUC and -0.0115 for top-5% recall.

The exploratory five-seed t-interval for the fixed-scaling AP difference was approximately [-0.0129, -0.0026].

## Interpretation

The near-identical results suggest that recalculating the two feature scales separately within each evaluation dataset does not materially explain the rule's observed performance difference in this experiment.

This result applies only to the specified synthetic data-generating process, sample sizes, rule, and five seeds. The five-seed interval is exploratory. It does not establish general robustness to distribution shift, and the rule's higher AP in the shifted regime does not imply that the regime shift is harmless.

## Reproducibility

Script: `authoring/research/fixed_scaling_rule_sensitivity.py`

Detailed per-seed results: `authoring/research/fixed_scaling_rule_sensitivity.csv`

Summary results: `authoring/research/fixed_scaling_rule_sensitivity_summary.csv`

The original evaluation and uncertainty-analysis outputs were not modified.
