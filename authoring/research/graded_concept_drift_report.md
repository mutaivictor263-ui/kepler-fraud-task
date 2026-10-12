# Graded Concept-Drift Experiment: Results and Interpretation

## Objective

This experiment evaluates how progressively changing the synthetic fraud-risk relationship affects the ranking performance of three trained models and two simple baselines. It extends the two-regime comparison by evaluating five drift levels: 0%, 25%, 50%, 75%, and 100%.

## Experimental design

The experiment uses 20 random seeds, 25,000 training observations per seed, and 17,000 evaluation observations per seed. Models are trained using the historical risk formula. The same evaluation feature rows are used at every drift level, while the risk score is interpolated between the historical and shifted formulas.

The training-derived intercept is held fixed across all evaluation levels. Shared uniform random draws are used to generate labels across levels, reducing random label variation when comparing drift severity. This design isolates changes in the conditional risk relationship rather than changes in the marginal feature distributions.

The evaluated methods are logistic regression, interaction logistic regression, histogram gradient boosting, random ranking, and a velocity-plus-distance rule. The rule uses standard deviations estimated from training data. Performance is measured using average precision (AP), ROC AUC, and recall among the highest-scoring 5% of evaluation observations.

## Main findings

At 0% shift, interaction logistic regression has the highest mean AP among the trained models (0.3629), followed by logistic regression (0.3507) and histogram gradient boosting (0.3448).

As shift severity increases, AP declines consistently for interaction logistic regression and histogram gradient boosting. At 100% shift, interaction logistic regression has mean AP of 0.2981, a paired change of -0.0648 from the historical condition (95% seed-level t interval: -0.0764 to -0.0532). Histogram gradient boosting has mean AP of 0.3019, a paired change of -0.0428 (95% interval: -0.0547 to -0.0309).

Logistic regression is comparatively stable in AP. Its mean AP is 0.3453 at 100% shift, compared with 0.3507 at 0% shift. The paired change is -0.0055 (95% interval: -0.0163 to 0.0053), which includes zero. Therefore, this experiment does not establish a clear AP decline for logistic regression.

AUC does not tell the same story as AP. At 100% shift, mean AUC is 0.9057 for histogram gradient boosting and 0.9074 for interaction logistic regression, compared with 0.9055 and 0.9083, respectively, at 0% shift. Their AP declines despite little change in AUC. The top-5% recall also falls for both models at the highest shift level: by 0.0389 for histogram gradient boosting and 0.0413 for interaction logistic regression.

The velocity-plus-distance rule improves as the synthetic risk relationship shifts, with mean AP increasing from 0.0394 to 0.0450 and mean AUC increasing from 0.6223 to 0.6496. Its overall performance remains substantially below that of the trained models. This result illustrates that a change in the data-generating relationship need not reduce every method's performance.

Mean fraud prevalence ranges from approximately 2.37% to 2.62% across the five shift levels. The main ranking changes therefore occur without a large change in the overall positive-class rate.

## Interpretation

The results support a model-dependent response to conditional risk drift in this synthetic setting. Interaction logistic regression and histogram gradient boosting lose AP as the risk formula moves away from the historical relationship, whereas ordinary logistic regression is more stable on this metric.

The divergence between AP and AUC is operationally relevant. AUC summarizes pairwise ranking across thresholds, while AP is more directly sensitive to precision among the positive predictions and is strongly influenced by class prevalence. Monitoring AUC alone could therefore overlook meaningful deterioration in fraud prioritization.

The apparent stability of ordinary logistic regression should not be interpreted as universal robustness. It is specific to the feature distributions, risk formulas, model specifications, sample sizes, and severity path used here.

## Uncertainty and limitations

The reported intervals are paired t-based intervals calculated across 20 random seeds. They quantify variation between experimental seeds under this synthetic design; they do not quantify all sources of uncertainty and do not establish performance on real transaction data. Multiple models, metrics, and shift levels are compared, so the intervals should be interpreted as exploratory rather than multiplicity-adjusted confirmatory tests.

The intermediate severity levels are constructed by linear interpolation between two risk-score formulas. They are controlled experimental settings, not calibrated representations of real-world drift severity. Evaluation feature distributions are held fixed across levels, so this experiment studies conditional risk or concept drift rather than covariate shift. It does not establish causal explanations for the differences between models.

## Conclusion

The graded experiment demonstrates that model performance under synthetic concept drift depends on the model and evaluation metric. AP and top-5% recall reveal deterioration for interaction logistic regression and histogram gradient boosting at high drift levels, even when AUC remains nearly unchanged. Logistic regression is more stable in AP under the tested conditions, while the simple rule improves but remains a weak baseline.

These findings motivate evaluating fraud models with multiple ranking metrics and testing them under explicitly defined drift scenarios. They should be treated as synthetic robustness evidence and validated against temporally separated real-world data before being used to support operational decisions.

## Reproducibility artifacts

- `graded_concept_drift_experiment.py`: experiment implementation
- `graded_concept_drift_results.csv`: seed-level results
- `graded_concept_drift_summary.csv`: means and standard deviations by level and method
- `graded_concept_drift_paired_differences.csv`: paired changes relative to 0% shift

The existing baseline and uncertainty analysis files were not intentionally modified by this experiment.
