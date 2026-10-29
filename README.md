# Reproducibility materials for applsci-4663183

Manuscript: **Predicting Perceived Barriers to Wider AI Use in Six Survey Samples**.

This archive contains the analysis code, configuration records, aggregate results and figure scripts for the revised study of ten-category perceived barriers to wider AI use in six survey samples. The analysed sample contains 3528 adults. All ten response categories are retained.

The included aggregate files can be inspected and all figures reproduced without access to individual responses. Full refitting requires the six authorised source workbooks. The source project's consent and provider conditions do not authorise unrestricted public redistribution of individual records; requests should be addressed to the corresponding author and remain subject to those conditions.

## Environment

The reported analyses used Python 3.12.14. Exact package versions are pinned in `requirements.txt`. The original environment is recorded in `results/environment.json`; the supplementary CatBoost environment and fixed settings are recorded in `results/revision/catboost/protocol.json`. CatBoost is pinned to version 1.2.8. From this directory, install the dependencies in a separate Python environment:

```bash
python -m pip install -r requirements.txt
```

Model fitting uses one numerical thread. The primary model seed and the random-fold split seed are both 20261014. In `analyze.py`, `MODEL_SEED` and `SPLIT_SEED` are separate variables so that changing model randomness does not silently change the random validation partitions. Their default results are unchanged from the original implementation.

## Recreate the figures from public aggregates

```bash
python plot_aggregate_results.py
```

This command reads CSV files in `results/` and writes four PNG/PDF pairs to `figures/`. It does not read the private data directory or fit any model:

- `Figure_1_barriers`: observed category percentages within each supplied site.
- `Figure_2_validation`: each original unscaled model's log-loss gain over its corresponding training prior under both evaluation designs, and random-forest gain in each held-out site.
- `Figure_3_diagnostics`: random-forest mean predicted minus observed category frequency, and logistic-regression permutation diagnostics.
- `Figure_S1_class_recall`: pooled category-specific recall for the full logistic regression, random forest and gradient-boosting models.

Full category names and codes are in Supplementary Table S1 and `outcome_codebook.csv`. Figure labels are abbreviated where necessary for readability.

## Full rerun with authorised source data

Place the following six original workbooks in `extracted/data/`, preserving their filenames:

```text
Zbiór_AŚ_Marcin_Budzinski_GZM.xlsx
Zbiór_AŚ_Marcin_Budzinski_Istambuł.xlsx
Zbiór_AŚ_Marcin_Budzinski_Montreal.xlsx
Zbiór_AŚ_Marcin_Budzinski_Nowy_Jork.xlsx
Zbiór_AŚ_Marcin_Budzinski_Pekin.xlsx
Zbiór_AŚ_Marcin_Budzinski_Praga.xlsx
```

`source_audit.json` records the SHA-256 hashes of the workbooks used for the reported analyses. Use those inputs for an exact rerun. Each workbook includes a coded worksheet and a labelled worksheet; the preparation script consults both and writes the harmonised data and codebook under `private/`. The GZM duration is converted from seconds to minutes. Missing code 99 in the confidence item is preserved in the harmonised data and that item is excluded from the predictor set. The separate source-questionnaire item 14 is absent from all delivered datasets; the response-quality sensitivity uses the thirteen available pre-AI agreement items p1-p13.

Run the commands in this order:

```bash
python prepare_data.py
python analyze.py primary
python analyze.py random
python analyze.py sensitivity
python summarize.py
python compare_priors.py
python rf_revisions.py all
python matched_training_size.py
python plot_aggregate_results.py
```

The original three `analyze.py` tasks run nested site-blocked tuning and temperature estimation. `summarize.py` reads private records and predictions to rebuild aggregate descriptions, class-wise results, bootstrap contrasts and permutation summaries. It is deliberately separate from the public-only figure script.

The additional analyses are implemented as follows:

- `compare_priors.py` reconstructs the original outer partitions and evaluates the respondent-weighted add-one prior, an equal-site average of separately add-one-smoothed training-site distributions, and an unsmoothed equal-site sensitivity. It reports each model's gain over each prior and the exact arithmetic contrast between evaluation designs.
- `rf_revisions.py all` repeats primary LOSO fitting for the nine additional seeds 20261015-20261023, combines them with the original 20261014 fit, and refits the quality-screen and no-concern-item sensitivities with the original seed. Hyperparameters and temperature are selected again for every fit. The outer partitions remain fixed. `seeds` and `sensitivities` run the components separately; `summarize` rebuilds the ten-seed summary from existing outputs.
- `matched_training_size.py` draws training sets of 2670 respondents using five fixed subsampling seeds 20261028-20261032, with proportional site-by-category stratification. The original outer test sets remain unchanged. Full logistic regression and random forest use the same training subsamples, and their preprocessing, tuning and calibration are refitted. Metrics are evaluated separately in each repetition; probabilities are not averaged into an ensemble.

The seed integers are reproducibility identifiers, not dates. Re-running commands overwrites their corresponding result files. Keep an untouched copy of the released aggregate archive when comparing reruns.

## Supplementary CatBoost comparison

The CatBoost comparison was added during revision after the original model results were known. Its settings were fixed before the first CatBoost fit. It is a bounded exploratory comparison; it does not establish the best achievable prediction or provide independent confirmation on new samples. All results are retained regardless of their comparative performance.

After preparing the same authorised data, validate the inputs and run the extension from this directory:

```bash
python catboost_extension.py check
python catboost_extension.py run --workers 4
```

`check` fits no models. It verifies the 3528-row, fifteen-predictor, ten-category specification, reconstructs the original partitions, and checks the recomputed training-prior metrics against the original aggregates. If the original private prediction arrays are available, it also checks their prior predictions in the original row order.

The fixed grid contains eight configurations: iterations in `{100, 500}`, depth in `{4, 6}` and L2 leaf regularization in `{3, 10}`, with learning rate 0.05 and CPU Ordered boosting. Each model uses one numerical thread and seed 20261014. The `--workers` option controls independent outer-fold processes, not model hyperparameters. Sex and education are handled natively as categorical variables; the remaining full-model predictors retain their original numeric interpretation. Only the fifteen declared predictor columns enter CatBoost. All quantization and categorical transformations are fitted within the relevant training subset.

Outer test partitions and inner site-blocked selection match the original analysis. No validation or outer test data are passed to CatBoost as an `eval_set`; early stopping and best-iteration selection are disabled. Candidate selection uses equal-site inner log loss. Temperature scaling uses the selected candidate's inner out-of-fold probabilities and the original bounds and optimizer. The script reports both raw and calibrated performance, both training-prior conventions, category recovery and the arithmetic contrast between validation designs.

The new public outputs are under `results/revision/catboost/`; individual predictions and test indices are under `private/catboost/` and must not be published. The script leaves the original model results unchanged. It freezes the data/code/version/settings identity in `protocol.json` before fitting and resumes only matching atomic checkpoints. Run the same `run` command to resume an interrupted calculation. A different protocol requires a separate output directory. The CLI also accepts `--data`, `--baseline-results`, `--baseline-private`, `--output` and `--private-output` for explicitly located inputs and outputs. `assemble` rebuilds aggregate reports from complete, matching private checkpoints without fitting models.

This addition uses one CatBoost seed. The existing ten-seed, matched-size, quality-screen and concern-item analyses retain their stated LR/RF scope; they are not extended to CatBoost. Adding an algorithm does not resolve undocumented sampling or language procedures. No new respondent data are introduced, and no individual records, predictions or split indices are included in the public package.

## Result files

| Location | Contents |
| --- | --- |
| `results/metrics_loso_full.csv`, `metrics_random_full.csv` | Original per-site, pooled and equal-site metrics for raw and temperature-scaled models |
| `results/config_*.json` | Selected parameters, temperatures and candidate inner-validation losses |
| `results/classwise_performance.csv`, `confusion_*.csv` | Category-level probability summaries, recall and confusion matrices |
| `results/paired_bootstrap_logloss.csv` | Fixed-prediction respondent bootstrap contrasts |
| `results/revision/priors/` | Both baseline conventions, model gains and arithmetic contrasts |
| `results/revision/rf/` | Ten-seed RF summaries, individual seed configurations and RF sensitivities |
| `results/revision/matched_size/` | Five matched-size repetitions, configurations, gains and arithmetic contrasts |
| `results/revision/catboost/` | Frozen supplementary protocol, eight-candidate configurations, raw/calibrated metrics, comparison tables, classwise results, confusion matrices and gain contrasts |
| `source_audit.json`, `source_codebook.json`, `outcome_codebook.csv` | Input hashes, source-column mappings and outcome category codes |
| `source_history/analyze_original.py` | Original analysis implementation underlying the archived source hash |

The `private/` directory contains individual data, prediction arrays and split indices when analyses are run locally. It and `extracted/` are excluded by `.gitignore` and are not included in this archive.

## Interpretation and scope

The primary summary is an equal-site mean; pooled metrics are labelled separately. The original training prior uses pooled training-respondent counts. The additional equal-site prior averages distributions across the training sites only.

The identity comparing model-loss differences, baseline-loss differences and differences in model gains is an arithmetic description. It does not identify a causal effect of geography or isolate changes in predictor-outcome relationships. Matching the outer training size does not match site composition or the number of sites available in the inner validation. Seed and subsampling ranges are not population confidence intervals. The response-quality sensitivity also changes the evaluated sample.

Geographic names are the provider-supplied sample labels. Recruitment procedures and the administered questionnaire versions were unavailable to the authors. Neither harmonised response codes nor the Polish export labels establish cross-language measurement equivalence.

## Integrity

`MANIFEST_SHA256.txt` contains a SHA-256 digest for every released file other than itself. It can be checked from this directory with `sha256sum -c MANIFEST_SHA256.txt` on systems that provide that command. `verify_manifest.py` provides the same check using Python's standard library.

The archive contains no licence grant beyond the rights applicable to its contents. The repository owner may add an appropriate software licence when publishing the code.
