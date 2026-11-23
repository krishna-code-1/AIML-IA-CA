# Bank Marketing Campaign Subscription Prediction System

A Streamlit application for exploring the project dataset and its analysis, preprocessing, modeling and evaluation artifacts, with individual and CSV batch predictions from the currently selected `duplicate_safe_v2` Random Forest pipeline.

## Current experiment

- **Selected model:** Random Forest (`ml/saved_models/duplicate_safe_v2/selected_model.joblib`)
- **Experiment:** `duplicate_safe_v2`
- **Algorithms compared:** six (Linear SVM, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, MLP and Random Forest)
- **Validation:** five-fold `StratifiedGroupKFold` on the training partition
- **Holdout:** group-aware `StratifiedGroupKFold` train/test split
- **Split artifacts:** `data/splits/duplicate_safe_v2/manifest.json`, `train_indices.csv`, `test_indices.csv`
- **CV artifacts:** `reports/splitting/duplicate_safe_v2/cross_validation_folds.csv`, `cross_validation_fold_summary.csv`, and `grid_search_cv_splits.json`
- **Training reports:** `reports/model_training/duplicate_safe_v2/model_comparison_summary.csv`, `cv_results.csv`, `fold_level_metrics.csv`, and `training_configuration.json`
- **Evaluation reports:** `reports/model_evaluation/duplicate_safe_v2/test_metrics.json`, `classification_report.json`, and `confusion_matrix.csv`
- **Application entry point:** `main.py`

Earlier four-model and six-model runs are historical experiments, with their reports retained under `reports/`. They do not identify the current prediction model. The currently selected artifact is the v2 Random Forest above.

The split groups rows with identical model inputs (including the derived `previously_contacted` feature) so that copies of the same input do not cross training, validation or test boundaries. The 237 groups whose identical inputs have mixed target labels remain in the dataset; they represent conflicting observed outcomes rather than duplicate rows that can safely be discarded.

## Environment and dependencies

The inspected project virtual environment uses Python 3.13.17. The saved pipeline requires scikit-learn **1.9.1**; use that exact version to load it. Runtime packages and exact versions are listed in `requirements.txt`. The versions were read from the existing `.venv`; no package was installed or upgraded for this documentation update. Pytest is not installed in that environment; its test dependency is listed separately in `requirements-dev.txt`.

On Windows, open PowerShell in this directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Streamlit prints the local URL (normally `http://localhost:8501`). Stop the app with `Ctrl+C`.

To run backend tests:

```powershell
python -m unittest discover -s tests -v
```

If pytest is installed from `requirements-dev.txt`, also run:

```powershell
python -m pytest
```

Install that optional test dependency with `python -m pip install -r requirements-dev.txt` when desired; pytest was unavailable in the inspected environment and no tested pytest version can be recorded.

The environment used for this update was Python 3.13.17. The project artifacts were produced with scikit-learn 1.9.1; loading the selected model with another scikit-learn version is explicitly rejected. Other Python versions have not been verified here. `.venv` is a local environment and is not a project artifact.

## Inputs and prediction behavior

The individual form accepts 19 fields: `age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, and `poutcome`. `previously_contacted` is derived from `pdays`. `duration` and target `y` are excluded from prediction.

The app displays the Random Forest's class prediction and model-estimated class scores. Probability calibration has not been verified; scores are not guaranteed subscription outcomes. Require human review before business action. Do not automatically approve or reject customers based on a prediction. Fairness has not been established through subgroup evaluation.

Unseen categorical values are accepted by the fitted encoder, and the app identifies them so the operator can correct a typo or continue when the value is intentional. Bulk upload accepts UTF-8 CSV (including a BOM), at most 10 MiB and 10,000 data rows. Streamlit can buffer an upload before application code runs; the page checks the reported size before calling `getvalue()`, and the backend independently checks the resulting byte length. This avoids an extra full-size copy for a file already reported over the limit, but does not prevent Streamlit's initial buffering.

Numeric validation uses these rules:

| Field | Accepted values | Rule type and reason |
|---|---:|---|
| `age` | 0–120, whole number | Broad plausibility guard; not an eligibility rule. Source data spans 17–98. |
| `campaign` | Whole number, at least 1 and at most 9,007,199,254,740,991 | A contact count starts at one; upper bound is the IEEE-754 safe-integer technical limit. |
| `pdays` | Whole number, 0–999 | Dataset convention; 999 means no prior contact and is normalized by the existing preprocessing contract. |
| `previous` | Whole number, 0–9,007,199,254,740,991 | Nonnegative count; upper bound is the safe-integer technical limit. |
| `emp.var.rate` | -9,007,199,254,740,991 to +9,007,199,254,740,991 | Signed numeric measure; broad safe-integer magnitude guard only. |
| `cons.price.idx` | 0 to 9,007,199,254,740,991 | Nonnegative index; broad safe-integer magnitude guard only. |
| `cons.conf.idx` | -9,007,199,254,740,991 to +9,007,199,254,740,991 | Signed index; broad safe-integer magnitude guard only. |
| `euribor3m` | -9,007,199,254,740,991 to +9,007,199,254,740,991 | Signed rate so negative values are allowed; broad safe-integer magnitude guard only. |
| `nr.employed` | 0 to 9,007,199,254,740,991 | Nonnegative measure; broad safe-integer magnitude guard only. |

All numeric inputs must also parse as finite numbers; integer count fields must be whole numbers. The technical bound keeps numeric magnitudes within the IEEE-754 safe-integer range used by the app's float parsing. It is not a business limit. These broad guards are not claims that values inside them are representative of training data; the source summary ranges are narrower. Exported uploaded text cells whose first non-whitespace/control character is `=`, `+`, `-` or `@` are prefixed with an apostrophe unless the cell is an ordinary finite numeric value. Prediction columns are unchanged. This sanitizes the CSV export only, not source values; spreadsheet import behavior varies, so CSV should still be treated as untrusted data and imported as text when possible. Invalid records remain in the output with validation reasons.

The bulk output preserves the existing columns. Formula escaping is applied only while preparing exported rows and does not mutate prediction results or input records.

No prediction history is stored by the application. Use only approved customer data in an appropriately secured environment. The held-out test data has already been used for final evaluation; do not reuse it for fitting, tuning or threshold selection. No public deployment is included.

## Artifact integrity baseline

`reports/integrity/phase2L_artifact_manifest.json` records SHA-256 hashes and byte sizes for protected project artifacts, excluding test-prediction files and the manifest itself. This is a new baseline for future comparisons. It does not establish that those artifacts were unchanged during earlier phases.
