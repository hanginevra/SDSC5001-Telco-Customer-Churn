# SDSC5001 Telco Customer Churn Prediction

This group project studies customer churn with logistic regression, random forest and XGBoost. The team-provided [IBM Telco Customer Churn sample workbook](data/Telco_customer_churn.xlsx) contains 7,043 customer records in the `Telco_Churn` sheet. The cleaning script converts the target to `Churn` (1 for churn, 0 for stay), handles blank total charges for zero-tenure customers and drops identifiers, geography, constant fields and fields that could leak churn information.

## Repository layout

```text
01_Data_Cleaning/
  01_data_cleaning.py       Clean the Excel workbook and write a cleaning report
  02_eda.py                 Produce full-dataset descriptive tables and figures
02_Logistic_Regression/
  experiments.py           Train and evaluate the majority baseline and LR variants
  figures.py               Produce LR performance and threshold figures
03_Random_Forest/
  rf_5001.R                Tune and evaluate ranger models and sensitivity variants
04_XGBoost/
  XGBoost.ipynb            Tune and evaluate XGBoost, then explain its predictions
data/
  Telco_customer_churn.xlsx
  telco_clean.csv          Committed cleaned data in workbook row order
  split.csv               Shared split for the Python models
collect_results.py         Validate test IDs and collect metrics and predictions
requirements.txt           Python direct dependencies
run_all.sh                 Run every stage in order
results/                   Generated outputs and selected review snapshots
```

## Setup and run

The complete workflow was checked on Linux x86-64 with Python 3.12, R 4.5.3 and `ranger` 0.18.0. Install R so that `Rscript` is available on your PATH. From the repository root, create a Python environment and install the dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
Rscript --vanilla -e 'install.packages("ranger", repos = "https://cloud.r-project.org")'
Rscript --vanilla -e 'packageVersion("ranger")'
bash run_all.sh
```

Install `ranger` into a writable R library if your system library is read-only. For the exact tested R package version, a conda-forge alternative is `conda create -n telco-r -c conda-forge r-base=4.5.3 r-ranger=0.18.0`. Use that environment's `Rscript` alongside the Python environment.

The runner regenerates the cleaned CSV in `results/data/` and passes it to each downstream stage via `TELCO_CLEAN_DATA`. Running individual stages uses the committed clean CSV by default. If supplying a different CSV to the Python models, use an absolute `TELCO_CLEAN_DATA` path and preserve the row order required by `data/split.csv`.

To run the stages individually from the repository root with the regenerated data:

```bash
python 01_Data_Cleaning/01_data_cleaning.py
export TELCO_CLEAN_DATA="$PWD/results/data/telco_clean.csv"
python 01_Data_Cleaning/02_eda.py
python 02_Logistic_Regression/experiments.py
Rscript --vanilla 03_Random_Forest/rf_5001.R
mkdir -p results/xgboost
python -m nbconvert --execute --to notebook \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=900 \
    --output XGBoost.executed.ipynb --output-dir results/xgboost \
    04_XGBoost/XGBoost.ipynb
python collect_results.py
```

You can also open the XGBoost notebook from the repository root or `04_XGBoost/` and run all cells using the project's Python environment.

## Experimental design

| Model | Split and features | Variants and thresholds |
| --- | --- | --- |
| Logistic regression | Shared 5,634 train / 1,409 test split with ten predefined training folds. Standardizes continuous variables within each fit. Uses 23 encoded features, excluding `ServiceCount`. | E1 majority baseline, E2 default LR, E3 class-weighted LR, E4 the E2 model with its F1 threshold chosen from training out-of-fold scores and E5 with 22 features after dropping `Total Charges`. E1/E2/E3/E5 use a cutoff of 0.5. |
| Random forest | Independent stratified 80/20 split and stratified ten-fold training CV in R with seed 42. Uses 24 encoded features, including `ServiceCount`. | Unweighted and class-weighted runs use `score > 0.5`. The sensitivity analysis drops `Total Charges` and retunes using 23 features. |
| XGBoost | Same split and ten folds as LR. Uses 23 encoded features, excluding `ServiceCount`. | Initial and tuned models at 0.5, plus the tuned model with its F1 threshold chosen from training out-of-fold scores. |

`row_id` is a zero-based position in the committed cleaned CSV. Keep the workbook row order to preserve the split mapping. The R script exports its realized split in the same format. The two holdouts overlap for 292 of 1,409 rows on the tested environment. Because the holdouts and feature sets differ, scores across all three model families are descriptive comparisons. The project computes no paired statistical tests across model families. The LR paired intervals compare its own variants evaluated on the same test rows.

## Outputs

Git tracks the two combined CSVs, the executed XGBoost notebook and the PNG/PDF figures as review snapshots. New runtime outputs under `results/` are ignored by default.

- `results/data/` and `results/eda/` contain the cleaned data, descriptive summaries and EDA figures.
- `results/logistic_regression/` and `results/random_forest/` contain the original model-specific metrics, predictions, plots and supporting tables. `rf_split.csv` records the R sample allocation and training folds.
- `results/xgboost/` contains the executed notebook with SHAP results, PNG figures, metrics and test predictions.
- `results/model_metrics.csv` reports ROC-AUC, F1, precision, recall, accuracy and confusion matrix counts (`tn`, `fp`, `fn`, `tp`) for each variant, along with split, feature count and threshold.
- `results/model_predictions.csv` reports test row ID, true class, positive-class score and predicted class for each variant. The collector validates IDs and labels against each model's corresponding split before writing either combined table.
