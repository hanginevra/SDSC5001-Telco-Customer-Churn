#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python 01_Data_Cleaning/01_data_cleaning.py
export TELCO_CLEAN_DATA="$PWD/results/data/telco_clean.csv"
python 01_Data_Cleaning/02_eda.py
python 02_Logistic_Regression/experiments.py
Rscript --vanilla 03_Random_Forest/rf_5001.R
mkdir -p results/xgboost
python -m nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=900 \
    --output XGBoost.executed.ipynb --output-dir results/xgboost \
    04_XGBoost/XGBoost.ipynb
python collect_results.py
python 05_Robustness_Analysis/final_analysis.py
