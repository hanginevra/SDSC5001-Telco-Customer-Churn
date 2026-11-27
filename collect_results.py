"""Validate and collect the native model exports without refitting any model."""

import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def collect(predictions, cases, split, churn, model, split_name):
    expected = split.loc[split["subset"].eq("test"), "row_id"]
    if not predictions["row_id"].equals(expected.reset_index(drop=True)):
        raise ValueError(f"{model} prediction IDs do not match its test split.")
    if not predictions["Churn"].equals(churn.iloc[expected].reset_index(drop=True)):
        raise ValueError(f"{model} prediction labels do not match the cleaned input.")

    metrics = []
    exports = []
    for variant, features, threshold, threshold_type, score_column, pred_column in cases:
        scores = predictions[score_column]
        predicted = predictions[pred_column]
        actual = predictions["Churn"]
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        labels = dict(model=model, variant=variant, split=split_name,
                      n_features=int(features), threshold=threshold, threshold_type=threshold_type)
        metrics.append(dict(
            **labels, n_train=split["subset"].eq("train").sum(), n_test=len(expected),
            roc_auc=roc_auc_score(actual, scores), f1=f1_score(actual, predicted),
            precision=precision_score(actual, predicted, zero_division=0),
            recall=recall_score(actual, predicted, zero_division=0),
            accuracy=accuracy_score(actual, predicted), tn=tn, fp=fp, fn=fn, tp=tp,
        ))
        exports.append(predictions[["row_id", "Churn"]].assign(
            **labels, score=scores, prediction=predicted,
        ))
    return metrics, exports


def main():
    churn = read_csv(os.environ.get("TELCO_CLEAN_DATA", ROOT / "data" / "telco_clean.csv"))["Churn"]
    shared_split = read_csv(ROOT / "data" / "split.csv")
    rf_split = read_csv(OUTPUT / "random_forest" / "rf_split.csv")
    if not rf_split.equals(shared_split):
        raise ValueError("Random Forest must use the shared test set and training folds.")
    metrics = []
    exports = []

    logistic = OUTPUT / "logistic_regression" / "tables"
    lr_table = read_csv(logistic / "test_metrics.csv")
    lr_cases = [
        (e.experiment, e.n_features, e.threshold, "optimized" if e.experiment == "E4" else "fixed",
         f"{e.experiment}_score", f"{e.experiment}_pred")
        for e in lr_table.itertuples(index=False)
    ]
    rows, frames = collect(read_csv(logistic / "test_predictions.csv"), lr_cases,
                           shared_split, churn, "Logistic Regression", "shared_split")
    metrics.extend(rows)
    exports.extend(frames)

    for analysis in ("main", "without_total_charges"):
        forest = OUTPUT / "random_forest" / analysis
        rf_table = read_csv(forest / "rf_test_metrics.csv")
        rf_cases = [
            (f"{analysis}/{e.weighting}", e.n_features, e.threshold, "fixed",
             f"{e.weighting}_probability", f"{e.weighting}_prediction")
            for e in rf_table.itertuples(index=False)
        ]
        predictions = read_csv(forest / "rf_test_predictions.csv").rename(
            columns={"csv_row_0based": "row_id", "churn": "Churn"},
        )
        rows, frames = collect(predictions, rf_cases, rf_split, churn,
                               "Random Forest", "shared_split")
        metrics.extend(rows)
        exports.extend(frames)

    xgboost = OUTPUT / "xgboost"
    xgb_table = read_csv(xgboost / "xgboost_test_metrics.csv")
    xgb_columns = [
        ("fixed", "xgb_initial_score", "xgb_initial_pred_0.5"),
        ("fixed", "xgb_tuned_score", "xgb_tuned_pred_0.5"),
        ("optimized", "xgb_tuned_score", "xgb_tuned_pred_oof_threshold"),
    ]
    xgb_cases = [
        (e.experiment, e.n_features, e.threshold, *columns)
        for e, columns in zip(xgb_table.itertuples(index=False), xgb_columns, strict=True)
    ]
    rows, frames = collect(read_csv(xgboost / "xgboost_test_predictions.csv"), xgb_cases,
                           shared_split, churn, "XGBoost", "shared_split")
    metrics.extend(rows)
    exports.extend(frames)

    pd.DataFrame(metrics).to_csv(OUTPUT / "model_metrics.csv", index=False)
    pd.concat(exports, ignore_index=True).to_csv(OUTPUT / "model_predictions.csv", index=False)
    print(f"Collected {len(metrics)} experiment rows in {OUTPUT / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
