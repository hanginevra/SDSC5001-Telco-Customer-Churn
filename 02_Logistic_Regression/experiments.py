"""Run python experiments.py from any directory to reproduce tables and figures."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from figures import make_figures

ROOT = Path(__file__).resolve().parent
C_GRID = [0.01, 0.1, 1.0, 10.0]
CONTINUOUS = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL = ["Internet Service", "Contract", "Payment Method"]
LEVELS = [["DSL", "Fiber optic", "No"], ["Month-to-month", "One year", "Two year"],
          ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]]
BINARY = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
          "Multiple Lines", "Online Security", "Online Backup", "Device Protection",
          "Tech Support", "Streaming TV", "Streaming Movies", "Paperless Billing"]
SEED = 42
REPLICATES = 2000


def logistic(continuous, weight=None):
    prep = ColumnTransformer([
        ("num", StandardScaler(), continuous),
        ("cat", OneHotEncoder(categories=LEVELS, drop="first"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ], verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("lr", LogisticRegression(class_weight=weight,
                                                               solver="lbfgs", max_iter=2000))])


def tune(model, X, y, folds, name):
    search = GridSearchCV(model, {"lr__C": C_GRID}, scoring="roc_auc", refit=True,
                          cv=folds, error_score="raise")
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({"experiment": name, "C": results.param_lr__C,
                          "mean_fold_auc": results.mean_test_score})
    for i in range(len(folds)):
        table[f"fold_{i+1}_auc"] = results[f"split{i}_test_score"]
    return search.best_estimator_, table


def select_threshold(y, scores):
    p, r, thresholds = precision_recall_curve(y, scores)
    p, r = p[:-1], r[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-15)
    curve = pd.DataFrame({"threshold": thresholds, "precision": p, "recall": r, "f1": f1})
    # Prefer the higher threshold for exact ties (fewer alerts).
    index = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-14))[-1]
    return float(thresholds[index]), curve


def bootstrap(y, scores, classes):
    """Paired stratified resampling; average ranks handle tied scores exactly."""
    pos, neg = (np.flatnonzero(y == label) for label in [1, 0])
    n_pos, n_neg = len(pos), len(neg)
    rng = np.random.default_rng(SEED)
    # E1 has constant scores; E4 shares the E2 scores.
    result = np.empty((REPLICATES, scores.shape[1], 2))
    for b in range(REPLICATES):
        sample = np.r_[rng.choice(pos, n_pos), rng.choice(neg, n_neg)]
        ranks = rankdata(scores[sample], method="average", axis=0)
        result[b, :, 0] = (ranks[:n_pos].sum(axis=0) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        predicted = classes[sample]
        result[b, :, 1] = 2 * predicted[:n_pos].sum(axis=0) / (n_pos + predicted.sum(axis=0))
    return result


def main():
    tables = ROOT / "tables"
    tables.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data/telco_clean.csv")
    split = pd.read_csv(ROOT / "data/split.csv").set_index("row_id")
    assert split.index.equals(df.index) and set(split.subset) == {"train", "test"}
    train = split.index[split.subset == "train"]
    test = split.index[split.subset == "test"]
    fold_id = split.loc[train, "cv_fold"].to_numpy()
    assert set(fold_id) == set(range(1, 11)) and (split.loc[test, "cv_fold"] == -1).all()
    folds = [(np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)) for k in range(1, 11)]
    X = df[CONTINUOUS + CATEGORICAL + BINARY]
    y = df["Churn"]
    assert not X.isna().any().any() and set(y) == {0, 1}
    Xtr, ytr = X.loc[train], y.loc[train]
    default, grid2 = tune(logistic(CONTINUOUS), Xtr, ytr, folds, "E2")
    balanced, grid3 = tune(logistic(CONTINUOUS, "balanced"), Xtr, ytr, folds, "E3")
    pd.concat([grid2, grid3]).to_csv(tables / "cv_grid.csv", index=False)
    default_c = default["lr"].C
    oof = cross_val_predict(clone(default), Xtr, ytr, cv=folds, method="predict_proba")[:, 1]
    threshold, curve = select_threshold(ytr, oof)
    curve.to_csv(tables / "threshold_curve.csv", index=False)
    pd.DataFrame({"row_id": train, "cv_fold": fold_id, "Churn": ytr.to_numpy(),
                  "score": oof}).to_csv(tables / "train_oof.csv", index=False)
    reduced = logistic(CONTINUOUS[:2]).set_params(lr__C=default_c).fit(Xtr, ytr)
    majority = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    models = [majority, default, balanced, default, reduced]
    cutoffs = np.array([0.5, 0.5, 0.5, threshold, 0.5])
    names = ["E1", "E2", "E3", "E4", "E5"]

    # Test evaluations start only after all fit, hyperparameter and cutoff decisions.
    yt = y.loc[test].to_numpy()
    scores = np.column_stack([model.predict_proba(X.loc[test])[:, 1] for model in models])
    classes = (scores >= cutoffs).astype(int)
    simulated = bootstrap(yt, scores, classes)
    intervals = np.quantile(simulated, [0.025, 0.975], axis=0)
    cases = pd.DataFrame({"row_id": test, "Churn": yt})
    rows = []
    for j, name in enumerate(names):
        p, predicted = scores[:, j], classes[:, j]
        tn, fp, fn, tp = confusion_matrix(yt, predicted, labels=[0, 1]).ravel()
        rows.append(dict(experiment=name, C=np.nan if j == 0 else models[j]["lr"].C,
                         threshold=cutoffs[j], auc=roc_auc_score(yt, p), f1=f1_score(yt, predicted),
                         auc_low=intervals[0,j,0], auc_high=intervals[1,j,0],
                         f1_low=intervals[0,j,1], f1_high=intervals[1,j,1],
                         precision=precision_score(yt, predicted, zero_division=0),
                         recall=recall_score(yt, predicted), accuracy=(predicted == yt).mean(),
                         TN=tn, FP=fp, FN=fn, TP=tp))
        cases[f"{name}_score"] = p
        cases[f"{name}_pred"] = predicted
    metrics = pd.DataFrame(rows).set_index("experiment")
    metrics.to_csv(tables / "test_metrics.csv")
    cases.to_csv(tables / "test_predictions.csv", index=False)
    contrasts = []
    for j in (2, 3, 4):
        low, high = np.quantile(simulated[:, j] - simulated[:, 1], [0.025, 0.975], axis=0)
        for k, metric in enumerate(["auc", "f1"]):
            contrasts.append(dict(comparison=f"{names[j]} minus E2", metric=metric,
                                  difference=metrics.loc[names[j], metric] - metrics.loc["E2", metric],
                                  low=low[k], high=high[k]))
    pd.DataFrame(contrasts).to_csv(tables / "paired_differences.csv", index=False)
    beta = default["lr"].coef_[0]
    pd.DataFrame({"feature": default["prep"].get_feature_names_out(), "beta": beta,
                  "odds_ratio": np.exp(beta)}).to_csv(tables / "default_coefficients.csv", index=False)
    make_figures(metrics, curve, ROOT / "figures")
    print(metrics.round(4).to_string())
    print(pd.DataFrame(contrasts).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
