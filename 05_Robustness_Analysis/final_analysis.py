"""Compare the three models and their Total Charges ablations on a shared holdout."""

import hashlib
import os
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}
COLORS = dict(zip(MODELS, ("#205579", "#167A6D", "#B35629"), strict=True))
FEATURE_SETS = ("full", "reduced")
POLICIES = {"fixed": "Fixed cutoff (0.5)", "optimized": "Training OOF cutoff"}
METRICS = {
    "roc_auc": "ROC-AUC", "f1": "F1-score", "precision": "Precision",
    "recall": "Recall", "accuracy": "Accuracy",
}
REPLICATES = 2000
SEED = 42


def load_study():
    data_path = Path(os.environ.get("TELCO_CLEAN_DATA", ROOT / "data/telco_clean.csv"))
    split_path = ROOT / "data/split.csv"
    split = pd.read_csv(split_path)
    labels = pd.read_csv(data_path)["Churn"]
    if not np.array_equal(split.row_id, np.arange(len(labels))) or set(labels) != {0, 1}:
        raise ValueError("The split must map every cleaned row to its zero-based ID and binary label.")
    design = split.assign(Churn=labels)
    train = design.subset.eq("train")
    test = design.subset.eq("test")
    if (not (train | test).all() or not design.loc[test, "cv_fold"].eq(-1).all()
            or set(design.loc[train, "cv_fold"]) != set(range(1, 11))
            or design.loc[train].groupby("cv_fold").Churn.nunique().ne(2).any()
            or design.loc[test, "Churn"].nunique() != 2):
        raise ValueError("Expected both classes in each training fold and in the shared holdout.")

    scores = {}
    audit = []
    reference = None
    hashes = {
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    for model, folder in MODELS.items():
        path = RESULTS / folder
        predictions = pd.read_csv(path / "study_scores.csv", float_precision="round_trip")
        features = pd.read_csv(path / "study_features.csv")
        if any(set(frame.feature_set) != set(FEATURE_SETS) for frame in (predictions, features)):
            raise ValueError(f"{model} must supply full and reduced exports.")
        names = {
            setting: sorted(features.loc[features.feature_set.eq(setting), "feature"].tolist())
            for setting in FEATURE_SETS
        }
        if (len(set(names["full"])) != 23 or len(names["full"]) != 23
                or "Total Charges" not in names["full"] or "ServiceCount" in names["full"]
                or names["reduced"] != [name for name in names["full"] if name != "Total Charges"]):
            raise ValueError(f"{model} reduced features must remove only Total Charges.")
        if reference is None:
            reference = names
        elif names != reference:
            raise ValueError(f"{model} has different feature names from the other models.")
        for setting in FEATURE_SETS:
            export = predictions.loc[predictions.feature_set.eq(setting)].sort_values("row_id")
            try:
                pd.testing.assert_frame_equal(export[list(design)].reset_index(drop=True), design)
            except AssertionError as error:
                raise ValueError(f"{model} ({setting}) IDs, labels or fold assignments differ.") from error
            if not export.score.between(0, 1).all():
                raise ValueError(f"{model} ({setting}) has missing or invalid probabilities.")
            scores[model, setting] = export.score.to_numpy()
            audit.append({
                "model": model, "feature_set": setting, "n_features": len(names[setting]),
                "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                "test_positives": int(design.loc[test, "Churn"].sum()),
                "cv_folds": design.loc[train, "cv_fold"].nunique(),
                "encoded_features": " | ".join(names[setting]), **hashes,
            })
    return design, scores, pd.DataFrame(audit)


def select_threshold(model, labels, scores):
    if model == "XGBoost":
        thresholds = np.arange(0.05, 0.96, 0.01)
        f1_values = [f1_score(labels, scores >= threshold) for threshold in thresholds]
        best = np.argmax(f1_values)
    else:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1], 1e-15,
        )
        best = np.flatnonzero(np.isclose(f1_values, f1_values.max(), atol=1e-14, rtol=0))[-1]
    return thresholds[best], f1_values[best]


def performance(labels, scores, predicted):
    """Compute metrics column-wise, using average ranks for tied probabilities."""
    pos = int(labels.sum())
    neg = len(labels) - pos
    ranks = rankdata(scores, method="average", axis=0)
    auc = (ranks[labels].sum(axis=0) - pos * (pos + 1) / 2) / (pos * neg)
    tp = predicted[labels].sum(axis=0)
    fp = predicted[~labels].sum(axis=0)
    tn, fn = neg - fp, pos - tp
    return {
        "roc_auc": auc, "f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1),
        "precision": tp / np.maximum(tp + fp, 1), "recall": tp / pos,
        "accuracy": (tp + tn) / len(labels), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def paired_bootstrap(labels, scores, predicted):
    rng = np.random.default_rng(SEED)
    pos = np.flatnonzero(labels)
    neg = np.flatnonzero(~labels)
    samples = np.empty((REPLICATES, scores.shape[1], len(METRICS)))
    for i in range(REPLICATES):
        rows = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        metrics = performance(labels[rows], scores[rows], predicted[rows])
        samples[i] = np.column_stack([metrics[metric] for metric in METRICS])
    return samples


def difference_table(pairs, index, observed, samples):
    rows = []
    for left, right, metadata in pairs:
        a, b = index[left], index[right]
        low, high = np.quantile(samples[:, b] - samples[:, a], [0.025, 0.975], axis=0)
        for k, metric in enumerate(METRICS):
            rows.append({
                **metadata, "metric": metric, "difference": observed[metric][b] - observed[metric][a],
                "ci_low": low[k], "ci_high": high[k], "bootstrap_replicates": REPLICATES,
            })
    return pd.DataFrame(rows)


def write_robustness_summary(audit, deltas, path):
    design = audit.iloc[0]
    counts = audit.groupby("feature_set").n_features.first()
    lines = [
        "Telco churn robustness audit",
        "",
        "All three models passed checks for shared row IDs, labels, folds and feature definitions.",
        f"Shared split: {design.train_rows:,} training rows, {design.test_rows:,} test rows "
        f"({design.test_positives:,} churn cases), {design.cv_folds} training folds.",
        f"Full: {counts['full']} features. Reduced: {counts['reduced']} features, removing only Total Charges.",
        f"Cleaned data SHA-256: {design.data_sha256}",
        f"Shared split SHA-256: {design.split_sha256}",
        "",
        f"Differences are reduced minus full. 95% CIs use {REPLICATES:,} paired, label-stratified "
        f"test bootstrap samples (seed {SEED}).",
        "Cutoffs and fitted models are held fixed within the bootstrap.",
    ]
    for (model, policy), group in deltas.groupby(["model", "threshold_policy"], sort=False):
        lines.extend(["", f"{model} | {POLICIES[policy]}"])
        for row in group.itertuples(index=False):
            lines.append(
                f"  {METRICS[row.metric]}: {row.difference:+.5f} "
                f"(95% CI [{row.ci_low:+.5f}, {row.ci_high:+.5f}])"
            )
    lines.extend(["", "See feature_set_performance.csv and paired_feature_differences.csv for full precision."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(figure, path):
    for extension in (".png", ".pdf"):
        figure.savefig(path.with_suffix(extension), dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_comparison(table, labels, scores, index, output):
    fig, ax = plt.subplots(figsize=(6.4, 5), layout="constrained")
    roc_points = []
    for model in MODELS:
        fpr, tpr, _ = roc_curve(labels, scores[:, index[model, "full", "fixed"]])
        auc = table.loc[model, "full", "fixed"].roc_auc
        ax.plot(fpr, tpr, color=COLORS[model], label=f"{model} ({auc:.3f})")
        roc_points.append(pd.DataFrame({"model": model, "false_positive_rate": fpr,
                                        "true_positive_rate": tpr}))
    ax.plot([0, 1], [0, 1], "--", color="#8596A0", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC on the shared test set")
    ax.legend(loc="lower right")
    save_plot(fig, output / "roc_curves")
    pd.concat(roc_points, ignore_index=True).to_csv(output / "roc_points.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6), layout="constrained")
    for i, policy in enumerate(POLICIES):
        for ax, model in zip(axes[i], MODELS, strict=True):
            row = table.loc[model, "full", policy]
            matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
            ConfusionMatrixDisplay(matrix, display_labels=["Stay", "Churn"]).plot(
                ax=ax, colorbar=False, cmap="Blues", values_format="d", im_kw={"vmin": 0, "vmax": len(labels)},
            )
            source = "fixed" if policy == "fixed" else "training OOF"
            ax.set_title(f"{model}\nThreshold {row.threshold:.3f} ({source})", fontsize=10)
    save_plot(fig, output / "confusion_matrices")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    positions = np.arange(len(METRICS))
    for ax, (policy, label) in zip(axes, POLICIES.items(), strict=True):
        for j, model in enumerate(MODELS):
            row = table.loc[model, "full", policy]
            ax.bar(positions + (j - 1) * 0.24, row[list(METRICS)], width=0.24,
                   color=COLORS[model], label=model)
        ax.set(xticks=positions, xticklabels=list(METRICS.values()), ylim=(0, 1), title=label)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test score")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncols=3, frameon=False)
    save_plot(fig, output / "performance_comparison")


def plot_robustness(differences, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True, layout="constrained")
    positions = np.arange(len(METRICS))
    extent = 1.1 * max(0.005, differences[["difference", "ci_low", "ci_high"]].abs().to_numpy().max())
    for ax, (policy, label) in zip(axes, POLICIES.items(), strict=True):
        ax.axvline(0, color="#8596A0", linewidth=1, linestyle="--")
        panel = differences.loc[differences.threshold_policy.eq(policy)]
        for j, model in enumerate(MODELS):
            part = panel.loc[panel.model.eq(model)].set_index("metric").loc[list(METRICS)]
            y = positions + (j - 1) * 0.19
            ax.hlines(y, part.ci_low, part.ci_high, color=COLORS[model])
            ax.plot(part.difference, y, "o", color=COLORS[model], label=model, markersize=4)
        ax.set(yticks=positions, yticklabels=list(METRICS.values()), xlim=(-extent, extent),
               ylim=(len(METRICS) - 0.55, -0.55), title=label,
               xlabel="Reduced minus full (95% paired bootstrap CI)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncols=3, frameon=False)
    save_plot(fig, output / "robustness_comparison")


def main():
    design, scores, audit = load_study()
    test = design.subset.eq("test")
    test_labels = design.loc[test, "Churn"].to_numpy(dtype=bool)
    train_labels = design.loc[~test, "Churn"].to_numpy()
    rows, test_scores, test_classes, test_exports = [], [], [], []
    for (model, setting), values in scores.items():
        threshold, oof_f1 = select_threshold(model, train_labels, values[~test])
        test_prob = values[test]
        test_exports.append(design.loc[test, ["row_id", "Churn"]].assign(
            model=model, feature_set=setting, score=test_prob,
            fixed_prediction=(test_prob >= 0.5).astype(int),
            optimized_threshold=threshold, optimized_prediction=(test_prob >= threshold).astype(int),
        ))
        for policy, cutoff, selection_f1 in (("fixed", 0.5, np.nan), ("optimized", threshold, oof_f1)):
            test_scores.append(test_prob)
            test_classes.append(test_prob >= cutoff)
            rows.append({
                "model": model, "feature_set": setting, "threshold_policy": policy,
                "threshold": cutoff, "oof_f1_at_selected_threshold": selection_f1,
            })

    scores_matrix = np.column_stack(test_scores)
    predictions = np.column_stack(test_classes)
    observed = performance(test_labels, scores_matrix, predictions)
    samples = paired_bootstrap(test_labels, scores_matrix, predictions)
    table = pd.DataFrame(rows).assign(**observed).merge(
        audit[["model", "feature_set", "n_features", "train_rows", "test_rows"]],
        on=["model", "feature_set"], validate="many_to_one",
    ).set_index(["model", "feature_set", "threshold_policy"])
    index = {key: index for index, key in enumerate(table.index)}
    model_pairs = [
        ((a, "full", policy), (b, "full", policy),
         {"difference_direction": f"{b} minus {a}", "threshold_policy": policy})
        for a, b in combinations(MODELS, 2) for policy in POLICIES
    ]
    robustness_pairs = [
        ((model, "full", policy), (model, "reduced", policy),
         {"model": model, "threshold_policy": policy, "difference_direction": "reduced minus full"})
        for model in MODELS for policy in POLICIES
    ]
    comparison = RESULTS / "model_comparison"
    robustness = RESULTS / "robustness_analysis"
    for output in (comparison, robustness):
        output.mkdir(parents=True, exist_ok=True)
    table.query("feature_set == 'full'").to_csv(comparison / "model_performance.csv")
    table.to_csv(robustness / "feature_set_performance.csv")
    audit.to_csv(comparison / "design_validation.csv", index=False)
    pd.concat(test_exports, ignore_index=True).to_csv(robustness / "test_predictions.csv", index=False)
    difference_table(model_pairs, index, observed, samples).to_csv(
        comparison / "paired_model_differences.csv", index=False,
    )
    deltas = difference_table(robustness_pairs, index, observed, samples)
    deltas.to_csv(robustness / "paired_feature_differences.csv", index=False)
    write_robustness_summary(audit, deltas, robustness / "audit_summary.txt")
    plot_comparison(table, test_labels, scores_matrix, index, comparison)
    plot_robustness(deltas, robustness)
    print(f"Verified {len(design):,} row assignments, {test.sum():,} shared test rows and 23/22 features.")


if __name__ == "__main__":
    main()
