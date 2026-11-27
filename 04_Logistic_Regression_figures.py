"""Publication figures. PDF text stays editable; PNGs render at 600 dpi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

INK = "#192F3D"
GRID = "#DCE4E6"
COLORS = ["#68757E", "#205579", "#B35629", "#167A6D", "#74618C"]
NAMES = ["Majority", "Default LR", "Balanced LR", "Tuned threshold", "No total charges"]


def finish(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def make_figures(metrics, curve, directory):
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#83939B",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.axisbelow": True, "axes.linewidth": 0.65,
                         "xtick.major.size": 3, "ytick.major.size": 0})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.85))
    fig.subplots_adjust(left=0.23, right=0.98, wspace=0.18, bottom=0.23, top=0.87)
    y = np.arange(len(metrics))[::-1]
    for ax, metric, label in zip(axes, ["auc", "f1"], ["a   ROC-AUC", "b   F1 score"]):
        for i, (_, row) in enumerate(metrics.iterrows()):
            lo, hi, point = row[metric+"_low"], row[metric+"_high"], row[metric]
            ax.hlines(y[i], lo, hi, lw=1.7, color=COLORS[i])
            ax.plot([lo, hi], [y[i], y[i]], "|", ms=5, color=COLORS[i])
            ax.plot(point, y[i], "o", ms=5, color=COLORS[i])
        ax.set_title(label, loc="left", fontweight="bold", fontsize=10.5)
        ax.set_ylim(-0.6, 4.6)
        ax.set_yticks(y, [f"E{i+1}  {s}" for i,s in enumerate(NAMES)] if metric=="auc" else [])
        ax.set_xlim((0.47, 0.93) if metric=="auc" else (-0.035, 0.75))
        ax.xaxis.set_major_locator(MultipleLocator(0.1 if metric=="auc" else 0.2))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_xlabel("Test estimate (95% CI)")
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.7)
    finish(fig, directory / "fig1_performance")

    fig, (ax, right) = plt.subplots(1, 2, figsize=(7.5, 3.25))
    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.18, top=0.84, wspace=0.45)
    ax.plot(curve.threshold, curve.f1, color="#73868D", lw=1.15)
    for eid, ypos in [("E4", 0.12), ("E2", 0.24)]:
        t = metrics.loc[eid, "threshold"]
        # At an arbitrary cutoff, use the next observed score >= cutoff.
        i = np.searchsorted(curve.threshold, t)
        value = curve.f1.iloc[i]
        color = COLORS[int(eid[1]) - 1]
        ax.vlines(t, 0, value, color=color, ls="--", lw=0.9)
        ax.plot(t, value, "o", color=color, ms=4.5)
        ax.annotate(f"{eid}  {t:.3f}", (t, ypos), xytext=(-5 if eid=="E4" else 5, 0),
                    textcoords="offset points", ha="right" if eid=="E4" else "left",
                    color=color, fontsize=8, fontweight="bold")
    ax.set(xlim=(0,1), ylim=(0,0.74), xlabel="Classification threshold", ylabel="Churn-class F1")
    ax.set_title("a   Training out-of-fold F1", loc="left", fontweight="bold", fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.7)
    y = np.array([1, 0])
    for offset, eid in [(0.17, "E2"), (-0.17, "E4")]:
        counts = metrics.loc[eid, ["TP","FP"]].to_numpy(dtype=int)
        color = COLORS[int(eid[1])-1]
        bars = right.barh(y + offset, counts, height=0.26, color=color, label=eid)
        right.bar_label(bars, padding=3, fontsize=8)
    right.set_yticks(y, ["True positives","False positives"])
    right.set_xlabel("Test customers flagged")
    right.set_ylim(-0.56, 1.55)
    right.set_xlim(0, metrics.loc[["E2","E4"],["TP","FP"]].max().max()*1.22)
    right.set_title("b   Alerts on the test set", loc="left", fontweight="bold", fontsize=10)
    right.legend(loc="lower right", frameon=False, ncols=2, fontsize=8, handlelength=1)
    right.grid(axis="x", color=GRID, lw=0.7)
    finish(fig, directory / "fig2_threshold")
