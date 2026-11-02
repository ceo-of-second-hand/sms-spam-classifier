"""Графіки для звіту -> results/figures/"""
import pickle

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data import load_split
from src.evaluate import choose_threshold, compute_metrics
from src.utils import RESULTS, load_json, run_dir
from src.vocab import tokenize

FIG = RESULTS / "figures"
SEEDS = [0, 1, 2, 3, 4]
BLUE, ORANGE, GREEN, RED, GREY = "#2E75B6", "#E07B00", "#2E7D32", "#C62828", "#888888"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def val_metric(name, key):
    return np.array([load_json(run_dir(name, s) / "metrics.json")["val"][key]
                     for s in SEEDS if (run_dir(name, s) / "metrics.json").exists()])


def data_overview():
    stats = load_json(RESULTS / "data_stats.json")
    train = load_split("train")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    parts = ["train", "val", "test"]
    ham = [stats[f"{p}_total"] - stats[f"{p}_spam"] for p in parts]
    spam = [stats[f"{p}_spam"] for p in parts]
    a.bar(parts, ham, color=BLUE, label="не спам")
    a.bar(parts, spam, bottom=ham, color=ORANGE, label="спам")
    for i, p in enumerate(parts):
        a.text(i, ham[i] + spam[i] + 40, f"{spam[i] / (ham[i] + spam[i]):.1%} спаму", ha="center", fontsize=8)
    a.set_title(f"Поділ після дедуплікації ({stats['duplicates_removed']} дублікатів прибрано)")
    a.set_ylabel("повідомлень")
    a.legend(frameon=False)
    lengths = train.text.map(lambda t: len(tokenize(t)))
    bins = np.arange(0, 80, 2)
    b.hist(lengths[train.y == 0], bins=bins, alpha=0.7, color=BLUE, label="не спам", density=True)
    b.hist(lengths[train.y == 1], bins=bins, alpha=0.7, color=ORANGE, label="спам", density=True)
    b.axvline(50, color=GREY, ls="--", lw=1)
    b.text(51, b.get_ylim()[1] * 0.85, "max_len = 50", fontsize=8, color=GREY)
    b.set_title("Довжина повідомлень у train (токенів)")
    b.legend(frameon=False)
    save(fig, "data_overview.png")


def mean_curve(name, col):
    logs = [pd.read_csv(run_dir(name, s) / "train_log.csv")[col].values for s in SEEDS]
    n = max(len(l) for l in logs)
    arr = np.full((len(logs), n), np.nan)
    for i, l in enumerate(logs):
        arr[i, :len(l)] = l
    return np.arange(1, n + 1), np.nanmean(arr, 0), np.nanstd(arr, 0)


def learning_curves():
    configs = [("e3_lr_1e-4", "lr = 1e-4"), ("base", "lr = 1e-3 (базовий)"), ("e3_lr_1e-2", "lr = 1e-2")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, (name, title) in zip(axes, configs):
        for col, color, label in [("train_loss", BLUE, "train loss"), ("val_loss", ORANGE, "val loss")]:
            x, m, s = mean_curve(name, col)
            ax.plot(x, m, color=color, label=label)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("епоха")
    axes[0].set_ylabel("loss (середнє ± std по 5 seed)")
    axes[0].legend(frameon=False)
    save(fig, "learning_curves_lr.png")


def seed_stability():
    configs = [("base", "base"), ("e3_lr_1e-4", "lr 1e-4"), ("e3_lr_1e-2", "lr 1e-2"), ("e4_emb16", "d 16"),
               ("e4_emb128", "d 128"), ("e5_dropout0", "dropout 0"), ("e5_dropout05", "dropout 0,5"),
               ("e6_classweight", "class weight"), ("e8_lstm", "LSTM")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, title in [(axes[0], "pr_auc", "val PR-AUC"), (axes[1], "recall_at_fpr", "val recall при FPR ≤ 0,5%")]:
        for i, (name, _) in enumerate(configs):
            v = val_metric(name, key)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=14, color=BLUE, zorder=3)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color=RED, lw=2)
        base = val_metric("base", key)
        ax.axhspan(base.mean() - base.std(), base.mean() + base.std(), color=GREY, alpha=0.15, lw=0)
        ax.set_xticks(range(len(configs)), [c[1] for c in configs], rotation=40, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("точка = 1 seed, риска = середнє", fontsize=8)
    save(fig, "seed_stability.png")


def sample_efficiency():
    fracs = [0.10, 0.25, 0.50, 1.0]
    nn = ["e7_frac010", "e7_frac025", "e7_frac050", "final"]
    lr = ["e7_logreg_frac010", "e7_logreg_frac025", "e7_logreg_frac050", "baseline_word"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key, title in [(axes[0], "pr_auc", "val PR-AUC"), (axes[1], "recall_at_fpr", "val recall при FPR ≤ 0,5%")]:
        for names, color, label in [(nn, BLUE, "нейромережа (усереднення)"), (lr, ORANGE, "TF-IDF + LogReg")]:
            vals = [val_metric(n, key) for n in names]
            ax.errorbar([f * 100 for f in fracs], [v.mean() for v in vals], yerr=[v.std() for v in vals],
                        marker="o", capsize=3, color=color, label=label)
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
        ax.set_xlabel("частка train (log-шкала)")
        ax.set_title(title)
    axes[0].legend(frameon=False)
    save(fig, "sample_efficiency.png")


def val_curves():
    val = load_split("val")
    models = [("final", 0, BLUE, "нейромережа (усереднення)"), ("e8_lstm", 0, GREEN, "LSTM"),
              ("baseline_word", 0, ORANGE, "TF-IDF(слова) + LogReg"), ("baseline_char", 0, RED, "TF-IDF(n-грами) + LogReg")]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.6))
    for name, seed, color, label in models:
        p = np.load(run_dir(name, seed) / "val_probs.npy")
        prec, rec, _ = precision_recall_curve(val.y, p)
        a.plot(rec, prec, color=color, label=label)
        fpr, tpr, _ = roc_curve(val.y, p)
        b.plot(fpr, tpr, color=color)
        c.plot(fpr, tpr, color=color)
        if name in ("final", "baseline_word"):
            m = compute_metrics(val.y, p, choose_threshold(val.y, p))
            a.scatter(m["recall"], m["precision"], color=color, s=30, zorder=4, edgecolor="black")
            c.scatter(m["fpr"], m["recall"], color=color, s=30, zorder=4, edgecolor="black")
    a.set(xlabel="recall", ylabel="precision", title="PR-криві (val); точка = обраний поріг", ylim=(0.5, 1.02))
    a.legend(frameon=False, fontsize=8, loc="lower left")
    b.set(xlabel="FPR", ylabel="recall (TPR)", title="ROC (val), уся крива")
    c.axvline(0.005, color=GREY, ls="--", lw=1)
    c.set(xlabel="FPR", title="ROC, зум на FPR ≤ 2%", xlim=(0, 0.02), ylim=(0.5, 1.0))
    c.set_xticks([0, 0.005, 0.01, 0.015, 0.02], ["0", "0,5%", "1%", "1,5%", "2%"])
    save(fig, "val_pr_roc.png")


def class_weight_effect():
    val = load_split("val")
    rows = []
    for name, label in [("base", "без ваг"), ("e6_classweight", "з pos_weight")]:
        for s in SEEDS:
            p = np.load(run_dir(name, s) / "val_probs.npy")
            for t_label, t in [("поріг 0,5", 0.5), ("обраний поріг", choose_threshold(val.y, p))]:
                m = compute_metrics(val.y, p, t)
                rows.append({"cfg": label, "t": t_label, "fp": m["fp"], "recall": m["recall"]})
    df = pd.DataFrame(rows).groupby(["t", "cfg"]).agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
    for ax, key, title in [(axes[0], "fp", "FP на val (з 677 нормальних)"), (axes[1], "recall", "recall на val")]:
        x = np.arange(2)
        for j, (cfg, color) in enumerate([("без ваг", BLUE), ("з pos_weight", ORANGE)]):
            vals = [df.loc[(t, cfg), (key, "mean")] for t in ["поріг 0,5", "обраний поріг"]]
            errs = [df.loc[(t, cfg), (key, "std")] for t in ["поріг 0,5", "обраний поріг"]]
            ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, yerr=errs, capsize=3, color=color, label=cfg)
        ax.set_xticks(x, ["поріг 0,5", "обраний поріг\n(FPR ≤ 0,5%)"])
        ax.set_title(title)
    axes[0].legend(frameon=False)
    save(fig, "class_weight_effect.png")


def confusion_test():
    df = pd.read_csv(RESULTS / "test_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, run, title in [(axes[0], "nn_seed0", "Нейромережа (seed 0)"), (axes[1], "logreg", "TF-IDF + LogReg")]:
        r = df[df.run == run].iloc[0]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]]).astype(int)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["не спам", "спам"])
        ax.set_yticks([0, 1], ["не спам", "спам"])
        ax.set_xlabel("модель каже")
        ax.set_ylabel("насправді")
        ax.set_title(f"{title}\nпоріг {r.threshold:.2f}")
    save(fig, "confusion_test.png")


def top_words():
    with open(run_dir("baseline_word", 0) / "model.pkl", "rb") as f:
        model = pickle.load(f)
    names = model.vec.get_feature_names_out()
    coef = model.clf.coef_[0]
    order = np.argsort(coef)
    idx = np.concatenate([order[:12], order[-12:]])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(idx)), coef[idx], color=[BLUE if c < 0 else ORANGE for c in coef[idx]])
    ax.set_yticks(range(len(idx)), names[idx])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Ваги LogReg: слова \"за спам\" (помаранчеві) і \"проти\" (сині)")
    save(fig, "top_words_logreg.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in [data_overview, learning_curves, seed_stability, sample_efficiency,
              val_curves, class_weight_effect, confusion_test, top_words]:
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    main()
