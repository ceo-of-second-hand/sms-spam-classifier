"""Зведена таблиця експериментів на validation: mean ± std по seed."""
import pandas as pd

from src.utils import RESULTS, load_json

ORDER = ["baseline_majority", "baseline_word", "baseline_char",
         "base", "e3_lr_1e-4", "e3_lr_1e-2", "e4_emb16", "e4_emb128",
         "e5_dropout0", "e5_dropout05", "e6_classweight", "final",
         "e7_frac010", "e7_frac025", "e7_frac050",
         "e7_logreg_frac010", "e7_logreg_frac025", "e7_logreg_frac050", "e8_lstm"]
METRICS = ["pr_auc", "recall_at_fpr", "f1", "f05", "precision", "recall", "fp", "roc_auc"]


def collect():
    rows = []
    for path in RESULTS.glob("*/seed*/metrics.json"):
        m = load_json(path)
        row = {"experiment": m["name"], "seed": m["seed"], "train_size": m["train_size"],
               "best_epoch": m.get("best_epoch")}
        row.update({k: m["val"][k] for k in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby("experiment")
    out = pd.DataFrame({"runs": g.size(), "train_size": g.train_size.first()})
    for k in METRICS + ["best_epoch"]:
        out[k + "_mean"] = g[k].mean()
        out[k + "_std"] = g[k].std().fillna(0)
    order = [e for e in ORDER if e in out.index] + sorted(set(out.index) - set(ORDER))
    return out.loc[order]


def to_markdown(s):
    lines = ["| Експеримент | Запусків | PR-AUC | Recall@FPR≤0,5% | F1 (t=0,5) | F0,5 (t=0,5) | FP (t=0,5) | Епоха |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in s.iterrows():
        def ms(k, d=4):
            return f"{r[k + '_mean']:.{d}f} ± {r[k + '_std']:.{d}f}"
        epoch = ms("best_epoch", 1) if pd.notna(r["best_epoch_mean"]) else "-"
        lines.append(f"| {name} | {int(r.runs)} | {ms('pr_auc')} | {ms('recall_at_fpr')} | {ms('f1')} | "
                     f"{ms('f05')} | {ms('fp', 1)} | {epoch} |")
    return "\n".join(lines)


def main():
    s = summarize(collect())
    s.to_csv(RESULTS / "summary.csv", float_format="%.4f")
    md = ("# Результати на validation (mean ± std по seed)\n\n"
          "F1, F0,5 і FP тут при порозі 0,5; фінальний поріг обирається окремо.\n\n" + to_markdown(s) + "\n")
    (RESULTS / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
