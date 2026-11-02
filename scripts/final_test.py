"""Фінальна оцінка на test. Запускається ОДИН раз, після всіх експериментів.

Поріг для кожної моделі обирається на val (max recall при FPR <= 0,5%), test лише вимірюється.
"""
import pickle
import shutil

import numpy as np
import pandas as pd
import torch

from src.data import load_split
from src.evaluate import choose_threshold, compute_metrics, errors_table
from src.models import build_model, predict_proba
from src.train import make_loader
from src.utils import ROOT, RESULTS, load_config, run_dir, save_json
from src.vocab import Vocab

NN, LOGREG, MAJORITY = "final", "baseline_word", "baseline_majority"
SEEDS = [0, 1, 2, 3, 4]


def nn_test_probs(seed, test_df):
    d = run_dir(NN, seed)
    cfg = load_config(d / "config.yaml")
    vocab = Vocab.load(d / "vocab.json")
    model = build_model(cfg, len(vocab))
    model.load_state_dict(torch.load(d / "model.pt"))
    loader = make_loader(test_df, vocab, cfg["data"]["max_len"], 256)
    return predict_proba(model, loader)


def sklearn_test_probs(name, test_df):
    with open(run_dir(name, 0) / "model.pkl", "rb") as f:
        return pickle.load(f).predict_proba(test_df.text)


def main():
    val_df, test_df = load_split("val"), load_split("test")
    rows, probs, thresholds, errors = [], {"y": test_df.y}, {}, []

    candidates = [(f"nn_seed{s}", "nn_meanpool", run_dir(NN, s), lambda s=s: nn_test_probs(s, test_df)) for s in SEEDS]
    candidates += [("logreg", "tfidf_logreg", run_dir(LOGREG, 0), lambda: sklearn_test_probs(LOGREG, test_df)),
                   ("majority", "majority", run_dir(MAJORITY, 0), lambda: sklearn_test_probs(MAJORITY, test_df))]

    for key, model_name, d, get_probs in candidates:
        t = choose_threshold(val_df.y, np.load(d / "val_probs.npy"))
        p = get_probs()
        thresholds[key] = t
        probs[key] = p
        rows.append({"model": model_name, "run": key, **compute_metrics(test_df.y, p, t)})
        if key in ("nn_seed0", "logreg"):
            e = errors_table(test_df, p, t)
            e.insert(0, "model", model_name)
            errors.append(e)

    table = pd.DataFrame(rows)
    table = table.astype({c: float for c in table.columns if c not in ("model", "run")})
    nn =table[table.model == "nn_meanpool"].drop(columns=["model", "run"])
    for stat in ("mean", "std"):
        table.loc[len(table)] = {"model": "nn_meanpool", "run": stat, **getattr(nn, stat)().to_dict()}

    table.to_csv(RESULTS / "test_metrics.csv", index=False, float_format="%.4f")
    pd.concat(errors).to_csv(RESULTS / "test_errors.csv", index=False)
    pd.DataFrame(probs).to_csv(RESULTS / "test_probs.csv", index=False, float_format="%.6f")
    save_json(thresholds, RESULTS / "thresholds.json")

    # модель для demo.py: фінальна нейромережа, seed 0
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    src = run_dir(NN, 0)
    shutil.copy(src / "model.pt", out / "final_model.pt")
    shutil.copy(src / "vocab.json", out / "vocab.json")
    shutil.copy(src / "config.yaml", out / "config.yaml")
    save_json({"threshold": thresholds["nn_seed0"]}, out / "threshold.json")

    cols = ["model", "run", "threshold", "precision", "recall", "f1", "f05", "pr_auc", "recall_at_fpr", "fp", "fn", "fpr"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
