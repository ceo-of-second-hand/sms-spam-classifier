"""Базові лінії: "завжди не спам" і TF-IDF + логістична регресія.

python -m src.baseline --config baseline_word --seed 0
"""
import argparse
import pickle

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_split
from src.evaluate import compute_metrics
from src.train import subsample
from src.utils import load_config, run_dir, save_json
from src.vocab import tokenize


def make_vectorizer(features):
    if features == "word":
        return TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=2)
    if features == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=2)
    raise ValueError(features)


class Majority:
    """Завжди каже "не спам"; ймовірність спаму = частка спаму в train."""

    def fit(self, texts, y):
        self.prior = float(np.mean(y))
        return self

    def predict_proba(self, texts):
        return np.full(len(texts), self.prior)


class TfidfLogReg:
    def __init__(self, features, C):
        self.vec = make_vectorizer(features)
        self.clf = LogisticRegression(C=C, max_iter=2000)

    def fit(self, texts, y):
        self.clf.fit(self.vec.fit_transform(texts), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]

    def top_words(self, k=20):
        names = self.vec.get_feature_names_out()
        order = np.argsort(self.clf.coef_[0])
        return {"spam": [names[i] for i in order[::-1][:k]], "ham": [names[i] for i in order[:k]]}


def build(cfg):
    if cfg["features"] == "majority":
        return Majority()
    return TfidfLogReg(cfg["features"], cfg.get("C", 1.0))


def run(cfg, seed, out=None):
    out = out or run_dir(cfg["name"], seed)
    out.mkdir(parents=True, exist_ok=True)
    train_df = subsample(load_split("train"), cfg.get("train_fraction", 1.0), seed)
    val_df = load_split("val")

    model = build(cfg).fit(train_df.text, train_df.y)
    p_val = model.predict_proba(val_df.text)
    metrics = {"name": cfg["name"], "seed": seed, "train_size": len(train_df),
               "val": compute_metrics(val_df.y, p_val)}
    if isinstance(model, TfidfLogReg):
        metrics["n_features"] = len(model.vec.vocabulary_)
        save_json(model.top_words(), out / "top_words.json")

    np.save(out / "val_probs.npy", p_val)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    save_json(metrics, out / "metrics.json")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = run(load_config(args.config), args.seed)
    v = m["val"]
    print(f"{m['name']} seed{m['seed']}: val PR-AUC {v['pr_auc']:.4f}, F1 {v['f1']:.4f}, "
          f"recall@FPR {v['recall_at_fpr']:.4f}")
