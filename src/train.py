"""Один запуск навчання нейромережі: python -m src.train --config base --seed 0"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data import load_split
from src.evaluate import compute_metrics
from src.models import build_model, predict_proba
from src.utils import load_config, run_dir, save_json, set_seed
from src.vocab import Vocab

torch.set_num_threads(4)


def subsample(df, fraction, seed):
    """Стратифікована частка train (для експерименту з обсягом даних)."""
    if fraction >= 1:
        return df
    return df.groupby("y", group_keys=False).sample(frac=fraction, random_state=seed)


def collate(batch):
    ids, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in ids])
    x = torch.zeros(len(ids), int(lengths.max()), dtype=torch.long)  # 0 = PAD
    for i, seq in enumerate(ids):
        x[i, :len(seq)] = torch.tensor(seq)
    return x, lengths, torch.tensor(ys, dtype=torch.float)


def make_loader(df, vocab, max_len, batch_size, shuffle=False, generator=None):
    data = [(vocab.encode(t, max_len), y) for t, y in zip(df.text, df.y)]
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=generator)


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, lengths, y in loader:
            total += loss_fn(model(x, lengths), y).item() * len(y)
            n += len(y)
    return total / n


def run(cfg, seed, out=None):
    out = out or run_dir(cfg["name"], seed)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    tc, dc = cfg["train"], cfg["data"]

    train_df = subsample(load_split("train"), tc.get("train_fraction", 1.0), seed)
    val_df = load_split("val")
    vocab = Vocab.build(train_df.text, dc["min_freq"])

    g = torch.Generator().manual_seed(seed)
    train_loader = make_loader(train_df, vocab, dc["max_len"], tc["batch_size"], shuffle=True, generator=g)
    val_loader = make_loader(val_df, vocab, dc["max_len"], 256)

    model = build_model(cfg, len(vocab))
    pos_weight = None
    if tc["class_weight"]:
        pos_weight = torch.tensor((train_df.y == 0).sum() / (train_df.y == 1).sum(), dtype=torch.float)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))

    log, best_loss, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
    start = time.time()
    for epoch in range(1, tc["max_epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for x, lengths, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)

        val_loss = evaluate_loss(model, val_loader, loss_fn)
        p_val = predict_proba(model, val_loader)
        m = compute_metrics(val_df.y, p_val)
        log.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                    "val_f1": m["f1"], "val_pr_auc": m["pr_auc"]})

        # early stopping: пам'ятаємо найкращу епоху за val loss
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, bad = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break

    model.load_state_dict(best_state)
    p_val = predict_proba(model, val_loader)
    metrics = {
        "name": cfg["name"], "seed": seed,
        "best_epoch": best_epoch, "epochs_run": len(log),
        "train_size": len(train_df), "vocab_size": len(vocab),
        "unk_rate_val": vocab.unk_rate(val_df.text),
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_time_s": round(time.time() - start, 2),
        "val": compute_metrics(val_df.y, p_val),
    }

    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    np.save(out / "val_probs.npy", p_val)
    torch.save(model.state_dict(), out / "model.pt")
    vocab.save(out / "vocab.json")
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
    print(f"{m['name']} seed{m['seed']}: epoch {m['best_epoch']}/{m['epochs_run']}, "
          f"val PR-AUC {v['pr_auc']:.4f}, F1 {v['f1']:.4f}, recall@FPR {v['recall_at_fpr']:.4f}")
