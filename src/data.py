"""Завантаження SMS Spam Collection, дедуплікація і поділ на train/val/test."""
import io
import re
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ROOT, RESULTS, save_json

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SPLIT_SEED = 42


def download():
    path = RAW / "SMSSpamCollection"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extract("SMSSpamCollection", RAW)
    return path


def load_raw():
    # quoting=3: лапки в SMS не обробляємо, інакше pandas склеює рядки
    df = pd.read_csv(download(), sep="\t", header=None, names=["label", "text"],
                     quoting=3, encoding="utf-8", encoding_errors="replace")
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def norm_key(text):
    return " ".join(text.lower().split())


def load_split(name):
    return pd.read_csv(PROCESSED / f"{name}.csv", keep_default_na=False)


def main():
    df = load_raw()
    stats = {"raw_total": len(df), "raw_spam": int(df.y.sum())}
    assert stats["raw_total"] == 5574 and stats["raw_spam"] == 747, stats

    df["key"] = df["text"].map(norm_key)
    labels_per_key = df.groupby("key")["y"].nunique()
    conflicts = set(labels_per_key[labels_per_key > 1].index)
    stats["label_conflicts"] = len(conflicts)
    df = df[~df["key"].isin(conflicts)]

    before = len(df)
    df = df.drop_duplicates(subset="key", keep="first").reset_index(drop=True)
    stats["duplicates_removed"] = before - len(df)
    stats["dedup_total"] = len(df)
    stats["dedup_spam"] = int(df.y.sum())

    # "майже дублікати": однаковий текст з різними числами (лише статистика, не видаляємо)
    df["key_num"] = df["key"].map(lambda s: re.sub(r"\d+", "<num>", s))
    stats["near_duplicates"] = int(df.duplicated(subset="key_num").sum())

    train, temp = train_test_split(df, test_size=0.30, stratify=df.y, random_state=SPLIT_SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp.y, random_state=SPLIT_SEED)

    for a, b in [(train, val), (train, test), (val, test)]:
        assert set(a.key).isdisjoint(b.key)

    # скільки повідомлень val/test мають "майже дубліката" в train
    train_num = set(train.key_num)
    for name, part in [("val", val), ("test", test)]:
        stats[f"{name}_near_dup_in_train"] = int(part.key_num.isin(train_num).sum())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[["label", "y", "text"]].to_csv(PROCESSED / f"{name}.csv", index=False)
        stats[f"{name}_total"] = len(part)
        stats[f"{name}_spam"] = int(part.y.sum())

    save_json(stats, RESULTS / "data_stats.json")
    for k, v in stats.items():
        print(f"{k:>24}: {v}")


if __name__ == "__main__":
    main()
