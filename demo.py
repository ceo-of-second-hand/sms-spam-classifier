"""Демо: python demo.py "текст повідомлення"  або просто python demo.py для вводу в циклі."""
import sys

import torch

from src.models import build_model
from src.utils import ROOT, load_config, load_json
from src.vocab import Vocab

MODELS = ROOT / "models"


def load():
    cfg = load_config(MODELS / "config.yaml")
    vocab = Vocab.load(MODELS / "vocab.json")
    model = build_model(cfg, len(vocab))
    model.load_state_dict(torch.load(MODELS / "final_model.pt"))
    model.eval()
    return model, vocab, cfg["data"]["max_len"], load_json(MODELS / "threshold.json")["threshold"]


def predict(text, model, vocab, max_len):
    ids = vocab.encode(text, max_len)
    with torch.no_grad():
        logit = model(torch.tensor([ids]), torch.tensor([len(ids)]))
    return torch.sigmoid(logit).item()


def show(text, *loaded):
    model, vocab, max_len, threshold = loaded
    p = predict(text, model, vocab, max_len)
    verdict = "СПАМ" if p >= threshold else "не спам"
    print(f"{verdict:8} p(спам) = {p:.3f}  (поріг {threshold:.2f})")


if __name__ == "__main__":
    loaded = load()
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]), *loaded)
    else:
        print("Введіть повідомлення англійською (порожній рядок - вихід)")
        while text := input("> ").strip():
            show(text, *loaded)
