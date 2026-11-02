import re
from collections import Counter

from src.utils import load_json, save_json

# слова (з апострофом), числа і кілька символів, які часто трапляються в спамі
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[£$!?]")

PAD, UNK = 0, 1


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, texts, min_freq=2):
        counts = Counter(t for text in texts for t in tokenize(text))
        words = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
        return cls(["<pad>", "<unk>"] + words)

    def __len__(self):
        return len(self.itos)

    def encode(self, text, max_len):
        ids = [self.stoi.get(t, UNK) for t in tokenize(text)][:max_len]
        return ids or [UNK]  # порожнє повідомлення (напр. лише смайлик)

    def unk_rate(self, texts):
        tokens = [t for text in texts for t in tokenize(text)]
        return sum(t not in self.stoi for t in tokens) / max(len(tokens), 1)

    def save(self, path):
        save_json(self.itos, path)

    @classmethod
    def load(cls, path):
        return cls(load_json(path))
