import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def make_head(in_dim, hidden, dropout):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))


class MeanPoolClassifier(nn.Module):
    """Embedding -> середнє векторів слів (без PAD) -> Dense + ReLU -> Dropout -> Dense(1)."""

    def __init__(self, vocab_size, emb_dim, hidden, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.head = make_head(emb_dim, hidden, dropout)

    def forward(self, x, lengths):
        mask = (x != 0).unsqueeze(-1).float()
        summed = (self.emb(x) * mask).sum(dim=1)
        mean = summed / lengths.unsqueeze(-1).float()
        return self.head(mean).squeeze(-1)  # logit, сигмоїда всередині функції втрат


class LSTMClassifier(nn.Module):
    """Embedding -> LSTM (читає слова по черзі) -> останній стан -> та сама голова."""

    def __init__(self, vocab_size, emb_dim, hidden, dropout, lstm_hidden=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden, batch_first=True)
        self.head = make_head(lstm_hidden, hidden, dropout)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1]).squeeze(-1)


def build_model(cfg, vocab_size):
    m = cfg["model"]
    if m["type"] == "meanpool":
        return MeanPoolClassifier(vocab_size, m["emb_dim"], m["hidden"], m["dropout"])
    if m["type"] == "lstm":
        return LSTMClassifier(vocab_size, m["emb_dim"], m["hidden"], m["dropout"], m.get("lstm_hidden", 64))
    raise ValueError(m["type"])


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    return torch.cat([torch.sigmoid(model(x, n)) for x, n, _ in loader]).numpy()
