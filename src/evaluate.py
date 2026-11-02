import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

MAX_FPR = 0.005  # не більше 0,5% нормальних повідомлень у спамі


def recall_at_fpr(y, p, max_fpr=MAX_FPR):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[fpr <= max_fpr].max())


def choose_threshold(y, p, max_fpr=MAX_FPR):
    """Поріг з максимальним recall серед тих, де FPR <= max_fpr (обирається на val)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= max_fpr)[0]
    best = ok[np.argmax(tpr[ok])]  # roc_curve іде від високих порогів до низьких
    return float(min(thr[best], 1.0))


def compute_metrics(y, p, threshold=0.5):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": fbeta_score(y, pred, beta=1, zero_division=0),
        "f05": fbeta_score(y, pred, beta=0.5, zero_division=0),
        "accuracy": float((pred == y).mean()),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "recall_at_fpr": recall_at_fpr(y, p),
        "fpr": fp / (fp + tn),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def errors_table(df, p, threshold):
    out = df[["y", "text"]].copy()
    out["p_spam"] = p
    out["pred"] = (p >= threshold).astype(int)
    out = out[out.pred != out.y].copy()
    out["error"] = np.where(out.y == 0, "FP", "FN")
    return out.sort_values(["error", "p_spam"], ascending=[True, False])
