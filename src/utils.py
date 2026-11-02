import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"


def _merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name_or_path):
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIGS / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # конфіг може наслідувати інший і перевизначати лише частину полів
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(parent), cfg)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_dir(name, seed):
    return RESULTS / name / f"seed{seed}"
