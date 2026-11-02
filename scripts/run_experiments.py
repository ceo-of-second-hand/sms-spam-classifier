"""Запуск серії конфігів x seed. Готові запуски (є metrics.json) пропускаються.

python -m scripts.run_experiments base e3_lr_1e-4 e3_lr_1e-2
"""
import argparse

from src import baseline, train
from src.utils import load_config, run_dir

SEEDS = [0, 1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for name in args.configs:
        cfg = load_config(name)
        is_baseline = cfg.get("kind") == "baseline"
        seeds = args.seeds
        # LogReg на повному train детермінована, різні seed дали б однаковий результат
        if is_baseline and cfg.get("train_fraction", 1.0) >= 1:
            seeds = [0]
        for seed in seeds:
            if (run_dir(cfg["name"], seed) / "metrics.json").exists() and not args.force:
                print(f"{cfg['name']} seed{seed}: вже є, пропускаю")
                continue
            m = (baseline if is_baseline else train).run(cfg, seed)
            v = m["val"]
            extra = f", epoch {m['best_epoch']}/{m['epochs_run']}, {m['train_time_s']}s" if "best_epoch" in m else ""
            print(f"{cfg['name']} seed{seed}: PR-AUC {v['pr_auc']:.4f}, F1 {v['f1']:.4f}, "
                  f"recall@FPR {v['recall_at_fpr']:.4f}{extra}")


if __name__ == "__main__":
    main()
