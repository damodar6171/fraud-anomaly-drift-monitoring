"""Metrics, cost-sensitive threshold selection, plots."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             confusion_matrix)
from . import config as C


def metrics(y, s):
    return {"pr_auc": float(average_precision_score(y, s)),
            "roc_auc": float(roc_auc_score(y, s)),
            "no_skill_pr_auc": float(np.mean(y))}


def recall_at_precision(y, s, p_min=0.5):
    p, r, _ = precision_recall_curve(y, s)
    ok = p >= p_min
    return float(r[ok].max()) if ok.any() else 0.0


def total_cost(y, flag, amount):
    """FN (missed fraud) costs amount + FN_FIXED_COST; FP (blocked legit) costs
    FP_FIXED_COST + FP_AMOUNT_FRACTION*amount."""
    y, flag, amount = map(np.asarray, (y, flag, amount))
    fn = (y == 1) & (~flag)
    fp = (y == 0) & flag
    return float((amount[fn] + C.FN_FIXED_COST).sum()
                 + (C.FP_FIXED_COST + C.FP_AMOUNT_FRACTION * amount[fp]).sum())


def cost_curve(y, s, amount, n_grid=400):
    grid = np.unique(np.quantile(s, np.linspace(0.5, 0.99999, n_grid)))
    costs = np.array([total_cost(y, s >= t, amount) for t in grid])
    return grid, costs


def best_threshold(y_val, s_val, amount_val, tol=0.10):
    """Cost-minimising threshold on a validation window.
    With very few frauds the cost curve is jagged, so instead of the single argmin we take the
    MEDIAN threshold of the near-optimal plateau (cost <= (1+tol) * min cost) -> more stable."""
    grid, costs = cost_curve(y_val, s_val, amount_val)
    plateau = grid[costs <= costs.min() * (1 + tol)]
    return float(np.median(plateau)), grid, costs


def confusion_at(y, s, thr):
    tn, fp, fn, tp = confusion_matrix(y, s >= thr, labels=[0, 1]).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)))


def plot_pr(curves: dict, y, path):
    plt.figure(figsize=(6, 4.5))
    for name, s in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        plt.plot(r, p, label=f"{name} (AP={average_precision_score(y, s):.3f})")
    plt.axhline(np.mean(y), ls="--", c="gray", label=f"no-skill ({np.mean(y):.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall (held-out later window)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_cost(grid, costs, thr, path):
    plt.figure(figsize=(6, 4))
    plt.plot(grid, costs); plt.axvline(thr, c="r", ls="--", label=f"chosen thr={thr:.3f}")
    plt.xlabel("Decision threshold"); plt.ylabel("Total cost (USD, validation)")
    plt.title("Cost vs threshold"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
