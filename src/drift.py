"""Drift detection (data drift + score/confidence drift + performance drift) and alert logic."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score
from . import config as C


def psi(reference, current, bins=10, eps=1e-6):
    """Population Stability Index with quantile bins fitted on the REFERENCE window."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference) + eps
    c = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift_report(ref: pd.DataFrame, cur: pd.DataFrame, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        p = psi(ref[c], cur[c])
        ks = ks_2samp(ref[c], cur[c])
        rows.append(dict(feature=c, psi=p, ks_stat=ks.statistic, ks_pvalue=ks.pvalue,
                         status="ALERT" if p > C.PSI_ALERT else "WARN" if p > C.PSI_WARN else "ok"))
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


@dataclass
class DriftDecision:
    retrain: bool
    reasons: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def evaluate_drift(ref_X, cur_X, ref_scores, cur_scores, cols, y_cur=None, ref_pr_auc=None) -> DriftDecision:
    """Alert logic. Retrain is flagged if ANY of:
      1. share of features with PSI > PSI_ALERT  >= FEATURE_SHARE_ALERT          (data drift)
      2. PSI of the model's score distribution   >  SCORE_PSI_ALERT              (confidence drift; label-free)
      3. PR-AUC fell by > PR_AUC_REL_DROP_ALERT relative to reference            (performance drift; needs labels,
                                                                                  which arrive late in fraud)
    """
    rep = feature_drift_report(ref_X, cur_X, cols)
    share = float((rep.psi > C.PSI_ALERT).mean())
    score_psi = psi(ref_scores, cur_scores)
    summary = dict(feature_share_alert=share, n_features_alert=int((rep.psi > C.PSI_ALERT).sum()),
                   n_features_warn=int(((rep.psi > C.PSI_WARN) & (rep.psi <= C.PSI_ALERT)).sum()),
                   score_psi=score_psi, mean_score_ref=float(np.mean(ref_scores)),
                   mean_score_cur=float(np.mean(cur_scores)), top_drifted=rep.head(5).feature.tolist())
    reasons = []
    if share >= C.FEATURE_SHARE_ALERT:
        reasons.append(f"{share:.0%} of features have PSI>{C.PSI_ALERT} (limit {C.FEATURE_SHARE_ALERT:.0%})")
    if score_psi > C.SCORE_PSI_ALERT:
        reasons.append(f"score-distribution PSI={score_psi:.3f} > {C.SCORE_PSI_ALERT}")
    if y_cur is not None and ref_pr_auc is not None and y_cur.sum() >= 5:
        pr = float(average_precision_score(y_cur, cur_scores))
        summary["pr_auc_cur"] = pr
        drop = (ref_pr_auc - pr) / max(ref_pr_auc, 1e-9)
        summary["pr_auc_rel_drop"] = float(drop)
        if drop > C.PR_AUC_REL_DROP_ALERT:
            reasons.append(f"PR-AUC dropped {drop:.0%} (limit {C.PR_AUC_REL_DROP_ALERT:.0%})")
    return DriftDecision(retrain=bool(reasons), reasons=reasons, summary=summary)


def rolling_monitor(df, cols, score_fn, ref_df, ref_pr_auc, score_ref_df=None,
                    n_windows=C.N_MONITOR_WINDOWS, target=C.TARGET):
    """Slide through chronological windows and emit one monitoring row per window.
    ref_df = feature reference (training window); score_ref_df = out-of-sample window used as the
    score-distribution reference (in-sample scores of a boosted model are overconfident)."""
    score_ref_df = ref_df if score_ref_df is None else score_ref_df
    ref_scores = score_fn(score_ref_df[cols])
    out = []
    df = df.sort_values("Time").reset_index(drop=True)
    for i, idx in enumerate(np.array_split(np.arange(len(df)), n_windows)):
        w = df.iloc[idx]
        d = evaluate_drift(ref_df[cols], w[cols], ref_scores, score_fn(w[cols]), cols,
                           y_cur=w[target].values, ref_pr_auc=ref_pr_auc)
        out.append(dict(window=i + 1, n=len(w), frauds=int(w[target].sum()), **d.summary,
                        retrain=d.retrain, reasons="; ".join(d.reasons)))
    return pd.DataFrame(out)
