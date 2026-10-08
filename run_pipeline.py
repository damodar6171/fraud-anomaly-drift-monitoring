"""End-to-end pipeline: train -> compare -> choose cost-optimal threshold -> drift monitoring.

Usage:
    python run_pipeline.py                 # real Kaggle data in data/creditcard.csv
    python run_pipeline.py --synthetic     # quick smoke test without downloading data
"""
import argparse, json
import numpy as np
import pandas as pd
from src import config as C
from src import data as D, models as M, evaluate as E, drift as R


def main(synthetic=False):
    C.OUT_DIR.mkdir(exist_ok=True)
    df = D.add_features(D.load_data(synthetic))
    cols = D.feature_cols(df)
    train, val, test = D.time_split(df)
    print(f"rows: train={len(train)} val={len(val)} test={len(test)} | frauds: "
          f"{train.Class.sum()}/{val.Class.sum()}/{test.Class.sum()} | features={len(cols)}")
    Xtr, ytr, Xva, yva, Xte, yte = (train[cols], train.Class.values, val[cols], val.Class.values,
                                    test[cols], test.Class.values)
    res = {"split": dict(train=len(train), val=len(val), test=len(test))}

    # ---- 1. Imbalance-handling ablation (supervised) ----
    abl = {}
    for name in ["none", "balanced", "smote"]:
        try:
            m = M.fit_supervised(Xtr, ytr, name)
        except ImportError:
            print(f"  (skipping '{name}': imbalanced-learn not installed)"); continue
        abl[name] = dict(val=E.metrics(yva, M.sup_score(m, Xva)), test=E.metrics(yte, M.sup_score(m, Xte)))
    res["imbalance_ablation"] = abl
    print("\nImbalance ablation (PR-AUC val/test):")
    for k, v in abl.items():
        print(f"  {k:9s} {v['val']['pr_auc']:.3f} / {v['test']['pr_auc']:.3f}")

    # ---- 2. Final models ----
    sup = M.fit_supervised(Xtr, ytr, "balanced")
    lr = M.fit_logreg(Xtr, ytr)
    iso = M.fit_isolation_forest(Xtr)           # unlabeled
    ae = M.AutoencoderDetector().fit(Xtr.values)  # unlabeled
    scorers = {
        "GradBoost (class-weighted)": lambda X: M.sup_score(sup, X),
        "LogReg (class-weighted)": lambda X: lr.predict_proba(X)[:, 1],
        "IsolationForest (unsup.)": lambda X: M.iforest_score(iso, X),
        "Autoencoder (unsup.)": lambda X: ae.score(np.asarray(X)),
    }
    test_scores = {k: f(Xte) for k, f in scorers.items()}
    val_scores = {k: f(Xva) for k, f in scorers.items()}
    comp = {}
    for k in scorers:
        m_ = E.metrics(yte, test_scores[k])
        m_["recall@precision>=0.5"] = E.recall_at_precision(yte, test_scores[k], 0.5)
        m_["val_pr_auc"] = E.metrics(yva, val_scores[k])["pr_auc"]
        comp[k] = m_
    res["model_comparison_test"] = comp
    print("\nModel comparison on held-out later window:")
    print(pd.DataFrame(comp).T.round(4).to_string())
    pd.DataFrame(comp).T.to_csv(C.OUT_DIR / "model_comparison.csv")
    E.plot_pr(test_scores, yte, C.OUT_DIR / "pr_curves.png")

    # ---- 3. Cost-based threshold (chosen on VALIDATION, evaluated on TEST) ----
    main_name = "GradBoost (class-weighted)"
    thr, grid, costs = E.best_threshold(yva, val_scores[main_name], val.Amount.values)
    E.plot_cost(grid, costs, thr, C.OUT_DIR / "cost_curve.png")
    no_model = E.total_cost(yte, np.zeros(len(yte), bool), test.Amount.values)
    block_all = E.total_cost(yte, np.ones(len(yte), bool), test.Amount.values)
    s_te = test_scores[main_name]
    cost_tbl = {}
    for label, t in [("naive thr=0.5", 0.5), ("cost-optimal thr", thr)]:
        cost_tbl[label] = dict(threshold=t, cost=E.total_cost(yte, s_te >= t, test.Amount.values),
                               **E.confusion_at(yte, s_te, t))
    cost_tbl["approve everything"] = dict(cost=no_model)
    cost_tbl["block everything"] = dict(cost=block_all)
    res["cost_analysis_test"] = cost_tbl
    res["cost_params"] = dict(FN_fixed=C.FN_FIXED_COST, FP_fixed=C.FP_FIXED_COST, FP_amount_frac=C.FP_AMOUNT_FRACTION)
    print("\nCost analysis (test):")
    print(pd.DataFrame(cost_tbl).T.round(3).to_string())

    # ---- 4. Drift monitoring ----
    ref_pr = comp[main_name]["val_pr_auc"]
    score_fn = scorers[main_name]
    ref_scores = score_fn(Xva)                 # out-of-sample reference scores
    scenarios = {"natural later window": test}
    top_feats = ["V1", "V2", "V3", "V4"] if "V1" in cols else cols[:4]
    scenarios["injected drift (mild, 0.3)"] = D.inject_drift(test, top_feats, 0.3)
    scenarios["injected drift (severe, 1.0)"] = D.inject_drift(test, top_feats + ["V10", "V14"], 1.0)
    drift_out, feat_tables = {}, {}
    print("\nDrift checks vs training window:")
    for name, w in scenarios.items():
        d = R.evaluate_drift(Xtr, w[cols], ref_scores, score_fn(w[cols]), cols,
                             y_cur=w.Class.values, ref_pr_auc=ref_pr)
        drift_out[name] = dict(retrain=d.retrain, reasons=d.reasons, **d.summary)
        feat_tables[name] = R.feature_drift_report(Xtr, w[cols], cols)
        print(f"  [{'RETRAIN' if d.retrain else 'ok     '}] {name}: score_psi={d.summary['score_psi']:.3f} "
              f"feat_alert_share={d.summary['feature_share_alert']:.0%} "
              f"pr_auc={d.summary.get('pr_auc_cur', float('nan')):.3f}  {d.reasons}")
    res["drift_scenarios"] = drift_out
    feat_tables["natural later window"].to_csv(C.OUT_DIR / "feature_drift_natural.csv", index=False)
    feat_tables["injected drift (severe, 1.0)"].to_csv(C.OUT_DIR / "feature_drift_severe.csv", index=False)

    # rolling monitor over everything after training (val + test), natural data
    later = pd.concat([val, test])
    roll = R.rolling_monitor(later, cols, score_fn, train, ref_pr, score_ref_df=val)
    roll.to_csv(C.OUT_DIR / "rolling_monitor.csv", index=False)
    print("\nRolling monitor (natural data):")
    print(roll[["window", "n", "frauds", "score_psi", "feature_share_alert", "retrain"]].round(3).to_string(index=False))

    # severe-drift rolling: drift starts half-way through the later period
    half = len(later) // 2
    stream = pd.concat([later.iloc[:half], D.inject_drift(later.iloc[half:], top_feats + ["V10", "V14"], 1.0)])
    roll2 = R.rolling_monitor(stream, cols, score_fn, train, ref_pr, score_ref_df=val)
    roll2.to_csv(C.OUT_DIR / "rolling_monitor_injected.csv", index=False)
    print("\nRolling monitor (drift injected from window 3):")
    print(roll2[["window", "n", "frauds", "score_psi", "feature_share_alert", "retrain"]].round(3).to_string(index=False))
    _plot_monitor(roll, roll2, C.OUT_DIR / "drift_monitor.png")

    json.dump(res, open(C.OUT_DIR / "results.json", "w"), indent=2, default=float)
    print(f"\nSaved results to {C.OUT_DIR}/")


def _plot_monitor(roll, roll2, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, r, t in [(ax[0], roll, "Natural data"), (ax[1], roll2, "Drift injected mid-stream")]:
        a.plot(r.window, r.score_psi, "o-", label="score PSI")
        a.plot(r.window, r.feature_share_alert, "s-", label="share of features PSI>0.25")
        a.axhline(C.SCORE_PSI_ALERT, c="r", ls="--", lw=1, label="alert bound")
        a.set_title(t); a.set_xlabel("window (chronological)")
    ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke test)")
    main(ap.parse_args().synthetic)
