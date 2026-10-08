# Fraud / Anomaly Detection with Drift Monitoring

A fraud-detection system that (1) flags suspicious card transactions with a supervised model **and** an
unsupervised anomaly detector, (2) chooses its decision threshold from an explicit **business cost** model, and
(3) **monitors itself for drift** and raises a retrain flag when data, scores, or performance move beyond set bounds.

> Dataset: [Credit Card Fraud Detection (Kaggle, mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) –
> 284,807 transactions, 492 frauds (0.172%). Columns: `Time`, `V1..V28` (PCA), `Amount`, `Class`.

## Repository layout
```
run_pipeline.py        # one command reproduces everything
src/config.py          # splits, cost parameters, drift thresholds (single place to tune)
src/data.py            # loading, time-ordered split, synthetic data, drift injection
src/models.py          # GradBoost (class-weighted), LogReg, IsolationForest, Autoencoder
src/evaluate.py        # PR-AUC, ROC-AUC, cost model, threshold selection, plots
src/drift.py           # PSI, KS test, score drift, performance drift, alert logic, rolling monitor
tests/test_core.py     # unit tests
WRITEUP.md             # approach, decisions, results
outputs/               # generated: metrics, plots, CSV drift reports
```

## Setup
```bash
git clone <your-repo-url> && cd fraud-drift-monitoring
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Download `creditcard.csv` from Kaggle and put it in `data/`:
```bash
pip install kaggle   # needs ~/.kaggle/kaggle.json
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

## Reproduce results
```bash
python run_pipeline.py              # full run on the real dataset  (~2-4 min on a laptop)
python run_pipeline.py --synthetic  # 20-second smoke test, no download needed
pytest -q                           # unit tests
```
Outputs land in `outputs/`: `results.json`, `model_comparison.csv`, `pr_curves.png`, `cost_curve.png`,
`drift_monitor.png`, `feature_drift_*.csv`, `rolling_monitor*.csv`. All seeds are fixed (`RANDOM_STATE=42`).

## Re-running evaluation / changing assumptions
* **Different cost assumptions** → edit `FN_FIXED_COST`, `FP_FIXED_COST`, `FP_AMOUNT_FRACTION` in `src/config.py`, re-run.
* **Different drift sensitivity** → edit `PSI_WARN`, `PSI_ALERT`, `FEATURE_SHARE_ALERT`, `SCORE_PSI_ALERT`, `PR_AUC_REL_DROP_ALERT`.
* **Different windows** → edit `TRAIN_FRAC`, `VAL_FRAC`, `N_MONITOR_WINDOWS`.
* **Monitor new data in production** → call `src.drift.evaluate_drift(ref_X, new_X, ref_scores, new_scores, cols, y_cur=..., ref_pr_auc=...)`; it returns `retrain` (bool) and human-readable `reasons`.

## Method in brief
| Step | Choice |
|---|---|
| Split | **Chronological** train 60% / validation 15% / later window 25% (never random – random splits hide drift) |
| Imbalance | Class weighting (primary), compared against no handling and SMOTE; plus unsupervised anomaly framing |
| Supervised | HistGradientBoosting (class-weighted) + LogReg baseline |
| Unsupervised | Isolation Forest, MLP autoencoder (reconstruction error) – both trained **without labels** |
| Metric | **PR-AUC** (primary), ROC-AUC (secondary), recall at precision ≥ 0.5 |
| Threshold | Minimise expected cost on the validation window, report on the later window |
| Drift | PSI + KS per feature, PSI on model scores, relative PR-AUC drop (when labels arrive) |
| Alert | Retrain if ≥20% features PSI>0.25 **or** score PSI>0.25 **or** PR-AUC drops >20% |

See `WRITEUP.md` for justifications and results.
