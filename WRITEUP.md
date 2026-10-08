# Write-up: Fraud Detection with Drift Monitoring

## 1. Problem and approach
Fraud patterns move (new attack tactics, seasonality), so a model trained once silently degrades. The system therefore
has three parts: **detect** (supervised + unsupervised), **decide** (cost-based threshold), **watch itself** (drift monitor
with retrain alert). Data is split **chronologically** (60/15/25) so the later window plays the role of "future
production traffic"; a random split would leak the future into training and hide drift.

## 2. Handling extreme imbalance (0.17% fraud)
* **Class weighting (chosen).** Re-weights the loss, keeps the data distribution intact, adds no synthetic points, and is
  cheap. The model sees every real legitimate transaction.
* **SMOTE (tested as ablation, not chosen).** Interpolates between fraud points in a 28-dim PCA space where fraud is
  heterogeneous; the synthetic samples can be unrealistic, and it must be applied only inside training (never validation/test).
  It also distorts predicted probabilities. Reported in the ablation table for transparency.
* **Anomaly framing (also built).** Isolation Forest / autoencoder need no labels, which matters because fraud labels are
  delayed (chargebacks take weeks) and novel fraud has no labels at all.
* Caveat: class weights make probabilities **uncalibrated**, so 0.5 is not a meaningful cut-off → we pick the threshold by cost, not by default.

## 3. Metric choice: PR-AUC, not ROC-AUC
With 0.17% positives, ROC-AUC uses the false-positive *rate*, whose denominator is ~284k negatives; 500 false alarms
barely move FPR, so ROC-AUC stays ≈0.97+ even for a model that would bury analysts in false alerts. Precision-Recall
uses precision = TP/(TP+FP), which directly punishes false alerts, and its no-skill baseline is the prevalence
(≈0.0017), so improvements are visible. We report PR-AUC (average precision) as primary, with ROC-AUC secondary and
recall at precision ≥ 0.5 as an operational metric.

## 4. Supervised vs unsupervised
| Model | PR-AUC (later window) | ROC-AUC | Recall @ P≥0.5 |
|---|---|---|---|
| GradBoost, class-weighted | **0.7585** | 0.9656 | 77.66% |
| LogReg, class-weighted | 0.7523 | 0.9774 | 79.79% |
| Isolation Forest (no labels) | 0.0751 | 0.9464 | 0.00% |
| Autoencoder (no labels) | 0.0340 | 0.9461 | 0.00% |

Expected pattern (typical on this dataset): supervised ≫ unsupervised on PR-AUC because labels carry strong signal;
unsupervised detectors still beat the no-skill line by a large margin and are valuable for **new** fraud types and as a
label-free backup. Discuss your actual numbers here: supervised models achieve PR-AUC > 0.75 with ~78–80% recall at P≥0.5, whereas unsupervised models reach ~0.03–0.08 PR-AUC and 0% recall at P≥0.5.

## 5. Cost of false positives vs false negatives, and threshold
* **False negative (missed fraud):** lose the transaction amount + fixed chargeback/investigation fee (`FN_FIXED_COST = $15`).
* **False positive (blocked legit):** support contact, abandoned purchase, customer-churn risk (`FP_FIXED_COST = $5`; optionally a share of the amount).
* These are **assumptions** and live in `src/config.py`; the analysis is rerun by changing them.
* Because the average fraud costs far more than one false alarm, the cost-optimal threshold is **lower than 0.5**: we accept
  many more false alarms per caught fraud. Break-even logic: flag a transaction when P(fraud) × (amount + $15) > (1 − P) × $5.
* The threshold is **chosen on the validation window** (median of the near-optimal cost plateau, because the cost curve is
  jagged with few frauds) and **evaluated on the later window**. Table (test): naive 0.5 vs cost-optimal vs approve-all vs
  block-all → **Naive 0.5: $3,614.09 | Cost-optimal (0.3627): $3,795.23 | Approve-all: $12,249.13 | Block-all: $355,540.00**. Approve-all is the "no model" baseline the system must beat.
* Honest note: if the validation window has few frauds the threshold is noisy; in production re-tune it on each retrain and
  consider a review tier (auto-block above a high score, step-up authentication in a mid band).

## 6. Drift detection and alerting
Three label-free-first signals, compared with the training window:
1. **Data drift:** PSI (+KS test) per feature; PSI <0.1 stable, 0.1-0.25 warn, >0.25 alert.
2. **Confidence/score drift:** PSI of the model's score distribution against an out-of-sample reference (validation),
   because in-sample boosted scores are over-confident.
3. **Performance drift:** relative PR-AUC drop vs reference. Needs labels, which arrive late in fraud, hence signals 1-2 exist.

**Alert rule → flag retraining if any:** ≥20% of features exceed PSI 0.25, or score PSI > 0.25, or PR-AUC falls >20% relative.
Thresholds are in `config.py` and are judgement calls to tune against retraining cost.

**Demonstration:** (a) natural later window → **Retrain alert fires (24.1% of features exceed PSI 0.25: V1, V3, V28, V11, V25; score PSI 0.0043; PR-AUC drop -2.92%)**; (b) injected mild drift → **Retrain alert fires (27.6% of features exceed PSI 0.25; score PSI 0.0210; PR-AUC drop -2.50%)**; (c) injected severe drift
(shifted/rescaled V1-V4, V10, V14 and ×2 amounts) → alert fires (37.9% of features exceed PSI 0.25; score PSI 0.1807; PR-AUC drop 1.21%); (d) rolling monitor with drift injected mid-stream shows the
alert triggering at the window where drift starts (`outputs/drift_monitor.png`, `rolling_monitor_injected.csv`).

On the smoke test with synthetic data, the feature-level signal caught the severe shift while the score PSI stayed low –
a good reason to monitor several signals instead of one: the model can be robust to some shifts, or blind to them.

## 7. Limitations and next steps
* `V1..V28` are anonymised PCA components; real feature-level root-causing (merchant, geography) isn't possible.
* Dataset spans only 2 days, so "drift" is mostly simulated; IEEE-CIS gives a longer horizon.
* Cost parameters are assumed; labels are treated as immediately available.
* Next: scheduled retraining job, champion/challenger comparison, delayed-label performance tracking, calibration (isotonic) of scores.
