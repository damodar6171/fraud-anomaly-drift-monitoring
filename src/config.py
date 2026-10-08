"""Central configuration. Change values here, not in the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
OUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TARGET = "Class"

# Time-ordered split (fractions of the timeline, NOT random) -> mimics production
TRAIN_FRAC = 0.60   # "training window"
VAL_FRAC = 0.15     # used to pick the decision threshold (no test leakage)
# remaining 0.25 = held-out "later window" / test

# ---- Business cost model (justification lives in WRITEUP.md) ----
# Missed fraud (FN): we lose the transaction amount (+ fixed chargeback/investigation fee).
# Blocked legit (FP): fixed cost of friction: support call, lost interchange, churn risk.
FN_FIXED_COST = 15.0       # chargeback / investigation fee per missed fraud (USD)
FP_FIXED_COST = 5.0        # cost of declining/reviewing a legit txn (USD)
FP_AMOUNT_FRACTION = 0.0   # optionally, fraction of amount lost on a blocked legit txn

# ---- Drift thresholds ----
PSI_WARN = 0.10            # rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major
PSI_ALERT = 0.25
KS_PVALUE = 0.01
FEATURE_SHARE_ALERT = 0.20    # retrain if >=20% of features have PSI > PSI_ALERT
SCORE_PSI_ALERT = 0.25        # retrain if model score distribution PSI > this
PR_AUC_REL_DROP_ALERT = 0.20  # retrain if PR-AUC drops >20% relative to reference
N_MONITOR_WINDOWS = 5         # sequential windows in the monitoring report
