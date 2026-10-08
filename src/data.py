"""Data loading, feature prep, time-ordered splitting, synthetic fallback."""
import numpy as np
import pandas as pd
from . import config as C


def make_synthetic(n=60000, fraud_rate=0.004, seed=C.RANDOM_STATE) -> pd.DataFrame:
    """Stand-in with the same schema as Kaggle creditcard.csv (Time, V1..V28, Amount, Class).
    Includes mild natural drift over time so the pipeline can be tested without the real data."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 172800, n))
    X = rng.normal(size=(n, 28))
    X = X + (t / t.max())[:, None] * np.r_[np.full(4, 0.6), np.zeros(24)]  # V1-V4 slowly drift
    y = (rng.random(n) < fraud_rate).astype(int)
    idx = np.where(y == 1)[0]
    shift = np.zeros(28)
    shift[[2, 3, 9, 13, 16]] = [-3, 3, -3, -3, -2.5]
    X[idx] += shift + rng.normal(scale=0.8, size=(len(idx), 28))
    amount = np.round(np.exp(rng.normal(3.3, 1.4, n)), 2)
    amount[idx] = np.round(np.exp(rng.normal(3.8, 1.8, len(idx))), 2)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", t)
    df["Amount"] = amount
    df[C.TARGET] = y
    return df


def load_data(synthetic: bool = False) -> pd.DataFrame:
    if synthetic:
        return make_synthetic()
    if not C.DATA_PATH.exists():
        raise FileNotFoundError(
            f"{C.DATA_PATH} not found. Download 'creditcard.csv' from Kaggle "
            "(mlg-ulb/creditcardfraud) into data/, or run with --synthetic.")
    return pd.read_csv(C.DATA_PATH)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    return out


def feature_cols(df: pd.DataFrame):
    """Time is excluded from model inputs: it is only an ordering key. Using it would make the
    model memorise the calendar, which is exactly what breaks under drift."""
    return [c for c in df.columns if c not in (C.TARGET, "Time", "Amount")]


def time_split(df: pd.DataFrame):
    """Chronological train / validation / later(test) split."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    i1, i2 = int(n * C.TRAIN_FRAC), int(n * (C.TRAIN_FRAC + C.VAL_FRAC))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def inject_drift(df: pd.DataFrame, cols, strength: float = 1.0, seed=C.RANDOM_STATE) -> pd.DataFrame:
    """Simulate a population change / new fraud tactic: shift & rescale selected features and
    inflate amounts (e.g. holiday spending spike)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in cols:
        out[c] = out[c] * (1 + 0.5 * strength) + strength * 1.5 + rng.normal(0, 0.1 * strength, len(out))
    out["Amount"] = out["Amount"] * (1 + strength)
    out["log_amount"] = np.log1p(out["Amount"])
    return out
