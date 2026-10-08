import numpy as np, pandas as pd
from src import drift as R, evaluate as E, data as D


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert R.psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert R.psi(rng.normal(size=5000), rng.normal(2, 1, 5000)) > 0.25


def test_cost_extremes():
    y = np.array([0, 0, 1]); amt = np.array([10., 10., 100.])
    assert E.total_cost(y, np.array([False] * 3), amt) > E.total_cost(y, np.array([False, False, True]), amt)


def test_time_split_is_chronological():
    tr, va, te = D.time_split(D.make_synthetic(n=5000))
    assert tr.Time.max() <= va.Time.min() and va.Time.max() <= te.Time.min()


def test_drift_decision_flags_injected_shift():
    df = D.add_features(D.make_synthetic(n=20000)); cols = D.feature_cols(df)
    ref, cur = df.iloc[:10000], D.inject_drift(df.iloc[10000:], cols[:10], 1.0)
    d = R.evaluate_drift(ref[cols], cur[cols], np.zeros(100), np.zeros(100), cols)
    assert d.retrain
