"""Supervised + unsupervised detectors (higher score = more suspicious)."""
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import config as C


def fit_supervised(X, y, weighting="balanced"):
    """Gradient boosting. weighting: 'balanced' (class weights, default) | 'none' | 'smote'."""
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
        class_weight="balanced" if weighting == "balanced" else None,
        random_state=C.RANDOM_STATE)
    if weighting == "smote":
        from imblearn.over_sampling import SMOTE  # optional dependency
        X, y = SMOTE(sampling_strategy=0.1, random_state=C.RANDOM_STATE).fit_resample(X, y)
    return clf.fit(X, y)


def sup_score(clf, X):
    return clf.predict_proba(X)[:, 1]


def fit_logreg(X, y):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(class_weight="balanced", max_iter=2000,
                                         random_state=C.RANDOM_STATE))
    return m.fit(X, y)


def fit_isolation_forest(X):
    """Unsupervised: fitted WITHOUT labels. We threshold the raw anomaly score ourselves."""
    return IsolationForest(n_estimators=300, max_samples=0.5, contamination="auto",
                           random_state=C.RANDOM_STATE, n_jobs=-1).fit(X)


def iforest_score(model, X):
    return -model.score_samples(X)


class AutoencoderDetector:
    """Small MLP autoencoder trained on (assumed-)legitimate transactions; score = reconstruction MSE.
    scikit-learn implementation -> no heavy deep-learning dependency."""

    def __init__(self, hidden=(16, 8, 16)):
        self.scaler = StandardScaler()
        self.net = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", max_iter=60,
                                early_stopping=True, random_state=C.RANDOM_STATE)

    def fit(self, X, y=None):
        Xs = self.scaler.fit_transform(X if y is None else X[y == 0])
        self.net.fit(Xs, Xs)
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        return ((Xs - self.net.predict(Xs)) ** 2).mean(axis=1)
