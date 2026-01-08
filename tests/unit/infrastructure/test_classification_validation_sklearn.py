"""
Validierungstests: Vergleich unserer Klassifikations-Implementierungen mit scikit-learn.

Analog zu `test_regression_validation_sklearn.py` werden hier alle
ML-Berechnungen (Logistic Regression, KNN, Metriken) gegen scikit-learn geprüft.
"""

import pytest
import numpy as np

# Optionaler Import für scikit-learn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - nur falls sklearn fehlt
    SKLEARN_AVAILABLE = False
    pytestmark = pytest.mark.skip(
        reason="scikit-learn nicht installiert. Installiere mit: pip install scikit-learn"
    )

from src.infrastructure.services.classification import ClassificationServiceImpl


class TestLogisticRegressionValidation:
    """Validierungstests für unsere logistische Regression gegen scikit-learn."""

    @pytest.fixture
    def service(self) -> ClassificationServiceImpl:
        return ClassificationServiceImpl()

    @pytest.fixture
    def sklearn_model(self):
        return LogisticRegression(solver="lbfgs")

    def _generate_binary_clusters(self, n: int = 200, seed: int = 42):
        """Erzeuge zwei klar getrennte Cluster für binäre Klassifikation."""
        rng = np.random.default_rng(seed)
        n_half = n // 2

        # Cluster 0 um (0, 0)
        X0 = rng.normal(loc=[0.0, 0.0], scale=0.8, size=(n_half, 2))
        y0 = np.zeros(n_half, dtype=int)

        # Cluster 1 um (3, 3)
        X1 = rng.normal(loc=[3.0, 3.0], scale=0.8, size=(n - n_half, 2))
        y1 = np.ones(n - n_half, dtype=int)

        X = np.vstack([X0, X1])
        y = np.concatenate([y0, y1])
        return X, y

    def test_logistic_regression_synthetic_binary(self, service, sklearn_model):
        """Vergleich Accuracy / Precision / Recall / F1 / AUC auf synthetischen Binärdaten."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn nicht verfügbar")

        X, y = self._generate_binary_clusters(n=300, seed=42)

        # Unsere Implementierung
        our_result = service.train_logistic(X, y, learning_rate=0.1, iterations=800)

        # scikit-learn Referenzmodell
        sklearn_model.fit(X, y)
        sk_probs = sklearn_model.predict_proba(X)[:, 1]
        sk_preds = sklearn_model.predict(X)

        # Vergleich der Metriken
        sk_acc = accuracy_score(y, sk_preds)
        sk_prec = precision_score(y, sk_preds)
        sk_rec = recall_score(y, sk_preds)
        sk_f1 = f1_score(y, sk_preds)
        sk_auc = roc_auc_score(y, sk_probs)

        # Toleranzen etwas großzügiger, da Optimierung unterschiedlich ist
        assert abs(our_result.metrics.accuracy - sk_acc) < 0.05
        assert abs(our_result.metrics.precision - sk_prec) < 0.05
        assert abs(our_result.metrics.recall - sk_rec) < 0.05
        assert abs(our_result.metrics.f1_score - sk_f1) < 0.05
        assert abs(our_result.metrics.auc - sk_auc) < 0.05


class TestKNNValidation:
    """Validierungstests für unseren KNN-Classifier gegen scikit-learn."""

    @pytest.fixture
    def service(self) -> ClassificationServiceImpl:
        return ClassificationServiceImpl()

    @pytest.fixture
    def sklearn_model(self):
        return KNeighborsClassifier(n_neighbors=3)

    def _generate_binary_clusters(self, n: int = 200, seed: int = 123):
        rng = np.random.default_rng(seed)
        n_half = n // 2

        X0 = rng.normal(loc=[-1.0, -1.0], scale=0.6, size=(n_half, 2))
        y0 = np.zeros(n_half, dtype=int)

        X1 = rng.normal(loc=[2.0, 2.0], scale=0.6, size=(n - n_half, 2))
        y1 = np.ones(n - n_half, dtype=int)

        X = np.vstack([X0, X1])
        y = np.concatenate([y0, y1])
        return X, y

    def test_knn_synthetic_binary(self, service, sklearn_model):
        """Vergleich Accuracy / F1 auf synthetischen Binärdaten."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn nicht verfügbar")

        X, y = self._generate_binary_clusters(n=300, seed=123)

        # Unsere Implementierung
        our_result = service.train_knn(X, y, k=3)

        # scikit-learn Referenzmodell
        sklearn_model.fit(X, y)
        sk_preds = sklearn_model.predict(X)
        sk_probs = sklearn_model.predict_proba(X)[:, 1]

        sk_acc = accuracy_score(y, sk_preds)
        sk_f1 = f1_score(y, sk_preds)
        sk_auc = roc_auc_score(y, sk_probs)

        assert abs(our_result.metrics.accuracy - sk_acc) < 0.05
        assert abs(our_result.metrics.f1_score - sk_f1) < 0.05
        # unsere AUC ist optional bei KNN; nur vergleichen, wenn vorhanden
        if our_result.metrics.auc is not None:
            assert abs(our_result.metrics.auc - sk_auc) < 0.05


class TestClassificationMetricsValidation:
    """Validierung der Metrik-Berechnung direkt gegen scikit-learn."""

    @pytest.fixture
    def service(self) -> ClassificationServiceImpl:
        return ClassificationServiceImpl()

    def test_binary_metrics_against_sklearn(self, service):
        """Vergleicht Accuracy / Precision / Recall / F1 / AUC im Binärfall."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn nicht verfügbar")

        rng = np.random.default_rng(42)
        n = 200
        # Zufällige aber reproduzierbare Binärlabels
        y_true = rng.integers(0, 2, size=n)
        # Künstliche Wahrscheinlichkeiten und Vorhersagen
        y_prob = rng.random(size=n)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = service.calculate_metrics(y_true, y_pred, y_prob)

        sk_acc = accuracy_score(y_true, y_pred)
        sk_prec = precision_score(y_true, y_pred, zero_division=0)
        sk_rec = recall_score(y_true, y_pred, zero_division=0)
        sk_f1 = f1_score(y_true, y_pred, zero_division=0)
        # AUC nur definiert, wenn beide Klassen vorkommen
        if len(np.unique(y_true)) > 1:
            sk_auc = roc_auc_score(y_true, y_prob)
        else:
            sk_auc = 0.5

        assert abs(metrics.accuracy - sk_acc) < 1e-8
        assert abs(metrics.precision - sk_prec) < 1e-8
        assert abs(metrics.recall - sk_rec) < 1e-8
        assert abs(metrics.f1_score - sk_f1) < 1e-8
        assert abs(metrics.auc - sk_auc) < 1e-6

