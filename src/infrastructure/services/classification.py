"""
Infrastructure: Klassifikations-Algorithmen (Logistic Regression, KNN).
Implementierung "from scratch" mittels NumPy zur Veranschaulichung der mathematischen Konzepte.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from ...core.domain import (
    IClassificationService,
    ClassificationResult,
    ClassificationMetrics,
    Result
)
from ...config import get_logger

logger = get_logger(__name__)


class ClassificationServiceImpl(IClassificationService):
    """
    Implementierung von Klassifikations-Algorithmen.
    
    Prinzipien:
    1. Reine NumPy-Implementierung (kein Scikit-Learn)
    2. Vektorisierte Operationen für Effizienz
    3. Transparente Metriken-Berechnung
    """
    
    # =========================================================================
    # LOGISTISCHE REGRESSION (Binär)
    # =========================================================================
    
    def train_logistic(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        learning_rate: float = 0.01,
        iterations: int = 1000
    ) -> ClassificationResult:
        """
        Trainiert eine binäre logistische Regression mittels Gradientenabstieg (Gradient Descent).
        
        Modell: p = sigmoid(X * w + b)
        Loss-Funktion: Binary Cross Entropy
        """
        n_samples, n_features = X.shape
        
        # Standardisierung der Daten zur Stabilisierung des Trainings
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - X_mean) / X_std
        
        # Initialisierung der Gewichte (Weights) und des Bias
        weights = np.zeros(n_features)
        bias = 0.0
        
        loss_history = []
        
        # Iterativer Optimierungsprozess
        for _ in range(iterations):
            # 1. Lineares Modell berechnen: z = Xw + b
            z = np.dot(X_scaled, weights) + bias
            
            # 2. Sigmoid-Aktivierung: Transformation in Wahrscheinlichkeiten [0, 1]
            predictions = 1 / (1 + np.exp(-z))
            
            # 3. Berechnung der Gradienten (dL/dw und dL/db)
            error = predictions - y
            dw = (1 / n_samples) * np.dot(X_scaled.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            # 4. Update der Parameter in Richtung des negativen Gradienten
            weights -= learning_rate * dw
            bias -= learning_rate * db
            
            # Protokollierung der Loss-Entwicklung (BCE-Loss)
            if _ % 100 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-15) + (1-y) * np.log(1-predictions + 1e-15))
                loss_history.append(loss)
        
        # Finale Vorhersagen auf den Trainingsdaten
        z_final = np.dot(X_scaled, weights) + bias
        probs = 1 / (1 + np.exp(-z_final))
        preds = (probs >= 0.5).astype(int)
        
        # Rückrechnung der Gewichte auf die Originalskala (für Interpretierbarkeit)
        real_weights = weights / X_std
        real_bias = bias - np.sum(weights * X_mean / X_std)
        
        metrics = self.calculate_metrics(y, preds, probs)
        
        return ClassificationResult(
            classes=[0, 1],
            predictions=preds,
            probabilities=probs,
            metrics=metrics,
            model_params={
                "coefficients": real_weights.tolist(),
                "intercept": float(real_bias),
                "iterations": iterations,
                "learning_rate": learning_rate,
                "loss_history": loss_history
            },
        )
    
    # =========================================================================
    # K-NEAREST NEIGHBORS (KNN)
    # =========================================================================
    
    def train_knn(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        k: int = 3
    ) -> ClassificationResult:
        """
        "Training" eines KNN-Modells.
        Da KNN ein 'Lazy Learner' ist, werden hier primär die Daten gespeichert.
        Zusätzlich berechnen wir Vorhersagen für die Trainingsdaten.
        """
        n_samples = len(y)
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Standardisierung für die Distanzberechnung (Euklidischer Abstand)
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - X_mean) / X_std
        
        preds = np.zeros(n_samples, dtype=int)
        probs = np.zeros((n_samples, n_classes))
        
        # Klassifizierung jedes Punktes durch Vergleich mit allen anderen Punkten
        for i in range(n_samples):
            # 1. Berechnung der euklidischen Distanzen zu allen Punkten
            diff = X_scaled - X_scaled[i]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            
            # 2. Finden der k nächsten Nachbarn
            nearest_indices = np.argsort(dists)[:k]
            nearest_labels = y[nearest_indices]
            
            # 3. Mehrheitsentscheidung (Voting)
            counts = np.bincount(nearest_labels, minlength=n_classes)
            preds[i] = np.argmax(counts)
            probs[i] = counts / k
            
        metrics = self.calculate_metrics(y, preds, probs[:, 1] if n_classes==2 else probs)
        
        return ClassificationResult(
            classes=classes.tolist(),
            predictions=preds,
            probabilities=probs,
            metrics=metrics,
            model_params={
                "k": k,
                "X_train": X,
                "y_train": y,
                "X_mean": X_mean,
                "X_std": X_std
            },
        )

    # =========================================================================
    # METRIKEN BERECHNUNG
    # =========================================================================

    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray
    ) -> ClassificationMetrics:
        """Berechnet umfassende Klassifikationsmetriken (Accuracy, Precision, Recall, F1)."""
        n = len(y_true)
        if n == 0:
            return ClassificationMetrics(0, 0, 0, 0, np.zeros((2,2)))
            
        # Unterscheidung zwischen binärer und Multiklassen-Klassifikation
        classes = np.unique(y_true)
        is_binary = len(classes) <= 2
        
        if is_binary:
            # Binär: 0=Negativ, 1=Positiv
            tp = np.sum((y_true == 1) & (y_pred == 1)) # True Positive
            tn = np.sum((y_true == 0) & (y_pred == 0)) # True Negative
            fp = np.sum((y_true == 0) & (y_pred == 1)) # False Positive
            fn = np.sum((y_true == 1) & (y_pred == 0)) # False Negative
            
            cm = np.array([[tn, fp], [fn, tp]]) # Confusion Matrix
            
            accuracy = (tp + tn) / n
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Manual AUC-Approximation (Trapezoidal Rule)
            if len(np.unique(y_true)) > 1:
                order = np.argsort(y_prob)[::-1]
                y_true_sorted = y_true[order]
                tpr = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
                fpr = np.cumsum(1 - y_true_sorted) / np.sum(1 - y_true_sorted)
                
                # Manual integration (trapz)
                auc = 0.0
                for i in range(1, len(fpr)):
                    auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0
            else:
                auc = 0.5
                
        else:
            # Multiklasse: Macro-Averaging
            n_classes = len(classes)
            precisions, recalls = [], []
            cm = np.zeros((n_classes, n_classes), dtype=int)
            
            # Aufbau der Konfusionsmatrix
            for i, c_true in enumerate(classes):
                for j, c_pred in enumerate(classes):
                    cm[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))
            
            accuracy = np.trace(cm) / n
            
            # Berechnung für jede Klasse einzeln
            for i in range(n_classes):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precisions.append(p)
                recalls.append(r)
            
            # Durchschnitt über alle Klassen (Macro)
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            auc = None 
            
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            auc=auc
        )

    # =========================================================================
    # VORHERSAGE & EVALUIERUNG
    # =========================================================================

    def predict_logistic(
        self,
        X: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vorhersage mittels logistischer Regressionsparameter."""
        weights = np.array(params["coefficients"])
        bias = params["intercept"]
        
        if X.shape[1] != len(weights):
             raise ValueError(f"Feature mismatch: X hat {X.shape[1]}, Modell erwartet {len(weights)}")
             
        # Anwendung der gelernten (unskalierten) Parameter auf die Rohdaten
        z = np.dot(X, weights) + bias
        probs = 1 / (1 + np.exp(-z))
        preds = (probs >= 0.5).astype(int)
        
        return preds, probs

    def predict_knn(
        self,
        X: np.ndarray,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vorhersage mittels KNN-Modell (Vergleich mit Trainingsdaten)."""
        k = params["k"]
        X_train = np.array(params["X_train"])
        y_train = np.array(params["y_train"])
        X_train_mean = np.array(params["X_mean"])
        X_train_std = np.array(params["X_std"])
        
        # Skalierung der Eingabedaten basierend auf dem Trainings-Set-Zustand
        X_scaled = (X - X_train_mean) / X_train_std
        X_train_scaled = (X_train - X_train_mean) / X_train_std
        
        n_samples = len(X)
        n_classes = len(np.unique(y_train))
        
        preds = np.zeros(n_samples, dtype=int)
        probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            diff = X_train_scaled - X_scaled[i]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            
            nearest_indices = np.argsort(dists)[:k]
            nearest_labels = y_train[nearest_indices]
            
            counts = np.bincount(nearest_labels, minlength=n_classes)
            preds[i] = np.argmax(counts)
            probs[i] = counts / k
            
        return preds, probs[:, 1] if n_classes==2 else probs

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        method: str
    ) -> ClassificationMetrics:
        """Evaluiert das Modell auf unbekannten (Test-)Daten."""
        if method == "knn":
            preds, probs = self.predict_knn(X, params)
        else:
            preds, probs = self.predict_logistic(X, params)
            
        return self.calculate_metrics(y, preds, probs)
