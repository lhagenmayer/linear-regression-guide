"""
Schritt 2: BERECHNEN (ML-Brücke)

Dieses Modul implementiert Berechnungen für das Kapitel "Statistik -> ML Brücke".
Es bietet Datengenerierung für Visualisierungen wie Loss Landscapes (Fehlerlandschaften),
Gradient-Descent-Pfade und Overfitting-Demonstrationen.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from ...config import get_logger

logger = get_logger(__name__)


@dataclass
class LossSurfaceResult:
    """Daten für die 3D-Visualisierung der Fehlerlandschaft (Loss Surface)."""
    w_grid: np.ndarray  # Gitter der Gewichte (Weights, X-Achse)
    b_grid: np.ndarray  # Gitter der Biases (Y-Achse)
    loss_grid: np.ndarray  # Gitter der Fehlerwerte (Losses, Z-Achse)
    path_w: List[float]  # Pfad des Gradientenabstiegs (Gewichte)
    path_b: List[float]  # Pfad des Gradientenabstiegs (Biases)
    path_loss: List[float]  # Pfad der Fehlerwerte während der Optimierung
    optimal_w: float
    optimal_b: float


@dataclass
class OptimizationResult:
    """Ergebnis eines Optimierungsprozesses (z.B. Gradient Descent)."""
    iterations: List[int]
    losses: List[float]
    final_w: float
    final_b: float
    converged: bool


@dataclass
class OverfittingDemoResult:
    """Daten für die Demonstration von Overfitting mittels Polynomen."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_plot: np.ndarray  # Dichtes Gitter für das Zeichnen glatter Kurven
    train_metrics: Dict[int, float]  # Grad des Polynoms -> MSE
    test_metrics: Dict[int, float]   # Grad des Polynoms -> MSE
    predictions: Dict[int, np.ndarray] # Grad -> y_plot Vorhersage


class MLBridgeService:
    """
    Service zur Berechnung von Konzepten an der Schnittstelle Statistik/ML.
    
    Prinzipien:
    1. Gitter-Generierung für 3D-Oberflächen
    2. Simulation iterativer Optimierungen
    3. Hilfestellungen für Bias-Variance-Dekomposition
    """
    
    def calculate_loss_surface(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        grid_size: int = 50,
        margin: float = 2.0
    ) -> LossSurfaceResult:
        """
        Berechnet die MSE-Fehlerlandschaft über ein Gitter von Parametern (w, b).
        Simuliert zusätzlich einen Gradientenabstiegs-Pfad auf dieser Oberfläche.
        """
        # 1. Optimale Parameter (OLS) finden, um das Gitter zu zentrieren
        X_b = np.column_stack([np.ones(len(X)), X])
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        opt_b, opt_w = theta_best[0], theta_best[1]
        
        # 2. Gitter um das Optimum herum erstellen
        w_range = np.linspace(opt_w - margin, opt_w + margin, grid_size)
        b_range = np.linspace(opt_b - margin, opt_b + margin, grid_size)
        w_grid, b_grid = np.meshgrid(w_range, b_range)
        
        # 3. Fehler (MSE) für jeden Gitterpunkt berechnen
        # Vektorisierte Implementierung für maximale Geschwindigkeit
        n = len(X)
        loss_grid = np.zeros_like(w_grid)
        
        for i in range(grid_size):
            for j in range(grid_size):
                w, b = w_grid[i, j], b_grid[i, j]
                preds = w * X + b
                loss_grid[i, j] = np.mean((preds - y)**2)
                
        # 4. Gradientenabstiegs-Pfad simulieren (Lernprozess)
        # Wir starten bewusst etwas abseits vom Optimum, um den Pfad zu zeigen
        start_w = opt_w - margin * 0.8
        start_b = opt_b - margin * 0.8
        
        path_w, path_b, path_loss = self._simulate_gradient_descent(
            X, y, start_w, start_b, learning_rate=0.1
        )
        
        return LossSurfaceResult(
            w_grid=w_grid,
            b_grid=b_grid,
            loss_grid=loss_grid,
            path_w=path_w,
            path_b=path_b,
            path_loss=path_loss,
            optimal_w=opt_w,
            optimal_b=opt_b
        )
    
    def _simulate_gradient_descent(
        self, X: np.ndarray, y: np.ndarray, w_init: float, b_init: float, learning_rate: float
    ) -> Tuple[List[float], List[float], List[float]]:
        """Simuliert Gradient Descent für die Pfad-Visualisierung."""
        w, b = w_init, b_init
        path_w, path_b, path_loss = [w], [b], []
        
        n = len(X)
        
        # Initialen Fehler berechnen
        initial_pred = w * X + b
        path_loss.append(np.mean((initial_pred - y)**2))
        
        for _ in range(20):  # Kurze Simulation für didaktische Zwecke
            pred = w * X + b
            error = pred - y
            
            # Gradienten-Berechnung
            dw = (2/n) * np.sum(error * X)
            db = (2/n) * np.sum(error)
            
            # Update-Schritt
            w -= learning_rate * dw
            b -= learning_rate * db
            
            path_w.append(w)
            path_b.append(b)
            path_loss.append(np.mean(((w * X + b) - y)**2))
            
        return path_w, path_b, path_loss

    def calculate_overfitting_demo(
        self, n_samples: int = 20, noise: float = 0.3
    ) -> OverfittingDemoResult:
        """
        Generiert Daten und fittet Polynome der Grade 1, 3, 12,
        um Underfitting vs. Optimum vs. Overfitting zu demonstrieren.
        """
        np.random.seed(42)
        
        # Wahre Funktion: sin(2πx)
        features = np.sort(np.random.uniform(0, 1, n_samples))
        target = np.sin(2 * np.pi * features) + np.random.normal(0, noise, n_samples)
        
        # Datensplit
        indices = np.random.permutation(n_samples)
        split = int(n_samples * 0.7)
        train_idx, test_idx = indices[:split], indices[split:]
        
        x_train, y_train = features[train_idx], target[train_idx]
        x_test, y_test = features[test_idx], target[test_idx]
        
        # Dichtes Gitter für glatte Kurvendarstellung
        x_plot = np.linspace(0, 1, 100)
        
        train_metrics = {}
        test_metrics = {}
        predictions = {}
        
        # Fitting für die Grade 1, 3, 12
        for degree in [1, 3, 12]:
            # Polynomialer Fit
            coeffs = np.polyfit(x_train, y_train, degree)
            poly = np.poly1d(coeffs)
            
            # Vorhersagen
            y_plot = poly(x_plot)
            train_pred = poly(x_train)
            test_pred = poly(x_test)
            
            # MSE
            train_metrics[degree] = np.mean((train_pred - y_train)**2)
            test_metrics[degree] = np.mean((test_pred - y_test)**2)
            predictions[degree] = y_plot
            
        return OverfittingDemoResult(
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test,
            x_plot=x_plot,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            predictions=predictions
        )

