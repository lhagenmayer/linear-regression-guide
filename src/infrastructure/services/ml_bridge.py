"""
Step 2: CALCULATE (ML Bridge)

This module implements calculations for the "Statistics -> ML" bridge chapter.
It provides data generation for visualizations like Loss Landscapes,
Gradient Descent paths, and Overfitting demonstrations.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from ...config import get_logger

logger = get_logger(__name__)


@dataclass
class LossSurfaceResult:
    """Data for 3D Loss Surface visualization."""
    w_grid: np.ndarray  # Grid of weights (X axis)
    b_grid: np.ndarray  # Grid of biases (Y axis)
    loss_grid: np.ndarray  # Grid of losses (Z axis)
    path_w: List[float]  # Gradient descent path (weights)
    path_b: List[float]  # Gradient descent path (biases)
    path_loss: List[float]  # Gradient descent path (losses)
    optimal_w: float
    optimal_b: float


@dataclass
class OptimizationResult:
    """Result of an optimization process (e.g., Gradient Descent)."""
    iterations: List[int]
    losses: List[float]
    final_w: float
    final_b: float
    converged: bool


@dataclass
class OverfittingDemoResult:
    """Data for polynomial overfitting demonstration."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_plot: np.ndarray  # Dense grid for plotting curves
    train_metrics: Dict[int, float]  # degree -> MSE
    test_metrics: Dict[int, float]   # degree -> MSE
    predictions: Dict[int, np.ndarray] # degree -> y_plot prediction


class MLBridgeService:
    """
    Service for calculating ML bridge concepts.
    
    Principles:
    1. Grid generation for 3D surfaces
    2. Iterative optimization simulation
    3. Bias-Variance decomposition helpers
    """
    
    def calculate_loss_surface(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        grid_size: int = 50,
        margin: float = 2.0
    ) -> LossSurfaceResult:
        """
        Calculate MSE loss surface over a grid of parameters (w, b).
        Also simulates a gradient descent path on this surface.
        """
        # 1. Find optimal parameters (OLS) for centering
        X_b = np.column_stack([np.ones(len(X)), X])
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        opt_b, opt_w = theta_best[0], theta_best[1]
        
        # 2. Create grid around optimum
        w_range = np.linspace(opt_w - margin, opt_w + margin, grid_size)
        b_range = np.linspace(opt_b - margin, opt_b + margin, grid_size)
        w_grid, b_grid = np.meshgrid(w_range, b_range)
        
        # 3. Calculate Loss (MSE) for each grid point
        # Vectorized implementation for speed
        n = len(X)
        loss_grid = np.zeros_like(w_grid)
        
        for i in range(grid_size):
            for j in range(grid_size):
                w, b = w_grid[i, j], b_grid[i, j]
                preds = w * X + b
                loss_grid[i, j] = np.mean((preds - y)**2)
                
        # 4. Simulate Gradient Descent Path (Learning)
        # Start far from optimum to show path
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
        """Simulate GD for path visualization."""
        w, b = w_init, b_init
        path_w, path_b, path_loss = [w], [b], []
        
        n = len(X)
        
        # Compute initial loss
        initial_pred = w * X + b
        path_loss.append(np.mean((initial_pred - y)**2))
        
        for _ in range(20):  # Short simulation for visualization
            pred = w * X + b
            error = pred - y
            
            # Gradients
            dw = (2/n) * np.sum(error * X)
            db = (2/n) * np.sum(error)
            
            # Update
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
        Generate data and fit polynomials of degrees 1, 3, 10
        to demonstrate Underfitting vs Optimal vs Overfitting.
        """
        np.random.seed(42)
        
        # True function: sin(2πx)
        features = np.sort(np.random.uniform(0, 1, n_samples))
        target = np.sin(2 * np.pi * features) + np.random.normal(0, noise, n_samples)
        
        # Split
        indices = np.random.permutation(n_samples)
        split = int(n_samples * 0.7)
        train_idx, test_idx = indices[:split], indices[split:]
        
        x_train, y_train = features[train_idx], target[train_idx]
        x_test, y_test = features[test_idx], target[test_idx]
        
        # Dense grid for plotting smooth lines
        x_plot = np.linspace(0, 1, 100)
        
        train_metrics = {}
        test_metrics = {}
        predictions = {}
        
        # Fit degrees 1, 3, 12
        for degree in [1, 3, 12]:
            # Polynomial fit
            coeffs = np.polyfit(x_train, y_train, degree)
            poly = np.poly1d(coeffs)
            
            # Predictions
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

