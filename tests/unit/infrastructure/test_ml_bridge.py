import pytest
import numpy as np
from src.infrastructure.services.ml_bridge import MLBridgeService

def test_calculate_loss_surface(sample_scaled_data):
    """Test the generation of the 3D loss surface grid and gradient path."""
    service = MLBridgeService()
    X = sample_scaled_data["x"]
    y = sample_scaled_data["y"]
    
    result = service.calculate_loss_surface(X, y, grid_size=20)
    
    # Check grid dimensions
    assert result.w_grid.shape == (20, 20)
    assert result.b_grid.shape == (20, 20)
    assert result.loss_grid.shape == (20, 20)
    
    # Path should have 21 points (initial + 20 iterations)
    assert len(result.path_w) == 21
    assert len(result.path_b) == 21
    assert len(result.path_loss) == 21
    
    # Path should generally decrease loss
    assert result.path_loss[-1] < result.path_loss[0]

def test_calculate_overfitting_demo():
    """Test the overfitting demo data and metrics."""
    service = MLBridgeService()
    result = service.calculate_overfitting_demo(n_samples=30, noise=0.1)
    
    assert len(result.x_train) > 0
    assert len(result.x_test) > 0
    assert 1 in result.train_metrics
    assert 3 in result.train_metrics
    assert 12 in result.train_metrics
    
    # Degree 12 usually fits training data better than Degree 1
    assert result.train_metrics[12] < result.train_metrics[1]
    
    # But Degree 12 usually performs worse on test data than Degree 3 (overfitting)
    # This might depend on seed but 42 is fixed in implementation
    assert result.test_metrics[12] > result.test_metrics[3]
    
    assert len(result.x_plot) == 100
    assert result.predictions[3].shape == (100,)
