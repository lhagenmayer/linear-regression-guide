"""
unit tests for ML services (Classification & Bridge).
"""
import pytest
import numpy as np
from src.container import Container
from src.infrastructure.services.classification import ClassificationServiceImpl
from src.infrastructure.services.ml_bridge import MLBridgeService

class TestClassificationService:
    def setup_method(self):
        self.service = ClassificationServiceImpl()
    
    def test_metrics_perfect(self):
        """Test metrics calculation with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        
        metrics = self.service.calculate_metrics(y_true, y_pred, y_prob)
        
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.confusion_matrix[0,0] == 2  # TN
        assert metrics.confusion_matrix[1,1] == 2  # TP
    
    def test_knn_logic(self):
        """Test KNN with simple known points."""
        # Train points: (1,1) -> 0, (5,5) -> 1
        X = np.array([[1,1], [5,5]])
        y = np.array([0, 1])
        
        # Test point (1.1, 1.1) should be 0
        # But train_knn returns training metrics, so check it fits training data perfect
        result = self.service.train_knn(X, y, k=1)
        
        assert result.metrics.accuracy == 1.0
        assert result.predictions[0] == 0
        assert result.predictions[1] == 1

    def test_logistic_logic(self):
        """Test Logistic Regression simple convergence."""
        # Simple linearly separable data
        X = np.array([[1], [2], [8], [9]])
        y = np.array([0, 0, 1, 1])
        
        result = self.service.train_logistic(X, y, learning_rate=0.1, iterations=100)
        
        assert result.metrics.accuracy == 1.0
        assert result.predictions[0] == 0  # Low X -> 0
        assert result.predictions[3] == 1  # High X -> 1


class TestMLBridgeService:
    def setup_method(self):
        self.service = MLBridgeService()
        
    def test_loss_surface_shape(self):
        """Test grid generation."""
        X = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        result = self.service.calculate_loss_surface(X, y, grid_size=10)
        
        assert result.w_grid.shape == (10, 10)
        assert result.b_grid.shape == (10, 10)
        assert len(result.path_w) > 0
        
    def test_overfitting_demo(self):
        """Test overfitting matrices."""
        result = self.service.calculate_overfitting_demo(n_samples=10)
        
        assert len(result.x_train) > 0
        assert 1 in result.train_metrics
        assert 12 in result.train_metrics
        # Degree 12 usually fits training better (lower MSE) than Degree 1
        assert result.train_metrics[12] < result.train_metrics[1]
