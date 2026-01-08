import pytest
import numpy as np
from src.infrastructure.services.classification import ClassificationServiceImpl

def test_logistic_regression_training(sample_classification_data):
    """Test training of logistic regression on binary data."""
    service = ClassificationServiceImpl()
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    result = service.train_logistic(X, y, iterations=500)
    
    assert result.is_success
    assert len(result.classes) == 2
    assert result.metrics.accuracy > 0.9  # Should solve this easy cluster problem
    assert "coefficients" in result.model_params
    assert "intercept" in result.model_params

def test_knn_training(sample_classification_data):
    """Test training of KNN on binary data."""
    service = ClassificationServiceImpl()
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    result = service.train_knn(X, y, k=3)
    
    assert result.metrics.accuracy > 0.95
    assert result.model_params["k"] == 3
    assert "X_train" in result.model_params

def test_classification_metrics():
    """Test calculation of classification metrics via service."""
    service = ClassificationServiceImpl()
    
    # Simple binary case
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1]) # 2 correctly, 2 incorrectly
    y_prob = np.array([0.9, 0.4, 0.1, 0.8])
    
    metrics = service.calculate_metrics(y_true, y_pred, y_prob)
    
    assert metrics.accuracy == 0.5
    # TP=1, TN=1, FP=1, FN=1
    # Precision = 1/(1+1) = 0.5
    # Recall = 1/(1+1) = 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1_score == 0.5

def test_predict_methods(sample_classification_data):
    """Test predict_logistic and predict_knn with previously trained parameters."""
    service = ClassificationServiceImpl()
    X = sample_classification_data["X"]
    y = sample_classification_data["y"]
    
    # 1. Logistic
    log_res = service.train_logistic(X, y, iterations=200)
    preds, probs = service.predict_logistic(X, log_res.model_params)
    assert len(preds) == len(y)
    assert np.all((probs >= 0.5).astype(int) == preds)
    
    # 2. KNN
    knn_res = service.train_knn(X, y, k=3)
    preds, probs = service.predict_knn(X, knn_res.model_params)
    assert len(preds) == len(y)
    # Binary KNN probabilities are probabilities for class 1
    assert probs.shape == (len(y),)
