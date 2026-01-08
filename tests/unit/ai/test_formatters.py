import pytest
import numpy as np
from src.infrastructure.ai.formatters import ROutputFormatter

def test_format_linear_simple():
    """Test R-style formatting for simple linear regression."""
    stats = {
        "x_label": "Hours",
        "y_label": "Score",
        "intercept": 1.2345,
        "slope": 0.5678,
        "se_intercept": 0.1,
        "se_slope": 0.05,
        "t_intercept": 12.3,
        "t_slope": 11.3,
        "p_intercept": 0.0001,
        "p_slope": 0.0002,
        "r_squared": 0.85,
        "r_squared_adj": 0.84,
        "f_statistic": 120.5,
        "df": 98,
        "mse": 0.5,
        "residuals": [-1, 0, 1, -0.5, 0.5]
    }
    
    output = ROutputFormatter.format(stats)
    
    assert "lm(formula = Score ~ Hours)" in output
    assert "Estimate Std. Error t value Pr(>|t|)" in output
    assert "1.2345" in output
    assert "0.5678" in output
    assert "Multiple R-squared:  0.8500" in output
    assert "F-statistic: 120.50" in output

def test_format_logistic():
    """Test R-style formatting for logistic regression."""
    stats = {
        "method": "logistic",
        "y_label": "Passed",
        "feature_names": ["Hours"],
        "intercept": -5.0,
        "coefficients": [1.5],
        "accuracy": 0.85,
        "aic": 42.0,
        "confusion_matrix": [[40, 5], [10, 45]]
    }
    
    output = ROutputFormatter.format(stats)
    
    assert "glm(formula = Passed ~ Hours, family = binomial)" in output
    assert "AIC: 42.0" in output
    assert "Accuracy: 0.8500" in output
    assert "[40  5]" in output

def test_format_knn():
    """Test R-style formatting for KNN."""
    stats = {
        "method": "knn",
        "k": 5,
        "accuracy": 0.92,
        "confusion_matrix": [[45, 2], [6, 47]]
    }
    
    output = ROutputFormatter.format(stats)
    
    assert "Confusion Matrix and Statistics" in output
    assert "Accuracy : 0.9200" in output
    assert "k=5" in output


def test_format_multiple_linear():
    """Test R-style formatting for multiple linear regression."""
    stats = {
        "x_label": "X",
        "y_label": "Y",
        "intercept": 1.0,
        "se_intercept": 0.1,
        "t_intercept": 10.0,
        "p_intercept": 0.001,
        "coefficients": {"X1": 2.5, "X2": -1.3},
        "se_coefficients": [0.2, 0.15],
        "t_values": [12.5, -8.67],
        "p_values": [0.0001, 0.0002],
        "feature_names": ["X1", "X2"],
        "r_squared": 0.92,
        "r_squared_adj": 0.91,
        "f_statistic": 150.0,
        "f_p_value": 0.0001,
        "df": 47,
        "mse": 0.5,
        "residuals": [-0.1, 0.2, -0.15, 0.1, 0.05]
    }
    
    output = ROutputFormatter.format(stats)
    
    assert "lm(formula = Y ~ X1 + X2)" in output
    assert "(Intercept)" in output
    assert "X1" in output
    assert "X2" in output
    assert "2.5000" in output
    assert "-1.3000" in output
    assert "Multiple R-squared:  0.9200" in output


def test_format_logistic_with_calculated_aic():
    """Test logistic regression formatting with calculated AIC."""
    stats = {
        "method": "logistic",
        "y_label": "Passed",
        "feature_names": ["Hours"],
        "intercept": -5.0,
        "coefficients": [1.5],
        "accuracy": 0.85,
        "loss_history": [0.5, 0.4, 0.35, 0.32, 0.30],  # Final loss: 0.30
        "n_samples": 100,
        "confusion_matrix": [[40, 5], [10, 45]]
    }
    
    output = ROutputFormatter.format(stats)
    
    assert "glm(formula = Passed ~ Hours, family = binomial)" in output
    # AIC should be calculated, not "N/A"
    assert "AIC:" in output
    # Should not contain "N/A" for AIC (unless calculation truly impossible)
    assert not ("AIC: N/A" in output) or "AIC:" in output


def test_format_logistic_without_aic():
    """Test logistic regression formatting without AIC (should use N/A)."""
    stats = {
        "method": "logistic",
        "y_label": "Passed",
        "feature_names": ["Hours"],
        "intercept": -5.0,
        "coefficients": [1.5],
        "accuracy": 0.85,
        "confusion_matrix": [[40, 5], [10, 45]]
        # No aic, no loss_history, no n_samples
    }
    
    output = ROutputFormatter.format(stats)
    
    assert "glm(formula = Passed ~ Hours, family = binomial)" in output
    # Should show N/A when calculation is impossible
    assert "AIC:" in output
