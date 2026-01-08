import pytest
import numpy as np
from src.infrastructure.services.regression import RegressionServiceImpl
from src.core.domain.value_objects import DomainError

def test_singular_matrix_error():
    """Test that regression service raises DomainError for singular matrices (strict mode)."""
    service = RegressionServiceImpl()
    
    # Perfect multicollinearity: X1 and X2 are identical
    x1 = [1, 2, 3, 4, 5]
    x2 = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    with pytest.raises(DomainError) as exc_info:
        service.train_multiple([x1, x2], y, ["V1", "V2"])
    
    assert exc_info.value.code == "SINGULAR_MATRIX"
    assert "singulär" in exc_info.value.message or "singular" in exc_info.value.message.lower()

def test_minimal_data_points():
    """Test regression with minimal data points (n=2)."""
    service = RegressionServiceImpl()
    
    x = [1, 2]
    y = [10, 20]
    
    # Simple regression should work with 2 points (exact fit)
    result = service.train_simple(x, y)
    assert result.parameters.intercept == pytest.approx(0.0)
    assert result.parameters.coefficients["x"] == pytest.approx(10.0)
    assert result.metrics.r_squared == pytest.approx(1.0)


def test_constant_features():
    """Test regression with constant features (should raise DomainError)."""
    service = RegressionServiceImpl()
    
    # Constant X (all same value) - leads to singular matrix
    x = [5.0, 5.0, 5.0, 5.0, 5.0]
    y = [10, 12, 11, 13, 10]
    
    # Should raise DomainError for constant features (ss_xx = 0)
    with pytest.raises((DomainError, ValueError)):
        result = service.train_simple(x, y)
        # If it doesn't raise, check that it handles gracefully
        # (but in strict mode, we expect an error)


def test_extreme_values():
    """Test regression with extreme values (very large/small numbers)."""
    service = RegressionServiceImpl()
    
    # Very large values
    x_large = [1e10, 2e10, 3e10, 4e10, 5e10]
    y_large = [1e12, 2e12, 3e12, 4e12, 5e12]
    
    result = service.train_simple(x_large, y_large)
    # Slope should be approximately 100 (y = 100x)
    assert result.parameters.coefficients["x"] == pytest.approx(100.0, rel=1e-6)
    
    # Very small values
    x_small = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
    y_small = [1e-8, 2e-8, 3e-8, 4e-8, 5e-8]
    
    result = service.train_simple(x_small, y_small)
    # Slope should be approximately 100 (y = 100x)
    assert result.parameters.coefficients["x"] == pytest.approx(100.0, rel=1e-6)


def test_multiple_regression_minimal():
    """Test multiple regression with minimal data (n=3 for 2 predictors raises DomainError)."""
    service = RegressionServiceImpl()
    
    x1 = [1, 2, 3]
    x2 = [0, 1, 2]
    y = [5, 10, 15]  # y = 5 + 5*x1 + 0*x2
    
    # n=3 with k=2 predictors + intercept = 3 parameters, but only 3 data points
    # This creates a singular matrix (perfect fit, but no degrees of freedom)
    with pytest.raises(DomainError) as exc_info:
        service.train_multiple([x1, x2], y, ["X1", "X2"])
    
    assert exc_info.value.code == "SINGULAR_MATRIX"


def test_multiple_regression_constant_feature():
    """Test multiple regression with one constant feature (raises DomainError)."""
    service = RegressionServiceImpl()
    
    x1 = [1, 2, 3, 4, 5]
    x2 = [10, 10, 10, 10, 10]  # Constant feature creates perfect multicollinearity with intercept
    y = [5, 10, 15, 20, 25]  # y = 5*x1
    
    # Constant feature + intercept creates singular matrix
    with pytest.raises(DomainError) as exc_info:
        service.train_multiple([x1, x2], y, ["X1", "X2"])
    
    assert exc_info.value.code == "SINGULAR_MATRIX"


def test_singular_matrix_insufficient_data():
    """Test that insufficient data (n < k+1) raises DomainError or ValueError."""
    service = RegressionServiceImpl()
    
    # n=2, but k=2 predictors (need at least n=3)
    # This creates a singular matrix, but may also cause invalid R²
    x1 = [1, 2]
    x2 = [3, 4]
    y = [5, 6]
    
    # May raise DomainError (singular matrix) or ValueError (invalid R²)
    with pytest.raises((DomainError, ValueError)):
        service.train_multiple([x1, x2], y, ["X1", "X2"])
