import pytest
import numpy as np
from src.infrastructure.services.calculate import StatisticsCalculator

def test_simple_regression_math(sample_linear_data):
    """Test OLS math for simple regression."""
    calc = StatisticsCalculator()
    x = sample_linear_data["x"]
    y = sample_linear_data["y"]
    
    result = calc.simple_regression(x, y)
    
    # y = 2x + 1 + noise. 
    # With 100 points, coefficients should be close to 2 and 1.
    assert pytest.approx(result.intercept, abs=0.5) == 1.0
    assert pytest.approx(result.slope, abs=0.2) == 2.0
    
    # R-squared should be high for this data
    assert result.r_squared > 0.9
    assert len(result.y_pred) == 100
    assert len(result.residuals) == 100

def test_multiple_regression_math(sample_multiple_linear_data):
    """Test matrix OLS math for multiple regression."""
    calc = StatisticsCalculator()
    x1 = sample_multiple_linear_data["x1"]
    x2 = sample_multiple_linear_data["x2"]
    y = sample_multiple_linear_data["y"]
    
    # multiple_regression(x1, x2, y)
    result = calc.multiple_regression(x1, x2, y)
    
    # y = 3x1 - 2x2 + 5 + noise
    assert pytest.approx(result.intercept, abs=0.5) == 5.0
    assert pytest.approx(result.coefficients[0], abs=0.2) == 3.0
    assert pytest.approx(result.coefficients[1], abs=0.2) == -2.0
    
    assert result.r_squared > 0.95

def test_basic_stats_math():
    """Test basic statistics helper."""
    calc = StatisticsCalculator()
    data = [1, 2, 3, 4, 5]
    stats = calc.basic_stats(data)
    
    assert stats["mean"] == 3.0
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["median"] == 3.0
    assert "std" in stats
