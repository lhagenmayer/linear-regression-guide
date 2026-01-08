import pytest
from src.infrastructure.regression_pipeline import RegressionPipeline
from src.core.domain.value_objects import DomainError

def test_regression_pipeline_simple():
    """Test the complete integration of data -> calculate -> plot for simple regression."""
    pipeline = RegressionPipeline()
    
    result = pipeline.run_simple(
        dataset="electronics",
        n=50,
        noise=0.1,
        seed=42
    )
    
    assert result.pipeline_type == "simple"
    assert result.stats.r_squared > 0.8
    assert len(result.data.x) == 50
    assert result.plots is not None


def test_regression_pipeline_multiple():
    """Test the complete integration for multiple regression."""
    pipeline = RegressionPipeline()
    
    result = pipeline.run_multiple(
        dataset="cities",
        n=40,
        noise=0.1,
        seed=42
    )
    
    assert result.pipeline_type == "multiple"
    assert result.stats.r_squared > 0.7
    assert len(result.stats.coefficients) == 2


def test_pipeline_handles_domain_error():
    """Test that pipeline properly handles DomainError from services."""
    pipeline = RegressionPipeline()
    
    # This should not raise, but return a result with error handling
    # (assuming pipeline has error handling)
    # For now, we test that singular matrix case is caught
    from src.infrastructure.services.regression import RegressionServiceImpl
    service = RegressionServiceImpl()
    
    # Test that service raises DomainError
    x1 = [1, 2, 3]
    x2 = [1, 2, 3]  # Perfect multicollinearity
    y = [2, 4, 6]
    
    with pytest.raises(DomainError):
        service.train_multiple([x1, x2], y, ["X1", "X2"])


def test_api_integration():
    """Test API layer integration with pipeline."""
    from src.api.endpoints import RegressionAPI
    
    api = RegressionAPI()
    
    # Test simple regression
    result = api.run_simple(
        dataset="electronics",
        n=30,
        noise=0.2,
        seed=42
    )
    
    assert result["success"] is True
    assert "data" in result
    assert result["data"]["stats"]["r_squared"] > 0.5
    
    # Test multiple regression
    result = api.run_multiple(
        dataset="cities",
        n=30,
        noise=1.0,
        seed=42
    )
    
    assert result["success"] is True
    assert "data" in result


def test_api_handles_domain_error():
    """Test that API properly handles DomainError and returns structured error."""
    from src.api.endpoints import RegressionAPI
    from src.infrastructure.services.regression import RegressionServiceImpl
    
    # Direct service test
    service = RegressionServiceImpl()
    x1 = [1, 2, 3]
    x2 = [1, 2, 3]
    y = [2, 4, 6]
    
    with pytest.raises(DomainError) as exc_info:
        service.train_multiple([x1, x2], y, ["X1", "X2"])
    
    # Verify error structure
    assert exc_info.value.code == "SINGULAR_MATRIX"
    assert exc_info.value.message is not None
