import pytest
from unittest.mock import MagicMock
from src.core.application.dtos import RegressionRequestDTO, RegressionResponseDTO
from src.core.application.use_cases import RunRegressionUseCase
from src.core.domain.value_objects import RegressionType, RegressionParameters, RegressionMetrics, ModelQuality
from src.core.domain.entities import RegressionModel

def test_regression_request_dto_validation():
    """Test that RegressionRequestDTO validates its input."""
    # Valid
    dto = RegressionRequestDTO(dataset_id="test", n_observations=100, noise_level=0.1, seed=42)
    assert dto.dataset_id == "test"
    
    # Invalid: n < 2
    with pytest.raises(ValueError, match="n_observations muss >= 2 sein"):
        RegressionRequestDTO(dataset_id="test", n_observations=1, noise_level=0.1, seed=42)
    
    # Invalid: negative noise
    with pytest.raises(ValueError, match="noise_level darf nicht negativ sein"):
        RegressionRequestDTO(dataset_id="test", n_observations=100, noise_level=-0.1, seed=42)

def test_run_regression_use_case_simple():
    """Test RunRegressionUseCase for simple regression."""
    # Mocks
    mock_data_provider = MagicMock()
    mock_regression_service = MagicMock()
    
    # Mock data result
    mock_data_result = {
        "x": [1.0, 2.0, 3.0],
        "y": [2.0, 4.0, 6.0],
        "x_label": "Hours",
        "y_label": "Score",
        "context_title": "Test Title",
        "context_description": "Test Desc",
        "metadata": MagicMock()
    }
    mock_data_provider.get_dataset.return_value = mock_data_result
    
    # Mock model
    model = RegressionModel()
    model.parameters = RegressionParameters(intercept=0.0, coefficients={"x": 2.0})
    model.metrics = RegressionMetrics(r_squared=1.0, r_squared_adj=1.0, mse=0.0, rmse=0.0, p_value=0.0)
    model.predictions = [2.0, 4.0, 6.0]
    model.residuals = [0.0, 0.0, 0.0]
    
    mock_regression_service.train_simple.return_value = model
    
    # Execute Use Case
    use_case = RunRegressionUseCase(mock_data_provider, mock_regression_service)
    request = RegressionRequestDTO(dataset_id="test", n_observations=3, noise_level=0.0, seed=42)
    response = use_case.execute(request)
    
    # Verifications
    assert isinstance(response, RegressionResponseDTO)
    assert response.success
    assert response.coefficients == {"x": 2.0}
    assert response.x_data == (1.0, 2.0, 3.0)
    assert response.y_data == (2.0, 4.0, 6.0)
    assert response.r_squared == 1.0
    assert response.slope == 2.0
    
    mock_data_provider.get_dataset.assert_called_once_with(
        dataset_id="test",
        n=3,
        noise=0.0,
        seed=42,
        true_intercept=0.6,
        true_slope=0.52,
        regression_type="simple"
    )
    mock_regression_service.train_simple.assert_called_once()

def test_run_regression_use_case_multiple():
    """Test RunRegressionUseCase for multiple regression."""
    mock_data_provider = MagicMock()
    mock_regression_service = MagicMock()
    
    mock_data_result = {
        "x1": [1.0, 2.0],
        "x2": [0.5, 1.5],
        "y": [10.0, 20.0],
        "x1_label": "Feature 1",
        "x2_label": "Feature 2",
        "y_label": "Target",
        "metadata": MagicMock()
    }
    mock_data_provider.get_dataset.return_value = mock_data_result
    
    model = RegressionModel()
    model.parameters = RegressionParameters(intercept=0.0, coefficients={"x1": 5.0, "x2": 10.0})
    model.metrics = RegressionMetrics(r_squared=0.9, r_squared_adj=0.8, mse=1.0, rmse=1.0)
    model.predictions = [10.0, 20.0]
    model.residuals = [0.0, 0.0]
    
    mock_regression_service.train_multiple.return_value = model
    
    use_case = RunRegressionUseCase(mock_data_provider, mock_regression_service)
    request = RegressionRequestDTO(
        dataset_id="test", 
        n_observations=2, 
        noise_level=0.1, 
        seed=42,
        regression_type=RegressionType.MULTIPLE
    )
    response = use_case.execute(request)
    
    assert response.success
    assert response.coefficients == {"x1": 5.0, "x2": 10.0}
    assert response.x_data == ((1.0, 2.0), (0.5, 1.5))
    assert response.x_label == "Feature 1 & Feature 2"
    
    mock_regression_service.train_multiple.assert_called_once()
