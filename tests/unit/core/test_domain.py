import pytest
import uuid
from src.core.domain.entities import RegressionModel
from src.core.domain.value_objects import (
    RegressionParameters,
    RegressionMetrics,
    ModelQuality,
    RegressionType,
    ClassificationMetrics,
    DatasetMetadata
)

def test_regression_parameters_validation():
    """Test that RegressionParameters validates its input."""
    # Valid parameters
    params = RegressionParameters(intercept=1.0, coefficients={"x": 2.0})
    assert params.intercept == 1.0
    assert params.coefficients == {"x": 2.0}
    assert params.slope == 2.0
    
    # Invalid: empty coefficients
    with pytest.raises(ValueError, match="coefficients darf nicht leer sein"):
        RegressionParameters(intercept=1.0, coefficients={})

def test_regression_metrics_validation():
    """Test that RegressionMetrics validates its input."""
    # Valid metrics
    metrics = RegressionMetrics(r_squared=0.8, r_squared_adj=0.79, mse=1.0, rmse=1.0)
    assert metrics.quality == ModelQuality.EXCELLENT
    
    # Invalid: r_squared out of range
    with pytest.raises(ValueError, match="r_squared muss zwischen 0 und 1 liegen"):
        RegressionMetrics(r_squared=1.5, r_squared_adj=1.4, mse=1.0, rmse=1.0)
    
    # Invalid: negative mse
    with pytest.raises(ValueError, match="mse darf nicht negativ sein"):
        RegressionMetrics(r_squared=0.5, r_squared_adj=0.4, mse=-1.0, rmse=1.0)

def test_regression_model_logic():
    """Test the business logic inside the RegressionModel entity."""
    model = RegressionModel(regression_type=RegressionType.SIMPLE)
    assert not model.is_trained()
    assert model.get_equation_string() == "Nicht trainiert"
    
    # Train the model with parameters and metrics
    params = RegressionParameters(intercept=1.0, coefficients={"x": 2.0})
    metrics = RegressionMetrics(r_squared=0.8, r_squared_adj=0.79, mse=1.0, rmse=1.0, p_value=0.01)
    
    model.parameters = params
    model.metrics = metrics
    model.predictions = [1.0, 2.0]
    model.residuals = [0.1, -0.1]
    
    assert model.is_trained()
    assert model.get_equation_string() == "ŷ = 1.0000 + 2.0000·x"
    assert model.get_quality() == ModelQuality.EXCELLENT
    assert model.is_significant(alpha=0.05)
    assert not model.is_significant(alpha=0.001)
    assert len(model.validate()) == 0

def test_regression_model_validation():
    """Test the validation logic of RegressionModel."""
    model = RegressionModel()
    # Not trained, no errors
    assert len(model.validate()) == 0
    
    # Trained but missing results
    model.parameters = RegressionParameters(intercept=1.0, coefficients={"x": 2.0})
    model.metrics = RegressionMetrics(r_squared=0.8, r_squared_adj=0.79, mse=1.0, rmse=1.0)
    
    errors = model.validate()
    assert "Ein trainiertes Modell sollte Vorhersagen enthalten" in errors
    assert "Ein trainiertes Modell sollte Residuen enthalten" in errors
    
    # Mismatching lengths
    model.predictions = [1.0, 2.0]
    model.residuals = [0.1]
    assert "Vorhersagen und Residuen müssen die gleiche Länge haben" in model.validate()

def test_classification_metrics_validation():
    """Test that ClassificationMetrics validates its input."""
    # Valid
    metrics = ClassificationMetrics(accuracy=0.9, precision=0.8, recall=0.7, f1_score=0.75, confusion_matrix=[[1, 0], [0, 1]])
    assert metrics.accuracy == 0.9
    
    # Invalid: accuracy out of range
    with pytest.raises(ValueError, match="accuracy muss zwischen 0 und 1 liegen"):
        ClassificationMetrics(accuracy=1.1, precision=0.8, recall=0.7, f1_score=0.75, confusion_matrix=[])

def test_dataset_metadata_validation():
    """Test that DatasetMetadata validates its input."""
    # Valid
    meta = DatasetMetadata(id="test", name="Test", description="Desc", source="Src", variables=("x", "y"), n_observations=100)
    assert meta.id == "test"
    
    # Invalid: negative n_observations
    with pytest.raises(ValueError, match="n_observations darf nicht negativ sein"):
        DatasetMetadata(id="test", name="Test", description="Desc", source="Src", variables=("x", "y"), n_observations=-1)
    
    # Invalid: empty id
    with pytest.raises(ValueError, match="id darf nicht leer sein"):
        DatasetMetadata(id="", name="Test", description="Desc", source="Src", variables=("x", "y"), n_observations=10)
