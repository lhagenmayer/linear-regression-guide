import pytest
import numpy as np
from src.infrastructure.data.provider import DataProviderImpl
from src.infrastructure.data.registry import DatasetRegistry

def test_dataset_registry_completeness():
    """Verify that the registry contains expected datasets."""
    registry = DatasetRegistry()
    all_datasets = registry.list_all()
    # Check for some common names
    names = [d.name for d in all_datasets]
    assert "electronics" in names
    assert "advertising" in names

def test_data_provider_simple_regression():
    """Test fetching simple regression data from provider."""
    provider = DataProviderImpl()
    result = provider.get_dataset("electronics", n=50, noise=0.1)
    
    assert "x" in result
    assert "y" in result
    assert len(result["x"]) == 50
    assert "metadata" in result
    assert result["x_label"] != ""

def test_data_provider_multiple_regression():
    """Test fetching multiple regression data."""
    provider = DataProviderImpl()
    # 'electronics' is simple only, 'cities' supports multiple
    result = provider.get_dataset("cities", n=30, noise=0.1, regression_type="multiple")
    
    assert "x1" in result
    assert "x2" in result
    assert len(result["x1"]) == 30
    assert "x1_label" != ""

def test_data_provider_classification():
    """Test fetching classification data."""
    provider = DataProviderImpl()
    # "fruits" is a classification dataset - it has 4 classes in our implementation
    result = provider.get_dataset("fruits", n=40, analysis_type="classification")
    
    assert "X" in result
    assert "y" in result
    # It returns lists in DataProviderImpl
    assert len(result["X"]) == 40
    # fruits has 2 real + 2 noise features = 4
    assert len(result["X"][0]) == 4
    assert len(result["y"]) == 40
    # fruits has 4 classes (Apples, Berries, etc. in implementation)
    assert len(set(result["y"])) == 4

def test_data_provider_invalid_dataset():
    """Test error handling for non-existent datasets."""
    provider = DataProviderImpl()
    # DataFetcher.get_simple fallbacks to 'electronics' if name is unknown, 
    # but we want it to be strict.
    # Let's see if we should fix the implementation or the test.
    # The user wants 'super strict', so let's fix the implementation to raise on unknown.
    with pytest.raises(ValueError, match="Unbekannter Datensatz"):
        provider.get_dataset("really_invalid_id_123", n=10)

def test_get_raw_data():
    """Test raw data retrieval for table previews."""
    provider = DataProviderImpl()
    raw = provider.get_raw_data("electronics")
    
    assert "columns" in raw
    assert "data" in raw
    assert len(raw["columns"]) > 0
    assert len(raw["data"]) > 0
