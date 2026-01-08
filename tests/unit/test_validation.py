import pytest
from src.api.endpoints import RegressionAPI
from src.api.schemas import DatasetType

class TestRegressionAPIValidation:
    
    def setup_method(self):
        self.api = RegressionAPI()
    
    def test_valid_simple_regression(self):
        """Test valid simple regression request."""
        # Using allowed N=50
        result = self.api.run_simple(dataset="electronics", n=50)
        assert result["success"] is True
        assert "data" in result
    
    def test_invalid_n_too_small(self):
        """Test N too small."""
        result = self.api.run_simple(n=5)
        assert result["success"] is False
        assert result["error"] == "Validation Error"
        # Pydantic details should be present
        assert any(e["loc"] == ("n",) for e in result["details"])
    
    def test_invalid_n_too_large(self):
        """Test N too large."""
        result = self.api.run_simple(n=5000)
        assert result["success"] is False
        assert "Validation Error" in result["error"]

    def test_invalid_dataset(self):
        """Test invalid dataset name."""
        result = self.api.run_simple(dataset="invalid_dataset")
        assert result["success"] is False
        assert "Validation Error" in result["error"]
        assert any(e["loc"] == ("dataset",) for e in result["details"])
        
    def test_invalid_noise_negative(self):
        """Test negative noise."""
        result = self.api.run_simple(noise=-1.0)
        assert result["success"] is False
        assert "Validation Error" in result["error"]

    def test_valid_multiple_regression(self):
        """Test valid multiple regression."""
        result = self.api.run_multiple(dataset="cities", n=100)
        assert result["success"] is True

    def test_invalid_multiple_regression_dataset(self):
        """Test invalid dataset for multiple regression (should still fail enum check)."""
        result = self.api.run_multiple(dataset="foobar")
        assert result["success"] is False
        assert "Validation Error" in result["error"]
