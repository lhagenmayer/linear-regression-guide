"""
Unit Tests für Dependency Injection Container.

Testet die korrekte Wire-Up von Dependencies.
"""
import pytest
from unittest.mock import Mock, patch

from src.container import Container
from src.core.application.use_cases import RunRegressionUseCase, RunClassificationUseCase


class TestContainer:
    """Tests für Container."""
    
    def test_container_initialization(self):
        """Test Container-Initialisierung."""
        container = Container()
        
        assert container._data_provider is not None
        assert container._regression_service is not None
        assert container._classification_service is not None
        assert container._ml_bridge_service is not None
    
    def test_run_regression_use_case(self):
        """Test RunRegressionUseCase-Property."""
        container = Container()
        use_case = container.run_regression_use_case
        
        assert isinstance(use_case, RunRegressionUseCase)
        assert use_case.data_provider is container._data_provider
        assert use_case.regression_service is container._regression_service
    
    def test_classification_service(self):
        """Test ClassificationService-Property."""
        container = Container()
        service = container.classification_service
        
        assert service is container._classification_service
    
    def test_ml_bridge_service(self):
        """Test MLBridgeService-Property."""
        container = Container()
        service = container.ml_bridge_service
        
        assert service is container._ml_bridge_service
    
    def test_run_classification_use_case(self):
        """Test RunClassificationUseCase-Property."""
        container = Container()
        use_case = container.run_classification_use_case
        
        assert isinstance(use_case, RunClassificationUseCase)
        assert use_case.data_provider is container._data_provider
        assert use_case.classification_service is container._classification_service
