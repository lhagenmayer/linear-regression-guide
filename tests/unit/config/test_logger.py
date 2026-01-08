"""
Unit Tests für Error-Logging-System.

Testet die strukturierte Error-Logging-Funktionalität.
"""
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
import uuid

from src.config.logger import (
    log_error_with_context,
    log_domain_error,
    log_api_error,
    log_service_error,
    error_handler,
    get_error_tracker,
    ErrorTracker
)
from src.core.domain.value_objects import DomainError


class TestErrorLoggingFunctions:
    """Tests für Error-Logging-Funktionen."""
    
    @pytest.fixture
    def logger(self):
        """Erstellt einen Mock-Logger."""
        logger = Mock(spec=logging.Logger)
        return logger
    
    def test_log_error_with_context(self, logger):
        """Test allgemeines Error-Logging mit Kontext."""
        error = ValueError("Test error")
        
        error_id = log_error_with_context(
            logger=logger,
            error=error,
            context="test_operation",
            param1="value1",
            param2=42
        )
        
        assert len(error_id) == 8  # UUID-8-Zeichen
        logger.error.assert_called_once()
        call_args = logger.error.call_args
        assert "ERROR_ID=" in call_args[0][0]
        assert "test_operation" in call_args[0][0]
        assert call_args[1]["exc_info"] is True
    
    def test_log_error_with_custom_id(self, logger):
        """Test Error-Logging mit benutzerdefinierter Error-ID."""
        error = ValueError("Test error")
        
        error_id = log_error_with_context(
            logger=logger,
            error=error,
            context="test_operation",
            error_id="custom123"
        )
        
        assert error_id == "custom123"
        call_args = logger.error.call_args
        assert "ERROR_ID=custom123" in call_args[0][0]
    
    def test_log_domain_error(self, logger):
        """Test DomainError-Logging."""
        error = DomainError("Domain error", code="TEST_ERROR")
        
        error_id = log_domain_error(
            logger=logger,
            error=error,
            context="domain_operation",
            dataset="test"
        )
        
        assert len(error_id) == 8
        call_args = logger.error.call_args
        assert "TEST_ERROR" in call_args[0][0]
        assert "domain_operation" in call_args[0][0]
    
    def test_log_api_error(self, logger):
        """Test API-Error-Logging."""
        error = ValueError("API error")
        
        error_id = log_api_error(
            logger=logger,
            error=error,
            endpoint="/api/test",
            method="POST",
            request_data={"param": "value"},
            user_id="user123"
        )
        
        assert len(error_id) == 8
        call_args = logger.error.call_args
        assert "/api/test" in call_args[0][0]
        assert "POST" in call_args[0][0]
    
    def test_log_service_error(self, logger):
        """Test Service-Error-Logging."""
        error = ValueError("Service error")
        
        error_id = log_service_error(
            logger=logger,
            error=error,
            service_name="TestService",
            operation="test_operation",
            input_params={"param": "value"}
        )
        
        assert len(error_id) == 8
        call_args = logger.error.call_args
        assert "TestService" in call_args[0][0]
        assert "test_operation" in call_args[0][0]


class TestErrorHandlerDecorator:
    """Tests für error_handler Decorator."""
    
    @pytest.fixture
    def logger(self):
        """Erstellt einen Mock-Logger."""
        logger = Mock(spec=logging.Logger)
        return logger
    
    def test_error_handler_success(self, logger):
        """Test dass Decorator bei Erfolg nichts tut."""
        @error_handler(logger, "test_context")
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
        logger.error.assert_not_called()
    
    def test_error_handler_logs_error(self, logger):
        """Test dass Decorator Fehler loggt."""
        @error_handler(logger, "test_context")
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_func()
        
        logger.error.assert_called_once()
        call_args = logger.error.call_args
        assert "test_context" in call_args[0][0]
    
    def test_error_handler_with_args(self, logger):
        """Test Decorator mit Funktions-Argumenten."""
        @error_handler(logger)
        def test_func(x, y, password="secret"):
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            test_func(1, 2, password="secret")
        
        # Password sollte nicht geloggt werden
        call_args = logger.error.call_args
        assert "password" not in str(call_args)


class TestErrorTracker:
    """Tests für ErrorTracker."""
    
    def test_record_error(self):
        """Test Fehleraufzeichnung."""
        tracker = ErrorTracker()
        
        tracker.record_error(
            error_id="test123",
            error_type="ValueError",
            error_code="TEST_ERROR",
            context="test_context"
        )
        
        assert "test123" in tracker.errors
        assert tracker.errors["test123"]["error_type"] == "ValueError"
        assert tracker.error_counts["ValueError:TEST_ERROR"] == 1
    
    def test_get_error_summary(self):
        """Test Error-Summary."""
        tracker = ErrorTracker()
        
        tracker.record_error("id1", "ValueError", "TEST1", "ctx1")
        tracker.record_error("id2", "ValueError", "TEST1", "ctx2")
        tracker.record_error("id3", "TypeError", "TEST2", "ctx3")
        
        summary = tracker.get_error_summary()
        
        assert summary["total_errors"] == 3
        assert summary["error_counts"]["ValueError:TEST1"] == 2
        assert summary["error_counts"]["TypeError:TEST2"] == 1
        assert len(summary["recent_errors"]) == 3
    
    def test_get_error_tracker_singleton(self):
        """Test dass get_error_tracker eine Singleton-Instanz zurückgibt."""
        tracker1 = get_error_tracker()
        tracker2 = get_error_tracker()
        
        assert tracker1 is tracker2
