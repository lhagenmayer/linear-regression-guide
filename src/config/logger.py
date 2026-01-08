"""
Logging configuration for the Linear Regression Guide.

This module provides a centralized logging setup with structured logging,
log rotation, and different log levels for various components.
"""

import logging
import logging.handlers
import os
import sys
import traceback
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log directory and file paths
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"
PERFORMANCE_LOG_FILE = LOG_DIR / "performance.log"

# Log levels
DEFAULT_LOG_LEVEL = logging.INFO
CONSOLE_LOG_LEVEL = logging.WARNING
FILE_LOG_LEVEL = logging.DEBUG

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - " "%(funcName)s:%(lineno)d - %(message)s"
CONSOLE_FORMAT = "%(levelname)s - %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log rotation settings
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Keep 5 backup files


def setup_logging(
    log_level: int = DEFAULT_LOG_LEVEL,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
) -> logging.Logger:
    """
    Set up application-wide logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to log to files
        enable_console_logging: Whether to log to console

    Returns:
        Configured root logger
    """
    # Create logs directory if it doesn't exist
    if enable_file_logging:
        LOG_DIR.mkdir(exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_formatter = logging.Formatter(CONSOLE_FORMAT)

    # Console handler (for warnings and errors)
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(CONSOLE_LOG_LEVEL)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation (for all logs)
    if enable_file_logging:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(FILE_LOG_LEVEL)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)

            # Separate error log file
            error_handler = logging.handlers.RotatingFileHandler(
                ERROR_LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_handler)

            # Performance log file
            perf_handler = logging.handlers.RotatingFileHandler(
                PERFORMANCE_LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(detailed_formatter)
            # Add filter to only log performance-related messages
            perf_handler.addFilter(lambda record: "performance" in record.name.lower())
            root_logger.addHandler(perf_handler)

        except (OSError, IOError) as e:
            # If file logging fails, at least log to console
            if enable_console_logging:
                root_logger.warning(f"Could not set up file logging: {e}")
            else:
                print(f"Warning: Could not set up logging: {e}", file=sys.stderr)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log a function call with its parameters.

    Args:
        logger: Logger instance
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params})")


def log_performance(logger: logging.Logger, operation: str, duration: float):
    """
    Log performance metrics.

    Args:
        logger: Logger instance
        operation: Description of the operation
        duration: Duration in seconds
    """
    logger.info(f"Performance: {operation} took {duration:.3f}s")


def log_data_info(logger: logging.Logger, data_name: str, shape: tuple, **info):
    """
    Log data generation or loading information.

    Args:
        logger: Logger instance
        data_name: Name of the dataset
        shape: Shape of the data (rows, columns)
        **info: Additional information about the data
    """
    info_str = ", ".join(f"{k}={v}" for k, v in info.items())
    logger.info(f"Data '{data_name}': shape={shape}, {info_str}")


def log_error_with_context(
    logger: logging.Logger, 
    error: Exception, 
    context: str, 
    error_id: Optional[str] = None,
    **details
) -> str:
    """
    Log an error with contextual information and generate an error ID.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Description of what was being done when error occurred
        error_id: Optional error ID (will be generated if not provided)
        **details: Additional details about the error context

    Returns:
        Error ID for tracking
    """
    if error_id is None:
        error_id = str(uuid.uuid4())[:8]
    
    # Build structured error message
    error_type = type(error).__name__
    error_message = str(error)
    
    # Extract error code if it's a DomainError
    error_code = getattr(error, 'code', 'UNKNOWN_ERROR')
    
    # Format details
    details_str = ", ".join(f"{k}={v}" for k, v in details.items()) if details else "None"
    
    # Get stack trace
    tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
    tb_str = "".join(tb_lines)
    
    # Log structured error
    logger.error(
        f"[ERROR_ID={error_id}] Error in {context} | "
        f"Type: {error_type} | Code: {error_code} | "
        f"Message: {error_message} | Details: {details_str}",
        exc_info=True,
        extra={
            'error_id': error_id,
            'error_type': error_type,
            'error_code': error_code,
            'context': context,
            'details': details,
            'traceback': tb_str
        }
    )
    
    return error_id


def log_domain_error(
    logger: logging.Logger,
    error: Exception,
    context: str,
    **details
) -> str:
    """
    Log a DomainError with enhanced context.

    Args:
        logger: Logger instance
        error: DomainError that occurred
        context: Description of what was being done when error occurred
        **details: Additional details about the error context

    Returns:
        Error ID for tracking
    """
    error_code = getattr(error, 'code', 'DOMAIN_ERROR')
    return log_error_with_context(
        logger=logger,
        error=error,
        context=context,
        error_id=None,  # Auto-generate
        error_code=error_code,
        **details
    )


def log_api_error(
    logger: logging.Logger,
    error: Exception,
    endpoint: str,
    method: str = "GET",
    request_data: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    **details
) -> str:
    """
    Log an API error with request context.

    Args:
        logger: Logger instance
        error: Exception that occurred
        endpoint: API endpoint where error occurred
        method: HTTP method (GET, POST, etc.)
        request_data: Request data/parameters
        user_id: Optional user identifier
        **details: Additional details

    Returns:
        Error ID for tracking
    """
    api_details = {
        'endpoint': endpoint,
        'method': method,
        'request_data': request_data or {},
        'user_id': user_id,
        **details
    }
    return log_error_with_context(
        logger=logger,
        error=error,
        context=f"API {method} {endpoint}",
        **api_details
    )


def log_service_error(
    logger: logging.Logger,
    error: Exception,
    service_name: str,
    operation: str,
    input_params: Optional[Dict[str, Any]] = None,
    **details
) -> str:
    """
    Log a service layer error with operation context.

    Args:
        logger: Logger instance
        error: Exception that occurred
        service_name: Name of the service
        operation: Operation being performed
        input_params: Input parameters to the operation
        **details: Additional details

    Returns:
        Error ID for tracking
    """
    service_details = {
        'service': service_name,
        'operation': operation,
        'input_params': input_params or {},
        **details
    }
    return log_error_with_context(
        logger=logger,
        error=error,
        context=f"{service_name}.{operation}",
        **service_details
    )


def error_handler(logger: logging.Logger, context: str = None):
    """
    Decorator for automatic error logging.

    Args:
        logger: Logger instance
        context: Context description (defaults to function name)

    Usage:
        @error_handler(logger, "data_processing")
        def process_data():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_context = context or f"{func.__module__}.{func.__name__}"
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Extract function arguments (sanitize for logging)
                func_args = {}
                if args:
                    func_args['args_count'] = len(args)
                if kwargs:
                    # Only log non-sensitive kwargs
                    safe_kwargs = {k: v for k, v in kwargs.items() 
                                 if not any(sensitive in k.lower() 
                                          for sensitive in ['password', 'token', 'key', 'secret'])}
                    func_args['kwargs'] = safe_kwargs
                
                error_id = log_error_with_context(
                    logger=logger,
                    error=e,
                    context=func_context,
                    function=func.__name__,
                    **func_args
                )
                # Re-raise the exception
                raise
        return wrapper
    return decorator


def cleanup_old_logs(days: int = 30):
    """
    Clean up log files older than specified days.

    Args:
        days: Number of days to keep logs (default: 30)
    """
    if not LOG_DIR.exists():
        return

    cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)

    for log_file in LOG_DIR.glob("*.log*"):
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                print(f"Deleted old log file: {log_file}")
        except (OSError, IOError) as e:
            print(f"Could not delete log file {log_file}: {e}")


# ============================================================================
# ERROR TRACKING AND MONITORING
# ============================================================================

class ErrorTracker:
    """Track errors for monitoring and analysis."""
    
    def __init__(self):
        self.errors: Dict[str, Dict[str, Any]] = {}
        self.error_counts: Dict[str, int] = {}
    
    def record_error(
        self, 
        error_id: str, 
        error_type: str, 
        error_code: str,
        context: str,
        **details
    ):
        """Record an error for tracking."""
        self.errors[error_id] = {
            'error_type': error_type,
            'error_code': error_code,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            **details
        }
        
        # Count errors by type
        error_key = f"{error_type}:{error_code}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recorded errors."""
        return {
            'total_errors': len(self.errors),
            'error_counts': self.error_counts,
            'recent_errors': list(self.errors.values())[-10:]  # Last 10 errors
        }


# Global error tracker instance
_error_tracker = ErrorTracker()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    return _error_tracker


# ============================================================================
# INITIALIZE LOGGING ON MODULE IMPORT
# ============================================================================

# Set up logging when module is imported
# This can be disabled by setting environment variable DISABLE_LOGGING=1
if not os.getenv("DISABLE_LOGGING"):
    setup_logging()
