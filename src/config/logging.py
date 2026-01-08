import logging
import logging.config
import os

def configure_logging(default_level=logging.INFO):
    """
    Configure structured logging for the application.
    """
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "INFO",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "json",
                "level": "INFO",
                "filename": "app.log",
                "mode": "a",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": default_level,
                "propagate": True,
            },
            "src": {
                "handlers": ["console", "file"],
                "level": default_level,
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(log_config)
    logging.info("Logging configured successfully")

# Auto-configure on import if not already configured
if not logging.getLogger().handlers:
    configure_logging()
