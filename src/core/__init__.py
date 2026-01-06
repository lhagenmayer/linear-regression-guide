"""
Core business logic package for the Linear Regression Guide.

This package contains statistical computations, data services, and core functionality.
"""

# Import from domain layer
from .domain.services import RegressionAnalysisService

__all__ = ['RegressionAnalysisService']