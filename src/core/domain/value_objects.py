"""
Value Objects for Regression Domain.
Pure Python, NO external dependencies (no numpy/pandas).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass(frozen=True)
class RegressionParameters:
    """Immutable parameters of a regression model."""
    intercept: float
    coefficients: Dict[str, float]  # variable_name -> value (slope)
    
    @property
    def slope(self) -> Optional[float]:
        """Helper for simple regression (single slope)."""
        if len(self.coefficients) == 1:
            return next(iter(self.coefficients.values()))
        return None

@dataclass(frozen=True)
class RegressionMetrics:
    """Immutable quality metrics of a regression model."""
    r_squared: float
    r_squared_adj: float
    mse: float
    rmse: float
    f_statistic: Optional[float] = None
    p_value: Optional[float] = None

@dataclass(frozen=True)
class DataPoint:
    """Single data point (observation)."""
    x: Dict[str, float]
    y: float

@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata about a dataset."""
    id: str
    name: str
    description: str
    source: str
    variables: List[str]
    n_observations: int
    is_time_series: bool = False
