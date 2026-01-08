"""
Data Transfer Objects (DTOs) for Application Layer.
Used to transfer data between API/CLI and Use Cases.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class RegressionRequestDTO:
    """Request DTO for running a regression."""
    dataset_id: str
    n_observations: int
    noise_level: float
    seed: int
    regression_type: str = "simple"  # 'simple' or 'multiple'
    
    # Optional overrides for synthetic data
    true_intercept: Optional[float] = None
    true_slope: Optional[float] = None

@dataclass
class RegressionResponseDTO:
    """Response DTO containing results and data."""
    model_id: str
    success: bool
    
    # Result Data
    coefficients: Dict[str, float]
    metrics: Dict[str, float]
    
    # Raw Data (for plotting by frontend)
    x_data: List[float]  # Or List[List[float]] for multiple
    y_data: List[float]
    residuals: List[float]
    predictions: List[float]
    
    # Metadata
    x_label: str
    y_label: str
    title: str
    description: str
    
    extra: Dict[str, Any] = field(default_factory=dict)
