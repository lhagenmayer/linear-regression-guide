"""
Domain Entities.
Objects with identity and lifecycle.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
import uuid

from .value_objects import RegressionParameters, RegressionMetrics, DatasetMetadata, DataPoint

@dataclass
class RegressionModel:
    """
    Core Domain Entity representing a trained regression model.
    Has identity (id) and holds the state of the analysis.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    dataset_metadata: Optional[DatasetMetadata] = None
    
    # State (initially None until trained)
    parameters: Optional[RegressionParameters] = None
    metrics: Optional[RegressionMetrics] = None
    
    # Store predictions/residuals? 
    # For large datasets, we might separate this. 
    # But for this educational app (n<1000), it's fine in memory.
    residuals: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    
    def is_trained(self) -> bool:
        """Check if model has been successfully trained."""
        return self.parameters is not None and self.metrics is not None

    def get_equation_string(self) -> str:
        """Domain logic to generate equation string representation."""
        if not self.is_trained():
            return "Not trained"
            
        parts = [f"{self.parameters.intercept:.2f}"]
        for name, coef in self.parameters.coefficients.items():
            sign = "+" if coef >= 0 else "-"
            parts.append(f"{sign} {abs(coef):.2f}*{name}")
            
        return "y = " + " ".join(parts)
