"""
Domain Interfaces (Ports).
Using Protocols for structural subtyping.
Infrastructure implementations will satisfy these protocols.
"""
from typing import Protocol, List, Dict, Any, Optional
from .entities import RegressionModel, DatasetMetadata
from .value_objects import RegressionParameters, RegressionMetrics

class IDataProvider(Protocol):
    """Interface for fetching data."""
    
    def get_dataset(self, dataset_id: str, n: int, **kwargs) -> Dict[str, Any]:
        """
        Fetch raw data. 
        Returns dictionary (e.g. {'x': [...], 'y': [...]})
        We use Dict for flexibility in raw layer, but Application layer converts to DTOs.
        """
        ...
        
    def list_datasets(self) -> List[DatasetMetadata]:
        """List available datasets."""
        ...

class IRegressionService(Protocol):
    """Interface for performing regression calculations."""
    
    def train_simple(self, x: List[float], y: List[float]) -> RegressionModel:
        """Train simple regression model."""
        ...
        
    def train_multiple(self, x: List[List[float]], y: List[float], variable_names: List[str]) -> RegressionModel:
        """Train multiple regression model."""
        ...
