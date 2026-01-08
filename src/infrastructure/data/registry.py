"""
Unified Dataset Registry - Clean Architecture.

Provides a single interface for accessing all datasets across all analysis types,
enabling students to explore the same data through progressively complex methods.

Architecture:
    DatasetRegistry (Infrastructure)
        ├── get_for_simple_regression()
        ├── get_for_multiple_regression()
        ├── get_for_classification()
        └── list_all() → metadata for UI

Usage:
    registry = DatasetRegistry()
    
    # Same dataset, different analysis levels
    simple = registry.get_for_simple_regression("electronics")
    multiple = registry.get_for_multiple_regression("electronics")
    binary = registry.get_for_classification("electronics", binary=True)
    multi = registry.get_for_classification("electronics", n_classes=4)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any
import numpy as np

from .generators import (
    DataFetcher, 
    DataResult, 
    MultipleRegressionDataResult, 
    ClassificationDataResult
)


class AnalysisType(Enum):
    """Types of analysis a dataset supports."""
    SIMPLE_REGRESSION = "simple_regression"
    MULTIPLE_REGRESSION = "multiple_regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


@dataclass
class DatasetMeta:
    """Metadata about a dataset and its capabilities."""
    name: str
    display_name: str
    description: str
    icon: str
    capabilities: Set[AnalysisType]
    feature_count: int
    typical_n: int
    domain: str  # e.g., "business", "science", "education"
    
    def supports(self, analysis_type: AnalysisType) -> bool:
        """Check if dataset supports given analysis type."""
        return analysis_type in self.capabilities


# =============================================================================
# DATASET REGISTRY
# =============================================================================

class DatasetRegistry:
    """
    Unified interface for all datasets across all analysis types.
    
    This is the recommended entry point for accessing data in the application.
    It ensures datasets are available consistently across the learning journey.
    
    Example:
        registry = DatasetRegistry()
        
        # Student explores electronics data through the full journey
        for level in ["simple", "multiple", "binary", "multiclass"]:
            data = registry.get("electronics", level)
            print(f"{level}: {data}")
    """
    
    def __init__(self, seed: int = 42):
        self._fetcher = DataFetcher()
        self._seed = seed
        self._metadata = self._build_metadata()
    
    def _build_metadata(self) -> Dict[str, DatasetMeta]:
        """Build metadata for all available datasets."""
        return {
            # BUSINESS DATASETS
            "electronics": DatasetMeta(
                name="electronics",
                display_name="🏪 Elektronikmarkt",
                description="Sales vs store size analysis. Perfect for starting with linear regression.",
                icon="🏪",
                capabilities={
                    AnalysisType.SIMPLE_REGRESSION,
                    AnalysisType.MULTIPLE_REGRESSION,
                    AnalysisType.BINARY_CLASSIFICATION,
                    AnalysisType.MULTICLASS_CLASSIFICATION,
                },
                feature_count=3,
                typical_n=50,
                domain="business"
            ),
            "advertising": DatasetMeta(
                name="advertising",
                display_name="📢 Werbekampagne",
                description="Advertising spend vs sales. Classic marketing analytics.",
                icon="📢",
                capabilities={
                    AnalysisType.SIMPLE_REGRESSION,
                    AnalysisType.MULTIPLE_REGRESSION,
                    AnalysisType.BINARY_CLASSIFICATION,
                    AnalysisType.MULTICLASS_CLASSIFICATION,
                },
                feature_count=2,
                typical_n=50,
                domain="business"
            ),
            "houses": DatasetMeta(
                name="houses",
                display_name="🏠 Immobilien",
                description="House pricing with area and amenities. Great for multiple regression.",
                icon="🏠",
                capabilities={
                    AnalysisType.SIMPLE_REGRESSION,
                    AnalysisType.MULTIPLE_REGRESSION,
                    AnalysisType.BINARY_CLASSIFICATION,
                    AnalysisType.MULTICLASS_CLASSIFICATION,
                },
                feature_count=2,
                typical_n=75,
                domain="business"
            ),
            "cities": DatasetMeta(
                name="cities",
                display_name="🌆 City Sales",
                description="Multi-city sales study with price and advertising factors.",
                icon="🌆",
                capabilities={
                    AnalysisType.MULTIPLE_REGRESSION,
                    AnalysisType.BINARY_CLASSIFICATION,
                    AnalysisType.MULTICLASS_CLASSIFICATION,
                },
                feature_count=2,
                typical_n=75,
                domain="business"
            ),
            
            # SWISS DATASETS
            "cantons": DatasetMeta(
                name="cantons",
                display_name="🇨🇭 Kantone",
                description="Swiss canton socioeconomic data. GDP prediction.",
                icon="🇨🇭",
                capabilities={
                    AnalysisType.SIMPLE_REGRESSION,
                    AnalysisType.MULTIPLE_REGRESSION,
                    AnalysisType.BINARY_CLASSIFICATION,
                    AnalysisType.MULTICLASS_CLASSIFICATION,
                },
                feature_count=3,
                typical_n=26,
                domain="economics"
            ),
            "weather": DatasetMeta(
                name="weather",
                display_name="🌤️ Wetter",
                description="Swiss weather stations: altitude, sunshine, temperature.",
                icon="🌤️",
                capabilities={
                    AnalysisType.SIMPLE_REGRESSION,
                    AnalysisType.MULTIPLE_REGRESSION,
                },
                feature_count=2,
                typical_n=50,
                domain="science"
            ),
            
            # SCIENCE DATASETS
            "temperature": DatasetMeta(
                name="temperature",
                display_name="🍦 Eisverkauf",
                description="Temperature vs ice cream sales. Simple causal relationship.",
                icon="🍦",
                capabilities={
                    AnalysisType.SIMPLE_REGRESSION,
                    AnalysisType.BINARY_CLASSIFICATION,
                },
                feature_count=1,
                typical_n=50,
                domain="science"
            ),
            
            # ML CASE STUDIES (Classification-native)
            "fruits": DatasetMeta(
                name="fruits",
                display_name="🍎 Fruit Classification",
                description="Professor's KNN case study. 4 fruit types by physical properties.",
                icon="🍎",
                capabilities={
                    AnalysisType.MULTICLASS_CLASSIFICATION,
                },
                feature_count=4,
                typical_n=59,
                domain="education"
            ),
            "digits": DatasetMeta(
                name="digits",
                display_name="🔢 Handwritten Digits",
                description="8x8 pixel digit images. Classic ML benchmark.",
                icon="🔢",
                capabilities={
                    AnalysisType.MULTICLASS_CLASSIFICATION,
                },
                feature_count=64,
                typical_n=100,
                domain="education"
            ),
        }
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def list_all(self) -> List[DatasetMeta]:
        """List all available datasets with their metadata."""
        return list(self._metadata.values())
    
    def list_by_capability(self, analysis_type: AnalysisType) -> List[DatasetMeta]:
        """List datasets that support a specific analysis type."""
        return [m for m in self._metadata.values() if m.supports(analysis_type)]
    
    def get_metadata(self, name: str) -> Optional[DatasetMeta]:
        """Get metadata for a specific dataset."""
        return self._metadata.get(name)
    
    def get_for_simple_regression(
        self, 
        name: str, 
        n: int = 50, 
        seed: Optional[int] = None
    ) -> DataResult:
        """
        Get dataset for simple linear regression (one X, one Y).
        
        Args:
            name: Dataset name
            n: Number of observations
            seed: Random seed (uses registry default if not specified)
            
        Returns:
            DataResult with x, y arrays and metadata
        """
        seed = seed or self._seed
        return self._fetcher.get_simple(name, n=n, seed=seed)
    
    def get_for_multiple_regression(
        self, 
        name: str, 
        n: int = 75, 
        seed: Optional[int] = None
    ) -> MultipleRegressionDataResult:
        """
        Get dataset for multiple regression (X1, X2, Y).
        
        Args:
            name: Dataset name
            n: Number of observations
            seed: Random seed
            
        Returns:
            MultipleRegressionDataResult with x1, x2, y arrays
        """
        seed = seed or self._seed
        return self._fetcher.get_multiple(name, n=n, seed=seed)
    
    def get_for_classification(
        self, 
        name: str, 
        n: int = 50,
        binary: bool = False,
        n_classes: int = 4,
        seed: Optional[int] = None
    ) -> ClassificationDataResult:
        """
        Get dataset for classification (KNN, Logistic Regression).
        
        Args:
            name: Dataset name
            n: Number of samples
            binary: If True, convert to binary classification
            n_classes: Number of classes for multi-class (if not binary)
            seed: Random seed
            
        Returns:
            ClassificationDataResult with X matrix, y array, metadata
        """
        seed = seed or self._seed
        
        # Native classification datasets
        if name in ["fruits", "digits"]:
            return self._fetcher.get_classification(name, n=n, seed=seed)
        
        # Convert regression datasets to classification
        return self._convert_to_classification(name, n, binary, n_classes, seed)
    
    # =========================================================================
    # CONVERSION METHODS (Regression → Classification)
    # =========================================================================
    
    def _convert_to_classification(
        self, 
        name: str, 
        n: int,
        binary: bool,
        n_classes: int,
        seed: int
    ) -> ClassificationDataResult:
        """Convert regression dataset to classification."""
        # Get regression data
        try:
            multi_data = self._fetcher.get_multiple(name, n=n, seed=seed)
            X = np.column_stack([multi_data.x1, multi_data.x2])
            y_continuous = multi_data.y
            feature_names = [multi_data.x1_label, multi_data.x2_label]
        except:
            simple_data = self._fetcher.get_simple(name, n=n, seed=seed)
            X = simple_data.x.reshape(-1, 1)
            y_continuous = simple_data.y
            feature_names = [simple_data.x_label]
        
        # Convert to classes
        if binary:
            y = (y_continuous > np.median(y_continuous)).astype(int)
            target_names = ["low", "high"]
        else:
            # Multi-class: quantile-based bins
            percentiles = np.linspace(0, 100, n_classes + 1)
            bins = np.percentile(y_continuous, percentiles)
            y = np.digitize(y_continuous, bins[1:-1])
            target_names = [f"tier_{i+1}" for i in range(n_classes)]
        
        meta = self._metadata.get(name)
        return ClassificationDataResult(
            X=X,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            context_title=meta.display_name if meta else name,
            context_description=f"Classification from {name} (binary={binary}, classes={len(target_names)})",
            extra={"source_dataset": name, "binary": binary}
        )
    
    # =========================================================================
    # LEARNING JOURNEY HELPER
    # =========================================================================
    
    def get_learning_journey(
        self, 
        name: str, 
        n: int = 50,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get all applicable analysis types for a dataset.
        
        Perfect for demonstrating progression from simple to complex.
        
        Returns:
            Dict with keys: 'simple', 'multiple', 'binary', 'multiclass'
            Each contains the data if applicable, None otherwise.
        """
        seed = seed or self._seed
        meta = self._metadata.get(name)
        
        if not meta:
            return {}
        
        journey = {}
        
        if meta.supports(AnalysisType.SIMPLE_REGRESSION):
            journey['simple_regression'] = self.get_for_simple_regression(name, n, seed)
        
        if meta.supports(AnalysisType.MULTIPLE_REGRESSION):
            journey['multiple_regression'] = self.get_for_multiple_regression(name, n, seed)
        
        if meta.supports(AnalysisType.BINARY_CLASSIFICATION):
            journey['binary_classification'] = self.get_for_classification(name, n, binary=True, seed=seed)
        
        if meta.supports(AnalysisType.MULTICLASS_CLASSIFICATION):
            journey['multiclass_classification'] = self.get_for_classification(name, n, binary=False, n_classes=4, seed=seed)
        
        return journey
