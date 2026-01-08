"""
Vereinheitlichte Datensatz-Registry - Clean Architecture.

Bietet eine zentrale Schnittstelle für den Zugriff auf alle Datensätze über alle Analysetypen hinweg.
Dies ermöglicht es Studierenden, denselben Datensatz durch progressiv komplexere Methoden zu untersuchen.

Architektur:
    DatasetRegistry (Infrastructure)
        ├── get_for_simple_regression()
        ├── get_for_multiple_regression()
        ├── get_for_classification()
        └── list_all() → Metadaten für die UI

Nutzung:
    registry = DatasetRegistry()
    
    # Derselbe Datensatz, verschiedene Analyseebenen
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
    """Arten der Analyse, die ein Datensatz unterstützt."""
    SIMPLE_REGRESSION = "simple_regression"
    MULTIPLE_REGRESSION = "multiple_regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


@dataclass
class DatasetMeta:
    """Metadaten über einen Datensatz und dessen Fähigkeiten."""
    name: str
    display_name: str
    description: str
    icon: str
    capabilities: Set[AnalysisType]
    feature_count: int
    typical_n: int
    domain: str  # z.B. "business", "science", "education"
    
    def supports(self, analysis_type: AnalysisType) -> bool:
        """Prüft, ob der Datensatz den gegebenen Analysetyp unterstützt."""
        return analysis_type in self.capabilities


# =============================================================================
# DATASET REGISTRY
# =============================================================================

class DatasetRegistry:
    """
    Vereinheitlichtes Interface für alle Datensätze über alle Analysetypen hinweg.
    
    Dies ist der empfohlene Einstiegspunkt für den Datenzugriff in der Anwendung.
    Es stellt sicher, dass Datensätze konsistent über die gesamte "Learning Journey" verfügbar sind.
    
    Beispiel:
        registry = DatasetRegistry()
        
        # Ein Studierender erkundet die Elektronikmarkt-Daten durch die gesamte Reise
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
        """Listet alle verfügbaren Datensätze mit ihren Metadaten auf."""
        return list(self._metadata.values())
    
    def list_by_capability(self, analysis_type: AnalysisType) -> List[DatasetMeta]:
        """Listet Datensätze auf, die einen spezifischen Analysetyp unterstützen."""
        return [m for m in self._metadata.values() if m.supports(analysis_type)]
    
    def get_metadata(self, name: str) -> Optional[DatasetMeta]:
        """Gibt Metadaten für einen spezifischen Datensatz zurück."""
        return self._metadata.get(name)
    
    def get_for_simple_regression(
        self, 
        name: str, 
        n: int = 50, 
        seed: Optional[int] = None
    ) -> DataResult:
        """
        Ruft einen Datensatz für die einfache lineare Regression ab (ein X, ein Y).
        
        Args:
            name: Name des Datensatzes
            n: Anzahl der Beobachtungen
            seed: Zufalls-Seed (nutzt Registry-Standard, falls nicht angegeben)
            
        Returns:
            DataResult mit x, y Arrays und Metadaten
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
        Ruft einen Datensatz für die multiple Regression ab (X1, X2, Y).
        
        Args:
            name: Name des Datensatzes
            n: Anzahl der Beobachtungen
            seed: Zufalls-Seed
            
        Returns:
            MultipleRegressionDataResult mit x1, x2, y Arrays
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
        Ruft einen Datensatz für Klassifikationsaufgaben ab (KNN, Logistische Regression).
        
        Args:
            name: Name des Datensatzes
            n: Anzahl der Stichproben
            binary: Falls True, wird in binäre Klassifikation konvertiert
            n_classes: Anzahl der Klassen für Multi-Class (falls nicht binär)
            seed: Zufalls-Seed
            
        Returns:
            ClassificationDataResult mit X-Matrix, y-Array und Metadaten
        """
        seed = seed or self._seed
        
        # Native Klassifikations-Datensätze
        if name in ["fruits", "digits"]:
            return self._fetcher.get_classification(name, n=n, seed=seed)
        
        # Konvertierung von Regressions-Datensätzen in Klassifikation
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
        """Konvertiert einen Regressions-Datensatz in eine Klassifikations-Aufgabe."""
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
        Gibt alle anwendbaren Analysetypen für einen Datensatz zurück.
        
        Ideal zur Demonstration der Progression von einfach zu komplex.
        
        Returns:
            Dict mit den Schlüsseln: 'simple', 'multiple', 'binary', 'multiclass'
            Enthält die Daten, falls anwendbar, andernfalls None.
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
