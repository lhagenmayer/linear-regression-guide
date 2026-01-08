"""
Data Transfer Objects (DTOs) für den Application-Layer.
Typsicherer Datentransport zwischen API/CLI und Use Cases.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum

# Import von Enums aus dem Domain-Layer
from ..domain.value_objects import RegressionType, ModelQuality


@dataclass(frozen=True)
class RegressionRequestDTO:
    """
    Unveränderliches Request-Objekt zum Starten einer Regressionsanalyse.
    Nutzt Enums für Typsicherheit.
    """
    dataset_id: str
    n_observations: int
    noise_level: float
    seed: int
    regression_type: RegressionType = RegressionType.SIMPLE
    
    # Optionale Overrides für synthetische Daten (Steigung/Achsenabschnitt)
    true_intercept: Optional[float] = None
    true_slope: Optional[float] = None
    
    def __post_init__(self):
        """Validierung der Request-Parameter."""
        if self.n_observations < 2:
            raise ValueError(f"n_observations muss >= 2 sein, erhalten wurde {self.n_observations}")
        if self.noise_level < 0:
            raise ValueError(f"noise_level darf nicht negativ sein, erhalten wurde {self.noise_level}")


@dataclass(frozen=True)
class RegressionResponseDTO:
    """
    Unveränderliches Response-Objekt, das Ergebnisse und Rohdaten enthält.
    
    Architektur-Hinweis: frozen=True stellt Unveränderlichkeit sicher, 
    aber Listen sind intern noch mutierbar. Für echte Immutability nutzen wir Tuples 
    für x_data, y_data etc.
    """
    model_id: str
    success: bool
    
    # Ergebnisdaten (als Dicts für einfache JSON-Serialisierung)
    coefficients: Dict[str, float]
    metrics: Dict[str, float]
    
    # Rohdaten (für die Visualisierung im Frontend)
    x_data: tuple  # Tuple für strikte Unveränderlichkeit
    y_data: tuple
    residuals: tuple
    predictions: tuple
    
    # Metadaten zur Beschriftung
    x_label: str
    y_label: str
    title: str
    description: str
    
    # Qualitätsbewertung
    quality: Optional[ModelQuality] = None
    is_significant: bool = False
    
    # Erweiterbarkeit für zusätzliche Informationen
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def r_squared(self) -> Optional[float]:
        """Bequemer Zugriff auf den R²-Wert."""
        return self.metrics.get("r_squared")
    
    @property
    def slope(self) -> Optional[float]:
        """Bequemer Zugriff auf die Steigung (nur bei einfacher Regression)."""
        if "x" in self.coefficients:
            return self.coefficients["x"]
        return None


@dataclass(frozen=True)
class ErrorDTO:
    """Standardisierte Fehlerrückgabe."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


# Typ-Alias für Resultate, die entweder Daten oder einen Fehler enthalten
ResponseResult = Union[RegressionResponseDTO, ErrorDTO]


@dataclass(frozen=True)
class ClassificationRequestDTO:
    """DTO für Klassifikations-Anfragen (Logistic / KNN)."""
    dataset_id: str
    n_observations: int
    noise_level: float
    seed: int
    method: str  # "logistic" oder "knn"
    k_neighbors: int = 3
    stratify: bool = False
    train_size: float = 0.8


@dataclass(frozen=True)
class ClassificationResponseDTO:
    """DTO für Klassifikations-Ergebnisse."""
    success: bool
    method: str
    classes: tuple
    
    # Metriken für Trainings- und Testdaten
    metrics: Dict[str, Any]      # Training
    test_metrics: Dict[str, Any] # Test
    
    # Modell-Parameter (z.B. Koeffizienten)
    parameters: Dict[str, Any]
    
    # Daten für die grafische Darstellung
    X_data: tuple # Tuple von Tuples (für Plotly 3D)
    y_data: tuple
    predictions: tuple
    probabilities: tuple
    
    # Metadaten zur Benennung
    feature_names: tuple
    target_names: tuple
    dataset_name: str
    dataset_description: str
    
    extra: Dict[str, Any] = field(default_factory=dict)
