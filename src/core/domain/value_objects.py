"""
Domain Value Objects.
Unveränderliche, validierte Datenstrukturen mit Geschäftslogik.
Pure Python - KEINE externen Abhängigkeiten.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto


class DomainError(Exception):
    """
    Basis-Exception für Domain-Fehler.
    
    Wird verwendet, um geschäftsspezifische Fehler zu signalisieren,
    die von der API-Schicht behandelt werden können.
    """
    def __init__(self, message: str, code: str = "DOMAIN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class RegressionType(Enum):
    """Typsichere Aufzählung der Regressionsarten."""
    SIMPLE = auto()
    MULTIPLE = auto()


class ModelQuality(Enum):
    """Klassifizierung der Modellqualität basierend auf R²."""
    POOR = auto()      # R² < 0.3 (Schlecht)
    FAIR = auto()      # 0.3 <= R² < 0.5 (Mäßig)
    GOOD = auto()      # 0.5 <= R² < 0.7 (Gut)
    EXCELLENT = auto() # R² >= 0.7 (Exzellent)


# Basisklasse für das Result-Pattern zur Fehlerbehandlung
@dataclass(frozen=True)
class Result:
    """Basisklasse für alle Result-Objekte."""
    pass


@dataclass(frozen=True)
class Success(Result):
    """Repräsentiert eine erfolgreiche Operation mit einem Wert."""
    value: Any


@dataclass(frozen=True)  
class Failure(Result):
    """Repräsentiert eine fehlgeschlagene Operation mit Fehlermeldung."""
    error: str
    code: str = "UNKNOWN"
    
    
@dataclass(frozen=True)
class RegressionParameters:
    """Unveränderliche Parameter eines Regressionsmodells."""
    intercept: float
    coefficients: Dict[str, float]
    
    def __post_init__(self):
        """Validierung: Koeffizienten dürfen nicht leer sein."""
        if not self.coefficients:
            raise ValueError("coefficients darf nicht leer sein")
    
    @property
    def slope(self) -> Optional[float]:
        """Helper für einfache Regression (einzelne Steigung)."""
        if len(self.coefficients) == 1:
            return next(iter(self.coefficients.values()))
        return None
    
    @property
    def variable_names(self) -> List[str]:
        """Gibt eine Liste der Variablennamen zurück."""
        return list(self.coefficients.keys())


@dataclass(frozen=True)
class RegressionResult(Result):
    """Container für die Ergebnisse einer Regressionsberechnung."""
    parameters: RegressionParameters
    metrics: RegressionMetrics
    predictions: Any  # Typ-Hinweis: np.ndarray (in Domain als Any markiert für Purity)
    residuals: Any
    model_equation: str


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metriken zur Bewertung der Klassifikationsleistung."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Any  # Format: [[TN, FP], [FN, TP]]
    auc: Optional[float] = None
    
    def __post_init__(self):
        """Validierung: Metriken müssen im Bereich [0, 1] liegen."""
        for name, value in [
            ("accuracy", self.accuracy),
            ("precision", self.precision), 
            ("recall", self.recall),
            ("f1_score", self.f1_score)
        ]:
            # Kleine Toleranz für Fließkomma-Ungenauigkeiten
            if not (-0.0001 <= value <= 1.0001):
                raise ValueError(f"{name} muss zwischen 0 und 1 liegen, erhalten wurde {value}")

@dataclass(frozen=True)
class ClassificationResult(Result):
    """Container für Klassifikationsergebnisse."""
    classes: List[Any]
    predictions: Any
    probabilities: Any
    metrics: ClassificationMetrics      # Metriken für Trainingsdaten
    model_params: Dict[str, Any]       # z.B. Koeffizienten oder k-Nachbarn
    test_metrics: Optional[ClassificationMetrics] = None # Metriken für Testdaten
    is_success: bool = True


@dataclass(frozen=True)
class RegressionMetrics:
    """Unveränderliche Qualitätsmetriken eines Regressionsmodells."""
    r_squared: float
    r_squared_adj: float
    mse: float
    rmse: float
    f_statistic: Optional[float] = None
    p_value: Optional[float] = None
    
    def __post_init__(self):
        """Validierung der Wertebereiche."""
        if not (-0.0001 <= self.r_squared <= 1.0001):
            raise ValueError(f"r_squared muss zwischen 0 und 1 liegen, erhalten wurde {self.r_squared}")
        if self.mse < 0:
            raise ValueError(f"mse darf nicht negativ sein, erhalten wurde {self.mse}")
        if self.rmse < 0:
            raise ValueError(f"rmse darf nicht negativ sein, erhalten wurde {self.rmse}")
    
    @property
    def quality(self) -> ModelQuality:
        """Klassifiziert die Modellqualität basierend auf dem R²-Wert."""
        if self.r_squared < 0.3:
            return ModelQuality.POOR
        elif self.r_squared < 0.5:
            return ModelQuality.FAIR
        elif self.r_squared < 0.7:
            return ModelQuality.GOOD
        else:
            return ModelQuality.EXCELLENT
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Prüft, ob das Modell statistisch signifikant ist (basierend auf p_value)."""
        if self.p_value is None:
            return False
        return self.p_value < alpha


@dataclass(frozen=True)
class DataPoint:
    """Ein einzelner Datenpunkt (Beobachtung)."""
    x: Dict[str, float]
    y: float
    
    def __post_init__(self):
        """Validierung: Features (x) dürfen nicht leer sein."""
        if not self.x:
            raise ValueError("x darf nicht leer sein")


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadaten über einen Datensatz."""
    id: str
    name: str
    description: str
    source: str
    variables: tuple  # Tuple für Unveränderlichkeit
    n_observations: int
    is_time_series: bool = False
    
    def __post_init__(self):
        """Validierung der Metadaten."""
        if not self.id:
            raise ValueError("id darf nicht leer sein")
        if self.n_observations < 0:
            raise ValueError(f"n_observations darf nicht negativ sein, erhalten wurde {self.n_observations}")


@dataclass(frozen=True)
class SplitConfig:
    """Konfiguration für die Datenaufteilung (Train/Test Split)."""
    train_size: float
    stratify: bool
    seed: int
    
    def __post_init__(self):
        if not (0 < self.train_size < 1):
            raise ValueError(f"train_size muss zwischen 0 und 1 liegen, erhalten wurde {self.train_size}")


@dataclass(frozen=True)
class SplitStats:
    """Statistiken über eine Datenaufteilung."""
    train_count: int
    test_count: int
    train_distribution: Dict[Any, int]
    test_distribution: Dict[Any, int]
    
    @property
    def total_count(self) -> int:
        """Gesamtanzahl der Datenpunkte."""
        return self.train_count + self.test_count
