"""
Domain Interfaces (Ports).
Definition der Schnittstellen mittels 'Protocols' für strukturelles Subtyping.
Single Responsibility Principle - Jedes Interface hat GENAU EINE Aufgabe.
"""
from typing import Protocol, List, Dict, Any, Optional
from .entities import RegressionModel
from .value_objects import DatasetMetadata, RegressionType, Result


# =============================================================================
# Data Provider Interfaces (Trennung nach Aufgabenbereichen)
# =============================================================================

class IDatasetFetcher(Protocol):
    """Schnittstelle zum Abrufen eines einzelnen Datensatzes."""
    
    def fetch(self, dataset_id: str, n: int, **kwargs) -> Result:
        """
        Ruft Rohdaten ab.
        Gibt Success(Dict) oder Failure(Fehlermeldung) zurück.
        """
        ...


class IDatasetLister(Protocol):
    """Schnittstelle zum Auflisten verfügbarer Datensätze."""
    
    def list_all(self) -> List[DatasetMetadata]:
        """Listet alle verfügbaren Datensätze auf."""
        ...


class IDataProvider(IDatasetFetcher, IDatasetLister, Protocol):
    """Kombiniertes Interface für Datenoperationen (Abwärtskompatibel)."""
    
    def get_dataset(self, dataset_id: str, n: int, **kwargs) -> Dict[str, Any]:
        """Legacy-Methode - intern sollte 'fetch()' für Result-basiertes Error-Handling genutzt werden."""
        ...
        
    def get_all_datasets(self) -> Dict[str, List[Dict[str, str]]]:
        """Gibt alle Datensätze gruppiert nach Typ zurück."""
        ...
        
    def get_raw_data(self, dataset_id: str) -> Dict[str, Any]:
        """Gibt rohe tabellarische Daten für einen Datensatz zurück (für den Explorer)."""
        ...


# =============================================================================
# Regression Service Interfaces (Trennung der Trainings-Verantwortlichkeiten)
# =============================================================================

class ISimpleRegressionTrainer(Protocol):
    """Schnittstelle für das Training einfacher Regressionsmodelle."""
    
    def train(self, x: List[float], y: List[float]) -> RegressionModel:
        """Trainiert eine einfache Regression: y = β₀ + β₁x."""
        ...


class IMultipleRegressionTrainer(Protocol):
    """Schnittstelle für das Training multipler Regressionsmodelle."""
    
    def train(
        self, 
        x: List[List[float]], 
        y: List[float], 
        variable_names: List[str]
    ) -> RegressionModel:
        """Trainiert eine multiple Regression: y = β₀ + β₁x₁ + β₂x₂ + ..."""
        ...


class IRegressionService(Protocol):
    """Kombiniertes Interface für Regressionsoperationen (Abwärtskompatibel)."""
    
    def train_simple(self, x: List[float], y: List[float]) -> RegressionModel:
        """Trainiert ein einfaches Regressionsmodell."""
        ...
        
    def train_multiple(
        self, 
        x: List[List[float]], 
        y: List[float], 
        variable_names: List[str]
    ) -> RegressionModel:
        """Trainiert ein multiples Regressionsmodell."""
        ...


# =============================================================================
# Model Repository Interface (Persistierung)
# =============================================================================

class IModelRepository(Protocol):
    """Schnittstelle für das Speichern und Abrufen von Modellen."""
    
    def save(self, model: RegressionModel) -> str:
        """Speichert ein Modell und gibt die model_id zurück."""
        ...
    
    def get(self, model_id: str) -> Optional[RegressionModel]:
        """Ruft ein Modell anhand seiner ID ab."""
        ...
    
    def delete(self, model_id: str) -> bool:
        """Löscht ein Modell und gibt den Erfolgsstatus zurück."""
        ...


# =============================================================================
# Prediction Interface (Vorhersagen)
# =============================================================================

class IPredictor(Protocol):
    """Protokoll für das Erstellen von Vorhersagen."""
    def predict(self, model: RegressionModel, data: Dict[str, Any]) -> float:
        """Erstellt eine Vorhersage basierend auf dem Modell und neuen Daten."""
        ...


class IClassificationService(Protocol):
    """Protokoll für Klassifikationsoperationen (Logistic Regression, KNN)."""
    
    def train_logistic(self, X: Any, y: Any) -> Any: # Typen als Any für Domain-Purity
        """Trainiert ein Logistisches Regressionsmodell."""
        ...
        
    def train_knn(self, X: Any, y: Any, k: int) -> Any:
        """Trainiert ein K-Nearest Neighbors Modell."""
        ...
        
    def calculate_metrics(self, y_true: Any, y_pred: Any, y_prob: Any) -> Any:
        """Berechnet Performance-Metriken für die Klassifikation."""
        ...
