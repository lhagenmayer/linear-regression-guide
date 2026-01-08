"""
Application Use Cases.
Orchestrierung von Domain-Objekten und Infrastructure-Services mittels Dependency Injection.
"""
from typing import Dict, Any, List
import numpy as np
from .dtos import RegressionRequestDTO, RegressionResponseDTO, ClassificationRequestDTO, ClassificationResponseDTO
from ..domain.interfaces import IDataProvider, IRegressionService, IClassificationService
from ..domain.entities import RegressionModel
from ..domain.value_objects import RegressionType


class RunRegressionUseCase:
    """
    Use Case: Ausführung einer Regressionsanalyse (Einfach oder Multipel).
    
    Folgt der Clean Architecture: Orchestriert Domain-Objekte, ohne selbst 
    Geschäftslogik (Mathematik) zu implementieren.
    """
    
    def __init__(self, data_provider: IDataProvider, regression_service: IRegressionService):
        """
        Dependency Injection über den Konstruktor. 
        Die konkreten Implementierungen werden vom DI-Container (container.py) injiziert.
        """
        self.data_provider = data_provider
        self.regression_service = regression_service
        
    def execute(self, request: RegressionRequestDTO) -> RegressionResponseDTO:
        """
        Führt die Regressions-Pipeline aus:
        1. Daten abrufen (IDataProvider)
        2. Modell trainieren (IRegressionService)
        3. Response DTO erstellen
        """
        # 1. Daten über das Interface abrufen
        data_result = self.data_provider.get_dataset(
            dataset_id=request.dataset_id,
            n=request.n_observations,
            noise=request.noise_level,
            seed=request.seed,
            true_intercept=request.true_intercept or 0.6,
            true_slope=request.true_slope or 0.52,
            regression_type=request.regression_type.name.lower()  # Enum zu String für Infrastruktur
        )
        
        # 2. Regression über den Service ausführen
        if request.regression_type == RegressionType.MULTIPLE:
            # Multiple Regression: Mehrere Features (x1, x2)
            x_data = [data_result["x1"], data_result["x2"]]
            y_data = data_result["y"]
            variable_names = [data_result.get("x1_label", "x1"), data_result.get("x2_label", "x2")]
            
            model = self.regression_service.train_multiple(x_data, y_data, variable_names)
        else:
            # Einfache Regression (Standard): Ein Feature (x)
            x_data = data_result["x"]
            y_data = data_result["y"]
            
            model = self.regression_service.train_simple(x_data, y_data)
        
        # 3. Metadaten hinzufügen
        model.dataset_metadata = data_result.get("metadata")
        model.regression_type = request.regression_type
        
        # 4. Response DTO konstruieren
        return self._build_response(model, data_result, request.regression_type)

    def _build_response(
        self, 
        model: RegressionModel, 
        data_raw: Dict[str, Any],
        regression_type: RegressionType
    ) -> RegressionResponseDTO:
        """Erzeugt ein unveränderliches Response-DTO aus dem Modell und den Rohdaten."""
        params = model.parameters
        metrics = model.metrics
        
        # X-Daten je nach Regressionsart aufbereiten
        if regression_type == RegressionType.MULTIPLE:
            x_data = tuple([
                tuple(data_raw.get("x1", [])), 
                tuple(data_raw.get("x2", []))
            ])
            x_label = f"{data_raw.get('x1_label', 'x1')} & {data_raw.get('x2_label', 'x2')}"
        else:
            x_data = tuple(data_raw.get("x", []))
            x_label = data_raw.get("x_label", "x")
        
        return RegressionResponseDTO(
            model_id=model.id,
            success=model.is_trained(),
            coefficients=params.coefficients,
            metrics={
                "r_squared": metrics.r_squared,
                "r_squared_adj": metrics.r_squared_adj,
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "f_statistic": metrics.f_statistic,
                "p_value": metrics.p_value
            },
            x_data=x_data,
            y_data=tuple(data_raw.get("y", [])),
            residuals=tuple(model.residuals),
            predictions=tuple(model.predictions),
            x_label=x_label,
            y_label=data_raw.get("y_label", "y"),
            title=data_raw.get("context_title", ""),
            description=data_raw.get("context_description", ""),
            quality=model.get_quality(),
            is_significant=model.is_significant(),
            extra=data_raw.get("extra", {})
        )


class RunClassificationUseCase:
    """
    Use Case: Ausführung einer Klassifikationsanalyse (Logistic Regression / KNN).
    """
    
    def __init__(self, data_provider: IDataProvider, classification_service: IClassificationService):
        self.data_provider = data_provider
        self.classification_service = classification_service
        
        # Lazy Loading des Splitter-Services (Infrastruktur-Abhängigkeit)
        from ...infrastructure.services.data_splitting import DataSplitterService
        self.splitter_service = DataSplitterService()
        
    def execute(self, request: ClassificationRequestDTO) -> ClassificationResponseDTO:
        """Führt die Klassifikations-Pipeline (Split, Train, Evaluate) aus."""
        # 1. Daten abrufen
        data_raw = self.data_provider.get_dataset(
            dataset_id=request.dataset_id,
            n=request.n_observations,
            noise=request.noise_level,
            seed=request.seed,
            analysis_type="classification"
        )
        
        # 2. Arrays extrahieren (NumPy wird hier für die Übergabe an Services genutzt)
        if "X" not in data_raw or "y" not in data_raw:
             raise ValueError("Datensatz enthält keine X und y Daten für Klassifikation")
             
        X = np.array(data_raw["X"])
        y = np.array(data_raw["y"])
        
        # 3. Daten aufteilen (Train/Test Split)
        from ..domain.value_objects import SplitConfig
        config = SplitConfig(
            train_size=request.train_size,
            stratify=request.stratify,
            seed=request.seed
        )
        X_train, X_test, y_train, y_test = self.splitter_service.split_data(X, y, config)
        
        # 4. Modell trainieren (auf dem Trainings-Set)
        if request.method == "knn":
            # Bei KNN besteht das 'Training' primär aus dem Speichern der Daten
            train_result = self.classification_service.train_knn(X_train, y_train, k=request.k_neighbors)
        else:
            # Logistische Regression (Gradient Descent)
            train_result = self.classification_service.train_logistic(X_train, y_train)
            
        # 5. Evaluierung auf dem Test-Set (Unabhängige Validierung)
        test_metrics = self.classification_service.evaluate(
            X_test, 
            y_test, 
            train_result.model_params, 
            request.method
        )
        
        # 6. Vorhersagen für den gesamten Datensatz (für die Visualisierung im Frontend)
        if request.method == "knn":
            full_preds, full_probs = self.classification_service.predict_knn(X, train_result.model_params)
        else:
            full_preds, full_probs = self.classification_service.predict_logistic(X, train_result.model_params)
            
        # 7. Response DTO bauen (Konvertierung von NumPy zurück zu Tuples/Listen für Serialisierung)
        return ClassificationResponseDTO(
            success=train_result.is_success,
            method=request.method,
            classes=tuple(train_result.classes),
            metrics={ # Trainings-Metriken
                "accuracy": train_result.metrics.accuracy,
                "precision": train_result.metrics.precision,
                "recall": train_result.metrics.recall,
                "f1": train_result.metrics.f1_score,
                "confusion_matrix": train_result.metrics.confusion_matrix.tolist()
            },
            test_metrics={ # Test-Metriken (Wichtiger für Generalisierung)
                "accuracy": test_metrics.accuracy,
                "precision": test_metrics.precision,
                "recall": test_metrics.recall,
                "f1": test_metrics.f1_score,
                "confusion_matrix": test_metrics.confusion_matrix.tolist()
            },
            parameters=train_result.model_params,
            # Rückgabe der VOLLSTÄNDIGEN Daten für den Plot-Kontext
            X_data=tuple(map(tuple, X)), 
            y_data=tuple(y),
            
            # Vorhersagen auf dem gesamten Datensatz
            predictions=tuple(full_preds),
            probabilities=tuple(full_probs) if full_probs is not None else (),
            
            feature_names=tuple(data_raw.get("feature_names", [])),
            target_names=tuple(data_raw.get("target_names", [])),
            dataset_name=data_raw.get("name", "Datensatz"),
            dataset_description=data_raw.get("description", ""),
            extra={}
        )


class PreviewSplitUseCase:
    """
    Use Case: Vorschau der Statistiken einer Datenaufteilung (Split).
    Erlaubt es dem Nutzer im Frontend zu sehen, wie sich 'stratify' auf die Verteilung auswirkt.
    """
    
    def __init__(self, data_provider: IDataProvider, splitter_service):
        self.data_provider = data_provider
        self.splitter_service = splitter_service
        
    def execute(self, dataset_id: str, n: int, noise: float, seed: int, train_size: float, stratify: bool) -> Any:
        # 1. Daten abrufen
        data_raw = self.data_provider.get_dataset(
            dataset_id=dataset_id,
            n=n,
            noise=noise,
            seed=seed,
            analysis_type="classification"
        )
        
        if "y" not in data_raw:
             raise ValueError("Datensatz enthält keine y-Werte für den Split")
             
        y = np.array(data_raw["y"])
        
        # 2. Konfiguration erstellen
        from ..domain.value_objects import SplitConfig
        config = SplitConfig(
            train_size=train_size,
            stratify=stratify,
            seed=seed
        )
        
        # 3. Statistiken berechnen (ohne das Modell zu trainieren)
        stats = self.splitter_service.preview_split(y, config)
        
        return stats
