"""
API Endpoints - Framework-agnostische Business-Logik.

Diese Klassen enthalten die Kernlogik der API, OHNE Abhängigkeit von einem Web-Framework 
(wie Flask oder FastAPI). Sie können von jedem Framework gekapselt werden.
Alle Methoden geben JSON-serialisierbare Dictionaries zurück.
"""

from typing import Dict, Any, Optional, List
import logging

from .serializers import (
    DataSerializer,
    StatsSerializer,
    PlotSerializer,
    ContentSerializer,
    PipelineSerializer,
)

from pydantic import ValidationError
from ..config.logging import configure_logging
from ..config.logger import log_api_error, log_domain_error, log_error_with_context
from .schemas import SimpleRegressionRequest, MultipleRegressionRequest, AIInterpretationRequest, DatasetType
from ..core.domain.value_objects import DomainError

logger = logging.getLogger(__name__)
configure_logging()


class RegressionAPI:
    """
    API für Regressionsanalysen.
    
    Bietet Endpunkte zur Durchführung von Berechnungen.
    Komplett framework-agnostisch - gibt reine Python-Dictionaries zurück.
    
    Usage (direct):
        api = RegressionAPI()
        result = api.run_simple(dataset="electronics", n=50)
        
    Usage (via Flask):
        @app.route('/api/regression/simple')
        def simple():
            return jsonify(api.run_simple(**request.json))
            
    Usage (via FastAPI):
        @app.post('/api/regression/simple')
        def simple(params: SimpleParams):
            return api.run_simple(**params.dict())
    """
    
    def __init__(self):
        """Initialisierung mit Lazy-Loading der Pipeline."""
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lädt die RegressionPipeline erst bei Bedarf (Lazy Loading)."""
        if self._pipeline is None:
            from ..infrastructure import RegressionPipeline
            self._pipeline = RegressionPipeline()
        return self._pipeline
    
    def run_simple(
        self,
        dataset: str = "electronics",
        n: int = 50,
        noise: float = 0.4,
        seed: int = 42,
        true_intercept: float = 0.6,
        true_slope: float = 0.52,
        include_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        Führt eine einfache lineare Regression aus.
        """
        logger.info(f"API: run_simple({dataset}, n={n})")
        
        try:
            # Validierung der Eingabeparameter mittels Pydantic
            request = SimpleRegressionRequest(
                dataset=dataset,
                n=n,
                noise=noise,
                seed=seed,
                true_intercept=true_intercept,
                true_slope=true_slope,
                include_predictions=include_predictions
            )
            
            # Ausführung über die Infrastruktur-Pipeline
            result = self.pipeline.run_simple(
                dataset=request.dataset.value,
                n=request.n,
                noise=request.noise,
                seed=request.seed,
                true_intercept=request.true_intercept,
                true_slope=request.true_slope,
            )
            # Rückgabe des serialisierten Ergebnisses
            return {
                "success": True,
                "data": PipelineSerializer.serialize(result, request.include_predictions),
            }
        except ValidationError as e:
            error_id = log_error_with_context(
                logger=logger,
                error=e,
                context="run_simple validation",
                endpoint="/api/regression/simple",
                method="POST",
                validation_errors=e.errors()
            )
            logger.warning(f"[ERROR_ID={error_id}] Validierungsfehler in run_simple: {e}")
            return {
                "success": False,
                "error": "Validierungsfehler",
                "error_id": error_id,
                "details": e.errors()
            }
        except DomainError as e:
            error_id = log_domain_error(
                logger=logger,
                error=e,
                context="run_simple domain validation",
                endpoint="/api/regression/simple",
                dataset=dataset,
                n=n
            )
            return {
                "success": False,
                "error": e.message,
                "error_code": e.code,
                "error_id": error_id
            }
        except Exception as e:
            error_id = log_api_error(
                logger=logger,
                error=e,
                endpoint="/api/regression/simple",
                method="POST",
                request_data={"dataset": dataset, "n": n, "noise": noise, "seed": seed}
            )
            return {
                "success": False,
                "error": str(e),
                "error_id": error_id
            }
    
    def run_multiple(
        self,
        dataset: str = "cities",
        n: int = 75,
        noise: float = 3.5,
        seed: int = 42,
        include_predictions: bool = True,
    ) -> Dict[str, Any]:
        """
        Führt eine multiple lineare Regression aus.
        """
        logger.info(f"API: run_multiple({dataset}, n={n})")
        
        try:
            # Validierung des Requests
            request = MultipleRegressionRequest(
                dataset=dataset,
                n=n,
                noise=noise,
                seed=seed,
                include_predictions=include_predictions
            )
            
            # Ausführung über die Pipeline
            result = self.pipeline.run_multiple(
                dataset=request.dataset.value,
                n=request.n,
                noise=request.noise,
                seed=request.seed,
            )
            
            return {
                "success": True,
                "data": PipelineSerializer.serialize(result, request.include_predictions),
            }
        except ValidationError as e:
            error_id = log_error_with_context(
                logger=logger,
                error=e,
                context="run_multiple validation",
                endpoint="/api/regression/multiple",
                method="POST",
                validation_errors=e.errors()
            )
            logger.warning(f"[ERROR_ID={error_id}] Validierungsfehler in run_multiple: {e}")
            return {
                "success": False,
                "error": "Validierungsfehler",
                "error_id": error_id,
                "details": e.errors()
            }
        except DomainError as e:
            error_id = log_domain_error(
                logger=logger,
                error=e,
                context="run_multiple domain validation",
                endpoint="/api/regression/multiple",
                dataset=dataset,
                n=n
            )
            return {
                "success": False,
                "error": e.message,
                "error_code": e.code,
                "error_id": error_id
            }
        except Exception as e:
            error_id = log_api_error(
                logger=logger,
                error=e,
                endpoint="/api/regression/multiple",
                method="POST",
                request_data={"dataset": dataset, "n": n, "noise": noise, "seed": seed}
            )
            return {
                "success": False,
                "error": str(e),
                "error_id": error_id
            }
    
    def get_datasets(self) -> Dict[str, Any]:
        """
        List available datasets.
        
        All datasets are available for BOTH simple and multiple regression.
        This is intentional for educational purposes:
        - Simple regression shows larger error term (omitted variable bias)
        - Multiple regression shows improved R² when adding relevant predictors
        - Students can directly compare and understand "AHH, that's why!"
        
        Returns:
            Dictionary of available datasets
        """
        return {
            "success": True,
            "data": {
                "simple": [
                    {
                        "id": "electronics",
                        "name": "Elektronikmarkt",
                        "description": "Verkaufsfläche → Umsatz",
                        "icon": "🏪",
                        "hint": "➡️ Multiple: +Marketingbudget",
                    },
                    {
                        "id": "advertising",
                        "name": "Werbestudie",
                        "description": "Werbeausgaben → Umsatz",
                        "icon": "📢",
                        "hint": "➡️ Multiple: +Produktqualität",
                    },
                    {
                        "id": "temperature",
                        "name": "Eisverkauf",
                        "description": "Temperatur → Verkauf",
                        "icon": "🍦",
                        "hint": "➡️ Multiple: +Wochenende",
                    },
                    {
                        "id": "cities",
                        "name": "Städtestudie (nur Preis)",
                        "description": "Preis → Umsatz ⚠️ Omitted Variable!",
                        "icon": "🏙️",
                        "hint": "💡 Vergleiche mit Multipler Regression!",
                        "educational": True,
                    },
                    {
                        "id": "houses",
                        "name": "Hauspreise (nur Fläche)",
                        "description": "Wohnfläche → Preis ⚠️ Omitted Variable!",
                        "icon": "🏠",
                        "hint": "💡 Pool-Effekt fehlt! Wechsle zu Multiple.",
                        "educational": True,
                    },
                    {
                        "id": "cantons",
                        "name": "🇨🇭 Schweizer Kantone",
                        "description": "Bevölkerung → BIP",
                        "icon": "🇨🇭",
                        "hint": "💡 Vergleiche mit Multipler Regression!",
                        "educational": True,
                    },
                    {
                        "id": "weather",
                        "name": "🌤️ Schweizer Wetter",
                        "description": "Höhe → Temperatur",
                        "icon": "🌤️",
                        "hint": "💡 Vergleiche mit Multipler Regression!",
                        "educational": True,
                    },
                    {
                        "id": "world_bank",
                        "name": "🏦 World Bank (Global)",
                        "description": "GDP -> Life Exp",
                        "icon": "🏦",
                        "hint": "💡 Preston Curve",
                        "educational": True,
                    },
                    {
                        "id": "fred_economic",
                        "name": "💰 FRED (US Economy)",
                        "description": "Unemployment -> GDP",
                        "icon": "💰",
                        "hint": "💡 Phillips Curve",
                        "educational": True,
                    },
                    {
                        "id": "who_health",
                        "name": "🏥 WHO (Health)",
                        "description": "Health Spend -> Life Exp",
                        "icon": "🏥",
                        "hint": "💡 Global Health",
                        "educational": True,
                    },
                    {
                        "id": "eurostat",
                        "name": "🇪🇺 Eurostat (EU)",
                        "description": "Emp -> GDP",
                        "icon": "🇪🇺",
                        "hint": "💡 EU Economics",
                        "educational": True,
                    },
                    {
                        "id": "nasa_weather",
                        "name": "🛰️ NASA POWER",
                        "description": "Temp -> Crop Yield",
                        "icon": "🛰️",
                        "hint": "💡 Agro-Climatology",
                        "educational": True,
                    },
                ],
                "multiple": [
                    {
                        "id": "cities",
                        "name": "Städtestudie",
                        "description": "Preis & Werbung → Umsatz",
                        "icon": "🏙️",
                        "hint": "➡️ Simple: Nur Preis (Bias Demo)",
                    },
                    {
                        "id": "houses",
                        "name": "Hauspreise",
                        "description": "Fläche & Pool → Preis",
                        "icon": "🏠",
                        "hint": "➡️ Simple: Nur Fläche (Bias Demo)",
                    },
                    {
                        "id": "electronics",
                        "name": "Elektronikmarkt (+Marketing)",
                        "description": "Fläche & Budget → Umsatz",
                        "icon": "🏪",
                        "educational": True,
                    },
                    {
                        "id": "advertising",
                        "name": "Werbestudie (+Qualität)",
                        "description": "Ausgaben & Rating → Umsatz",
                        "icon": "📢",
                        "educational": True,
                    },
                    {
                        "id": "temperature",
                        "name": "Eisverkauf (+Wochenende)",
                        "description": "Grad & Tag → Einheiten",
                        "icon": "🍦",
                        "educational": True,
                    },
                    {
                        "id": "cantons",
                        "name": "🇨🇭 Schweizer Kantone",
                        "description": "Bevölkerung, Ausländer → BIP",
                        "icon": "🇨🇭",
                        "hint": "💡 3 Prädiktoren (Sozioökonomisch)",
                        "educational": True,
                    },
                    {
                        "id": "weather",
                        "name": "🌤️ Schweizer Wetter",
                        "description": "Höhe & Sonne → Temperatur",
                        "icon": "🌤️",
                        "hint": "💡 Negative Korrelation bei Höhe!",
                        "educational": True,
                    },
                    {
                        "id": "world_bank",
                        "name": "🏦 World Bank (Global)",
                        "description": "GDP, Education -> Life Exp",
                        "icon": "🏦",
                        "hint": "💡 Development Data",
                        "educational": True,
                    },
                    {
                        "id": "fred_economic",
                        "name": "💰 FRED (US Economy)",
                        "description": "Unemployment, Interest -> GDP",
                        "icon": "💰",
                        "hint": "💡 Macro Data",
                        "educational": True,
                    },
                    {
                        "id": "who_health",
                        "name": "🏥 WHO (Health)",
                        "description": "Spending, Sanitation -> Life Exp",
                        "icon": "🏥",
                        "hint": "💡 Health Data",
                        "educational": True,
                    },
                    {
                        "id": "eurostat",
                        "name": "🇪🇺 Eurostat (EU)",
                        "description": "Employment, Education -> GDP",
                        "icon": "🇪🇺",
                        "hint": "💡 EU Data",
                        "educational": True,
                    },
                    {
                        "id": "nasa_weather",
                        "name": "🛰️ NASA POWER",
                        "description": "Temp, Solar -> Crop Yield",
                        "icon": "🛰️",
                        "hint": "💡 Climate Data",
                        "educational": True,
                    },
                ],
            },
        }


class ContentAPI:
    """
    API für edukative Inhalte.
    
    Gibt framework-agnostische Inhaltsstrukturen zurück, die vom Frontend 
    (z.B. React oder Streamlit) gerendert werden können.
    """
    
    def __init__(self):
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lädt die RegressionPipeline erst bei Bedarf (Lazy Loading)."""
        if self._pipeline is None:
            from ..infrastructure import RegressionPipeline
            self._pipeline = RegressionPipeline()
        return self._pipeline
    
    def get_simple_content(
        self,
        dataset: str = "electronics",
        n: int = 50,
        noise: float = 0.4,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Liefert edukative Inhalte für die einfache Regression.
        """
        logger.info(f"API: get_simple_content({dataset}, n={n})")
        
        try:
            # Generierung des Inhalts über den ContentBuilder in der Pipeline
            content = self.pipeline.get_simple_content(
                dataset=dataset, n=n, noise=noise, seed=seed
            )
            # Serialisierung für den Transport über die API
            return {
                "success": True,
                "data": ContentSerializer.serialize(content)
            }
        except Exception as e:
            error_id = log_api_error(
                logger=logger,
                error=e,
                endpoint="/api/content/simple",
                method="GET",
                request_data={"dataset": dataset, "n": n, "noise": noise, "seed": seed}
            )
            return {"success": False, "error": str(e), "error_id": error_id}
    
    def get_multiple_content(
        self,
        dataset: str = "cities",
        n: int = 75,
        noise: float = 3.5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Liefert edukative Inhalte für die multiple Regression.
        """
        logger.info(f"API: get_multiple_content({dataset}, n={n})")
        
        try:
            content = self.pipeline.get_multiple_content(
                dataset=dataset, n=n, noise=noise, seed=seed
            )
            return {
                "success": True,
                "data": ContentSerializer.serialize(content)
            }
        except Exception as e:
            error_id = log_api_error(
                logger=logger,
                error=e,
                endpoint="/api/content/simple",
                method="GET",
                request_data={"dataset": dataset, "n": n, "noise": noise, "seed": seed}
            )
            return {"success": False, "error": str(e), "error_id": error_id}
    
    def get_content_schema(self) -> Dict[str, Any]:
        """
        Get schema for content elements.
        
        Useful for frontend developers to understand the structure.
        """
        return {
            "success": True,
            "schema": {
                "element_types": [
                    "markdown", "metric", "metric_row", "formula", "plot",
                    "table", "columns", "expander", "info_box", "warning_box",
                    "success_box", "code_block", "divider", "chapter", "section"
                ],
                "structure": {
                    "EducationalContent": {
                        "title": "string",
                        "subtitle": "string",
                        "chapters": "Chapter[]",
                    },
                    "Chapter": {
                        "number": "string",
                        "title": "string",
                        "icon": "string (emoji)",
                        "sections": "Section[] | ContentElement[]",
                    },
                    "Section": {
                        "title": "string",
                        "icon": "string (emoji)",
                        "content": "ContentElement[]",
                    },
                    "Markdown": {"text": "string (markdown)"},
                    "Metric": {"label": "string", "value": "string", "help_text": "string", "delta": "string"},
                    "MetricRow": {"metrics": "Metric[]"},
                    "Formula": {"latex": "string", "inline": "boolean"},
                    "Plot": {"plot_key": "string", "title": "string", "description": "string", "height": "number"},
                    "Table": {"headers": "string[]", "rows": "string[][]", "caption": "string"},
                    "Columns": {"columns": "ContentElement[][]", "widths": "number[]"},
                    "Expander": {"title": "string", "content": "ContentElement[]", "expanded": "boolean"},
                    "InfoBox": {"content": "string"},
                    "WarningBox": {"content": "string"},
                    "SuccessBox": {"content": "string"},
                    "CodeBlock": {"code": "string", "language": "string"},
                    "Divider": {},
                },
            },
        }


    def get_classification_content(
        self,
        dataset: str = "fruits",
        n: int = 100,
        noise: float = 0.2,
        seed: int = 42,
        method: str = "knn",
        k: int = 3,
        train_size: float = 0.8,
        stratify: bool = False
    ) -> Dict[str, Any]:
        """
        Get educational content for classification.
        
        Args:
            dataset: Dataset ID
            n: Number of points
            noise: Noise level
            seed: Random seed
            method: "logistic" or "knn"
            k: Neighbors for KNN
            train_size: Training set proportion
            stratify: Whether to stratify split
            
        Returns:
            Complete content structure for Streamlit/API
        """
        logger.info(f"API: get_classification_content({dataset}, method={method})")
        
        try:
            # 1. Use DI Container to get Use Case
            from ..core.application.dtos import ClassificationRequestDTO
            from ..container import Container
            
            container = Container()
            use_case = container.run_classification_use_case
            
            # 2. Execute Use Case
            request = ClassificationRequestDTO(
                dataset_id=dataset,
                n_observations=n,
                noise_level=noise,
                seed=seed,
                method=method,
                k_neighbors=k,
                train_size=train_size,
                stratify=stratify
            )
            response_dto = use_case.execute(request)
            
            if not response_dto.success:
                 return {"success": False, "error": response_dto.error}

            # 3. Flatten Stats for ContentBuilder
            from .serializers import ClassificationSerializer
            stats = ClassificationSerializer.to_flat_dict(response_dto)
            
            # 4. Select Content Builder
            from ..infrastructure.content.logistic_regression import LogisticRegressionContent
            from ..infrastructure.content.ml_fundamentals import MLFundamentalsContent
            
            # Map "logical" names to PlotBuilder keys
            # PlotBuilder returns: scatter (Main 3D), residuals (Confusion Matrix), diagnostics (ROC)
            # ContentBuilders expect: scatter, confusion_matrix_interactive, roc_curve, etc.
            # We map the PlotBuilder outputs to the keys expected by the content
            plot_keys = {
                # Logistic Regression
                "linear_on_binary": None, # Not generated by standard builder yet
                "sigmoid_function": None,
                "decision_boundary": "scatter", # Main 3D plot shows decision usually
                "confusion_matrix_interactive": "residuals", # Abused key
                "roc_curve": "diagnostics", # Abused key
                "precision_recall_tradeoff": None,
                "precision_recall_curve": None,
                
                # ML Fundamentals
                "knn_visualization": "scatter", # Main 3D plot
                "knn_decision_boundaries": "scatter",
                "curse_of_dimensionality": None,
            }
            
            if method == "logistic":
                builder = LogisticRegressionContent(stats, plot_keys)
            else:
                builder = MLFundamentalsContent(stats, plot_keys)
                
            content = builder.build()
            
            # 5. Generate Plots
            # Reconstruct Value Objects for PlotBuilder
            from ..core.domain.value_objects import ClassificationResult, ClassificationMetrics
            from ..infrastructure.data.generators import ClassificationDataResult
            from ..infrastructure.services.plot import PlotBuilder
            import numpy as np
            
            # Reconstruct Data
            # DTO arrays might be tuples/lists, convert to numpy
            X_arr = np.array(response_dto.X_data)
            y_arr = np.array(response_dto.y_data)
            
            data_vo = ClassificationDataResult(
                X=X_arr,
                y=y_arr,
                feature_names=list(response_dto.feature_names) if response_dto.feature_names else [],
                target_names=list(response_dto.target_names) if response_dto.target_names else [],
                context_title=response_dto.dataset_name,
                context_description=response_dto.dataset_description
            )
            
            # Reconstruct Result
            # Metrics need specialized object if PlotBuilder uses it deeply?
            # PlotBuilder uses result.metrics.confusion_matrix directly.
            metrics_vo = ClassificationMetrics(
                accuracy=response_dto.metrics.get("accuracy", 0),
                precision=response_dto.metrics.get("precision", 0),
                recall=response_dto.metrics.get("recall", 0),
                f1_score=response_dto.metrics.get("f1", 0),
                confusion_matrix=np.array(response_dto.metrics.get("confusion_matrix")) if response_dto.metrics.get("confusion_matrix") else None,
                auc=response_dto.metrics.get("auc"),
            )
            
            result_vo = ClassificationResult(
                model_params=response_dto.parameters,
                metrics=metrics_vo,
                probabilities=np.array(response_dto.probabilities) if response_dto.probabilities else None,
                predictions=np.array(response_dto.predictions) if response_dto.predictions else None,
                classes=np.array(response_dto.classes) if response_dto.classes else None
            )
            
            plot_builder = PlotBuilder()
            plots_collection = plot_builder.classification_plots(data_vo, result_vo)
            
            # 6. Serialize
            from .serializers import ContentSerializer, PlotSerializer
            
            return {
                "success": True,
                "content": ContentSerializer.serialize(content),
                "plots": PlotSerializer.serialize_collection(plots_collection),
                "data": {
                    "X": response_dto.X_data,
                    "y": response_dto.y_data,
                    "target_names": response_dto.target_names
                },
                "results": {
                    "metrics": response_dto.metrics,
                    "test_metrics": response_dto.test_metrics,
                    "params": response_dto.parameters,
                    "method": response_dto.method
                },
                "stats": stats,
                "params": response_dto.parameters
            }

        except Exception as e:
            error_id = log_api_error(
                logger=logger,
                error=e,
                endpoint="/api/ai/interpret",
                method="POST",
                request_data={"stats": stats, "dataset": dataset, "regression_type": regression_type}
            )
            return {
                "success": False,
                "error": str(e),
                "error_id": error_id
            }

    def get_split_preview(
        self,
        dataset: str,
        train_size: float = 0.8,
        stratify: bool = False,
        seed: int = 42,
        n: int = 100,
        noise: float = 0.2
    ) -> Dict[str, Any]:
        """
        Get statistics for a potential data split.
        Useful for interactive preview in frontend.
        """
        try:
            from ..container import Container
            from dataclasses import asdict
            
            container = Container()
            use_case = container.preview_split_use_case
            
            stats = use_case.execute(
                dataset_id=dataset,
                n=n,
                noise=noise,
                seed=seed,
                train_size=train_size,
                stratify=stratify
            )
            
            return {
                "success": True,
                "stats": asdict(stats)
            }
            
        except Exception as e:
            error_id = log_api_error(
                logger=logger,
                error=e,
                endpoint="/api/data/split-preview",
                method="GET",
                request_data={"dataset": dataset, "train_size": train_size, "seed": seed}
            )
            return {"success": False, "error": str(e), "error_id": error_id}

    def get_datasets_list(self) -> Dict[str, Any]:
        """List all available datasets."""
        try:
            # We bypass Use Case for simple data listing, or use Provider directly
            # Since Provider is infrastructure, we should ideally go through a Use Case or Service.
            # But for simplicity in this read-only operation:
            from ..container import Container
            container = Container()
            # Accessing provider directly via private member or property?
            # Provider is injected into Use Cases. Container creates it.
            # We can expose it or create a simple ReadService.
            # Let's use the provider instance directly from container logic (re-instantiate)
            # OR better: add `dataset_service` to container if we want to be strict.
            
            # Temporary: direct instantiation as in container
            from ..infrastructure import DataProviderImpl
            provider = DataProviderImpl()
            return {
                "success": True, 
                "datasets": provider.get_all_datasets()
            }
        except Exception as e:
            error_id = log_api_error(
                logger=logger,
                error=e,
                endpoint="/api/data/datasets",
                method="GET"
            )
            return {"success": False, "error": str(e), "error_id": error_id}

    def get_dataset_raw(self, dataset_id: str) -> Dict[str, Any]:
        """Get raw data for preview table."""
        try:
            from ..infrastructure import DataProviderImpl
            provider = DataProviderImpl()
            raw_data = provider.get_raw_data(dataset_id)
            return {
                "success": True,
                "data": raw_data
            }
        except Exception as e:
            error_id = log_api_error(
                logger=logger,
                error=e,
                endpoint=f"/api/data/datasets/{dataset_id}",
                method="GET",
                request_data={"dataset_id": dataset_id}
            )
            return {"success": False, "error": str(e), "error_id": error_id}



class AIInterpretationAPI:
    """
    AI Interpretation API.
    
    Provides AI-powered interpretation of regression results.
    Completely framework agnostic.
    """
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        """Lazy load Perplexity client."""
        if self._client is None:
            from ..ai import PerplexityClient
            self._client = PerplexityClient()
        return self._client
    
    def interpret(
        self,
        stats: Dict[str, Any],
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Get AI interpretation of regression statistics.
        
        Args:
            stats: Statistics dictionary (from StatsSerializer.to_flat_dict)
            use_cache: Whether to use cached responses
            
        Returns:
            AI interpretation result
        """
        logger.info("API: interpret")
        
        response = self.client.interpret_r_output(stats, use_cache)
        
        return {
            "success": not response.error,
            "interpretation": {
                "content": response.content,
                "model": response.model,
                "cached": response.cached,
                "latency_ms": response.latency_ms,
            },
            "usage": response.usage,
            "citations": response.citations,
        }
    
    def get_r_output(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate R-style output for display.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            R-output string
        """
        r_output = self.client.generate_r_output(stats)
        
        return {
            "success": True,
            "r_output": r_output,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get AI service status.
        
        Returns:
            Status information
        """
        return {
            "success": True,
            "status": self.client.get_status(),
        }
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear interpretation cache.
        
        Returns:
            Cache clear result
        """
        result = self.client.clear_cache()
        return {
            "success": True,
            **result,
        }


# =========================================================================
# Unified API - Combines all endpoints
# =========================================================================

class UnifiedAPI:
    """
    Unified API combining all endpoints.
    
    Single entry point for all API functionality.
    """
    
    def __init__(self):
        self.regression = RegressionAPI()
        self.content = ContentAPI()
        self.ai = AIInterpretationAPI()
    
    def get_openapi_spec(self) -> Dict[str, Any]:
        """
        Generate OpenAPI specification.
        
        Returns:
            OpenAPI 3.0 specification
        """
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Regression Analysis API",
                "version": "1.0.0",
                "description": "Platform-agnostic API for regression analysis with educational content",
            },
            "paths": {
                "/api/regression/simple": {
                    "post": {
                        "summary": "Run simple regression",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "dataset": {"type": "string", "default": "electronics"},
                                            "n": {"type": "integer", "default": 50},
                                            "noise": {"type": "number", "default": 0.4},
                                            "seed": {"type": "integer", "default": 42},
                                        },
                                    },
                                },
                            },
                        },
                        "responses": {"200": {"description": "Regression result"}},
                    },
                },
                "/api/regression/multiple": {
                    "post": {
                        "summary": "Run multiple regression",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "dataset": {"type": "string", "default": "cities"},
                                            "n": {"type": "integer", "default": 75},
                                            "noise": {"type": "number", "default": 3.5},
                                            "seed": {"type": "integer", "default": 42},
                                        },
                                    },
                                },
                            },
                        },
                        "responses": {"200": {"description": "Regression result"}},
                    },
                },
                "/api/content/simple": {
                    "post": {
                        "summary": "Get educational content for simple regression",
                        "responses": {"200": {"description": "Educational content"}},
                    },
                },
                "/api/content/multiple": {
                    "post": {
                        "summary": "Get educational content for multiple regression",
                        "responses": {"200": {"description": "Educational content"}},
                    },
                },
                "/api/content/schema": {
                    "get": {
                        "summary": "Get content schema",
                        "responses": {"200": {"description": "Content schema"}},
                    },
                },
                "/api/ai/interpret": {
                    "post": {
                        "summary": "Get AI interpretation",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "stats": {"type": "object"},
                                            "use_cache": {"type": "boolean", "default": True},
                                        },
                                        "required": ["stats"],
                                    },
                                },
                            },
                        },
                        "responses": {"200": {"description": "AI interpretation"}},
                    },
                },
                "/api/ai/r-output": {
                    "post": {
                        "summary": "Generate R-style output",
                        "responses": {"200": {"description": "R output"}},
                    },
                },
                "/api/ai/status": {
                    "get": {
                        "summary": "Get AI service status",
                        "responses": {"200": {"description": "Status"}},
                    },
                },
                "/api/datasets": {
                    "get": {
                        "summary": "List available datasets",
                        "responses": {"200": {"description": "Dataset list"}},
                    },
                },
            },
        }


class ClassificationAPI:
    """
    API für Klassifikationsanalysen (Logistische Regression, KNN).
    """
    
    def __init__(self):
        # Lazy Loading der Komponenten
        self._use_case = None
        self._plot_builder = None
        
    @property
    def use_case(self):
        """Initialisiert den Use Case für Klassifikation."""
        if self._use_case is None:
            from ..core.application.use_cases import RunClassificationUseCase
            from ..infrastructure.data.provider import DataProviderImpl
            from ..infrastructure.services.classification import ClassificationServiceImpl
            
            self._use_case = RunClassificationUseCase(
                data_provider=DataProviderImpl(),
                classification_service=ClassificationServiceImpl()
            )
        return self._use_case
        
    @property
    def plot_builder(self):
        """Initialisiert den PlotBuilder."""
        if self._plot_builder is None:
            from ..infrastructure.services.plot import PlotBuilder
            self._plot_builder = PlotBuilder()
        return self._plot_builder
        
    def run_classification(
        self,
        dataset: str = "fruits",
        n: int = 100,
        noise: float = 0.2,
        seed: int = 42,
        method: str = "logistic",
        k: int = 3,
        train_size: float = 0.8,
        stratify: bool = False
    ) -> Dict[str, Any]:
        """Führt eine Klassifikationsanalyse aus (Training & Evaluation)."""
        logger.info(f"API: run_classification({method}, {dataset})")
        
        try:
            from ..core.application.dtos import ClassificationRequestDTO
            
            # 1. Request-DTO erstellen
            request_dto = ClassificationRequestDTO(
                dataset_id=dataset,
                n_observations=n,
                noise_level=noise,
                seed=seed,
                method=method,
                k_neighbors=k,
                train_size=train_size,
                stratify=stratify
            )
            
            # 2. Use Case ausführen (Business Logik)
            response = self.use_case.execute(request_dto)
            
            # 3. Plots generieren (Infrastruktur-Service)
            from ..infrastructure.data.generators import ClassificationDataResult
            import numpy as np
            
            data_res = ClassificationDataResult(
                X=np.array(response.X_data), 
                y=np.array(response.y_data),
                target_names=list(response.target_names) if response.target_names else None,
                feature_names=list(response.feature_names) if response.feature_names else None
            )
            
            # Rekonstruktion der Domain-Objekte für den PlotBuilder
            from ..core.domain.value_objects import ClassificationResult, ClassificationMetrics
            
            # Metriken für Trainingsdaten
            cm = np.array(response.metrics['confusion_matrix']) if response.metrics.get('confusion_matrix') else None
            metrics_obj = ClassificationMetrics(
                accuracy=response.metrics['accuracy'],
                precision=response.metrics['precision'],
                recall=response.metrics['recall'],
                f1_score=response.metrics['f1'],
                confusion_matrix=cm,
                auc=None
            )

            # Metriken für Testdaten
            test_metrics_obj = None
            if response.test_metrics:
                 cm_test = np.array(response.test_metrics['confusion_matrix']) if response.test_metrics.get('confusion_matrix') else None
                 test_metrics_obj = ClassificationMetrics(
                    accuracy=response.test_metrics['accuracy'],
                    precision=response.test_metrics['precision'],
                    recall=response.test_metrics['recall'],
                    f1_score=response.test_metrics['f1'],
                    confusion_matrix=cm_test,
                    auc=None
                )
            
            # Komplettes Klassifikations-Resultat
            class_result = ClassificationResult(
                is_success=response.success,
                classes=list(response.classes),
                model_params=response.parameters,
                metrics=metrics_obj,
                test_metrics=test_metrics_obj,
                predictions=np.array(response.predictions),
                probabilities=np.array(response.probabilities) if response.probabilities else None
            )
            
            # Erstellung der Plots (z.B. Decision Boundaries)
            plot_collection = self.plot_builder.classification_plots(data_res, class_result)
            
            # 4. Alles serialisieren und zurückgeben
            from .serializers import PlotSerializer
            
            return {
                "success": True,
                "data": {
                    "X": response.X_data,
                    "y": response.y_data,
                    "feature_names": response.feature_names,
                    "target_names": response.target_names
                },
                "results": {
                    "metrics": response.metrics,
                    "test_metrics": response.test_metrics,
                    "params": response.parameters,
                    "method": response.method
                },
                "plots": PlotSerializer.serialize_collection(plot_collection),
                "metadata": {
                    "dataset": response.dataset_name,
                    "description": response.dataset_description
                }
            }
            
        except Exception as e:
            error_id = log_api_error(
                logger=logger,
                error=e,
                endpoint="/api/ml/classify",
                method="POST",
                request_data={"dataset": dataset, "n": n, "k": k, "seed": seed}
            )
            return {
                "success": False,
                "error": str(e),
                "error_id": error_id
            }
