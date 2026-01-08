"""
Regressions-Pipeline - Einfache 4-stufige Datenverarbeitung.

Dieses Modul bietet eine vereinheitlichte Pipeline, die folgende Schritte orchestriert:
    1. GET      → Daten abrufen
    2. CALCULATE → Statistiken berechnen
    3. PLOT     → Visualisierungen erstellen
    4. DISPLAY  → In der UI rendern
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

from ..config import get_logger
from .data.generators import DataFetcher, DataResult, MultipleRegressionDataResult
from .services.calculate import StatisticsCalculator, RegressionResult, MultipleRegressionResult
from .services.plot import PlotBuilder, PlotCollection

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """
    Vollständiges Ergebnis einer Pipeline-Ausführung.
    
    Enthält alle Daten, Berechnungen und Plots für die Anzeige.
    """
    # Schritt 1: Daten
    data: Union[DataResult, MultipleRegressionDataResult]
    
    # Schritt 2: Berechnungen
    stats: Union[RegressionResult, MultipleRegressionResult]
    
    # Schritt 3: Plots
    plots: PlotCollection
    
    # Metadaten
    pipeline_type: str  # "simple" oder "multiple"
    params: Dict[str, Any]


class RegressionPipeline:
    """
    Einfache 4-stufige Regressions-Pipeline.
    
    Die Pipeline folgt einem klaren, linearen Fluss:
    
        ┌─────────┐    ┌───────────┐    ┌──────┐    ┌─────────┐
        │   GET   │ → │ CALCULATE │ → │ PLOT │ → │ DISPLAY │
        └─────────┘    └───────────┘    └──────┘    └─────────┘
    
    Beispiel:
        pipeline = RegressionPipeline()
        
        # Komplette Pipeline ausführen
        result = pipeline.run_simple(
            dataset="electronics",
            n=50,
            noise=0.4,
            seed=42
        )
        
        # In Streamlit anzeigen
        pipeline.display(result)
    """
    
    def __init__(self):
        """Initialisiert die Pipeline mit allen notwendigen Komponenten."""
        self.fetcher = DataFetcher()
        self.calculator = StatisticsCalculator()
        self.plotter = PlotBuilder()
        
        # Der Renderer wird verzögert geladen, um Streamlit-Import-Probleme zu vermeiden
        self._renderer = None
        
        logger.info("RegressionPipeline initialisiert")
    
    @property
    def renderer(self):
        """Lazy Load für den UIRenderer."""
        if self._renderer is None:
            from .display import UIRenderer
            self._renderer = UIRenderer()
        return self._renderer
    
    def run_simple(
        self,
        dataset: str = "electronics",
        n: int = 50,
        noise: float = 0.4,
        seed: int = 42,
        true_intercept: float = 0.6,
        true_slope: float = 0.52,
        show_true_line: bool = True,
    ) -> PipelineResult:
        """
        Führt die komplette Pipeline für die einfache Regression aus.
        
        Args:
            dataset: Name des Datensatzes
            n: Stichprobengröße
            noise: Rausch-Niveau
            seed: Zufalls-Seed
            true_intercept: Wahrer Achsenabschnitt (für synthetische Daten)
            true_slope: Wahre Steigung (für synthetische Daten)
            show_true_line: Wahre Regressionslinie anzeigen
        
        Returns:
            PipelineResult mit Daten, Statistiken und Plots
        """
        logger.info(f"Starte einfache Regressions-Pipeline: {dataset}, n={n}")
        
        # Schritt 1: DATEN ABRUFEN
        data = self.fetcher.get_simple(
            dataset=dataset,
            n=n,
            noise=noise,
            seed=seed,
            true_intercept=true_intercept,
            true_slope=true_slope,
        )
        
        # Schritt 2: BERECHNEN
        stats = self.calculator.simple_regression(data.x, data.y)
        
        # Schritt 3: PLOTS ERSTELLEN
        plots = self.plotter.simple_regression_plots(
            data=data,
            result=stats,
            show_true_line=show_true_line,
            true_intercept=true_intercept,
            true_slope=true_slope,
        )
        
        params = {
            "dataset": dataset,
            "n": n,
            "noise": noise,
            "seed": seed,
            "true_intercept": true_intercept,
            "true_slope": true_slope,
            "show_true_line": show_true_line,
        }
        
        return PipelineResult(
            data=data,
            stats=stats,
            plots=plots,
            pipeline_type="simple",
            params=params,
        )
    
    def run_multiple(
        self,
        dataset: str = "cities",
        n: int = 75,
        noise: float = 3.5,
        seed: int = 42,
    ) -> PipelineResult:
        """
        Führt die komplette Pipeline für die multiple Regression aus.
        """
        logger.info(f"Starte multiple Regressions-Pipeline: {dataset}, n={n}")
        
        # Schritt 1: DATEN ABRUFEN
        data = self.fetcher.get_multiple(
            dataset=dataset,
            n=n,
            noise=noise,
            seed=seed,
        )
        
        # Schritt 2: BERECHNEN
        stats = self.calculator.multiple_regression(data.x1, data.x2, data.y)
        
        # Schritt 3: PLOTS ERSTELLEN
        plots = self.plotter.multiple_regression_plots(data=data, result=stats)
        
        params = {
            "dataset": dataset,
            "n": n,
            "noise": noise,
            "seed": seed,
        }
        
        return PipelineResult(
            data=data,
            stats=stats,
            plots=plots,
            pipeline_type="multiple",
            params=params,
        )
    
    def display(
        self,
        result: PipelineResult,
        show_formulas: bool = True,
    ) -> None:
        """
        Schritt 4: DISPLAY - Rendert die Ergebnisse in Streamlit.
        """
        if result.pipeline_type == "simple":
            self.renderer.simple_regression(
                data=result.data,
                result=result.stats,
                plots=result.plots,
                show_formulas=show_formulas,
            )
        else:
            self.renderer.multiple_regression(
                data=result.data,
                result=result.stats,
                plots=result.plots,
                show_formulas=show_formulas,
            )
    
    # =========================================================
    # Hilfsmethoden für einzelne Schritte
    # =========================================================
    
    def get_data(self, regression_type: str = "simple", **kwargs):
        """Schritt 1: Nur Daten abrufen."""
        if regression_type == "simple":
            return self.fetcher.get_simple(**kwargs)
        else:
            return self.fetcher.get_multiple(**kwargs)
    
    def calculate(self, data, regression_type: str = "simple"):
        """Schritt 2: Nur Berechnungen ausführen."""
        if regression_type == "simple":
            return self.calculator.simple_regression(data.x, data.y)
        else:
            return self.calculator.multiple_regression(data.x1, data.x2, data.y)
    
    def plot(self, data, stats, regression_type: str = "simple", **kwargs):
        """Schritt 3: Nur Plots erstellen."""
        if regression_type == "simple":
            return self.plotter.simple_regression_plots(data, stats, **kwargs)
        else:
            return self.plotter.multiple_regression_plots(data, stats)
